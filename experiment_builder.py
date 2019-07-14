from abc import ABC
import sys
import pickle
import pathlib

import gc

import numpy as np
from parameter_config import ParameterConfig
from models import MultiCompartmentModel
from helpers import UniformInitialiser, ConstantInitialiser, create_plot, visualise_mnist, \
    show_plots, load_model, compute_non_linear_transform, create_transfer_function, remove_directory
from dynamics_simulator import StandardDynamicsSimulator, SimplifiedDynamicsSimulator
from data_streams import InputOutputStream, CompositeStream, ConstantStream, CyclingStream, MNISTInputOutputStream, \
    SmoothStream, NoneStream
from monitors import MonitorBuilder, ExponentialAverageMonitor
import argparse


class Experiment:
    def __init__(self, config, experiment_name, model_file=None):
        self.config = config
        self.experiment_name = experiment_name

        self.network_architecture = self.config['network_architecture']
        self.input_size = self.network_architecture['input_size']
        self.output_size = self.network_architecture['output_size']
        self.layer_sizes = self.network_architecture['layer_sizes']

        self.weight_inititalision = self.config['weight_intialisation']
        self.weight_inititalision_type = self.weight_inititalision['type']

        if self.weight_inititalision_type == 'uniform':
            lower_bound = self.weight_inititalision['lower_bound']
            upper_bound = self.weight_inititalision['upper_bound']
            self.feedforward_weight_inititaliser = UniformInitialiser(lower_bound, upper_bound)
            self.predict_weight_inititaliser = UniformInitialiser(lower_bound, upper_bound)
            self.interneuron_weight_inititaliser = UniformInitialiser(lower_bound, upper_bound)
            self.feedback_weight_inititaliser = UniformInitialiser(lower_bound, upper_bound)

        else:
            raise Exception("Invalid weight initialisation type specified.")

        self.feedforward_learning_rates = self.config['feedforward_learning_rates']
        self.predict_learning_rates = self.config['predict_learning_rates']
        self.interneuron_learning_rates = self.config['interneuron_learning_rates']
        self.feedback_learning_rates = self.config['feedback_learning_rates']

        if model_file is None:
            self.model = MultiCompartmentModel(self.input_size,
                                               self.layer_sizes,
                                               self.output_size,
                                               self.feedforward_weight_inititaliser,
                                               self.predict_weight_inititaliser,
                                               self.interneuron_weight_inititaliser,
                                               self.feedback_weight_inititaliser,
                                               self.feedforward_learning_rates,
                                               self.predict_learning_rates,
                                               self.interneuron_learning_rates,
                                               self.feedback_learning_rates)

            print("Created {}".format(str(self.model)))
        else:
            self.model = load_model(model_file)
            print("Loaded {}".format(str(self.model)))

        self.dynamics = self.config['dynamics']
        self.dynamics_simulator = None
        self.input_output_stream = None
        self.monitors = []
        self.monitor_frequency = 100

    def set_input_output_stream(self, input_output_stream):
        self.input_output_stream = input_output_stream

    def initialise_dynamics_simulator(self):
        if self.dynamics['type'] == 'standard':
            self.dynamics_simulator = StandardDynamicsSimulator(self.model, self.input_output_stream, self.dynamics,
                                                                self.monitors)
        elif self.dynamics['type'] == 'simplified':
            self.dynamics_simulator = SimplifiedDynamicsSimulator(self.model, self.input_output_stream, self.dynamics,
                                                                  self.monitors)
        else:
            raise Exception("Invalid dynamics type: {}".format({self.dynamics['type']}))

    def add_monitors(self):
        self.monitors += [MonitorBuilder.create_weight_diff_monitor('feedforward_predict_weight_difference',
                                                                    self.model, 0, 'feedforward_predict_diff',
                                                                    update_frequency=self.monitor_frequency)]

        self.monitors += [MonitorBuilder.create_weight_diff_monitor('feedback_interneuron_weight_difference',
                                                                    self.model, 0, 'feedback_interneuron_diff',
                                                                    update_frequency=self.monitor_frequency)]

        self.monitors += [MonitorBuilder.create_weight_angle_monitor('feedforward_feedback_weight_angle',
                                                                     self.model, 0, 'feedforward_feedback_angle',
                                                                     update_frequency=self.monitor_frequency)]

        self.monitors += [MonitorBuilder.create_weight_angle_monitor('feedforward_predict_weight_angle',
                                                                     self.model, 0, 'feedforward_predict_angle',
                                                                     update_frequency=self.monitor_frequency)]
        self.monitors += [MonitorBuilder.create_weight_angle_monitor('feedback_interneuron_weight_angle',
                                                                     self.model, 0, 'feedback_interneuron_angle',
                                                                     update_frequency=self.monitor_frequency)]

        self.monitors += [MonitorBuilder.create_weight_monitor('individual_first_layer_feedforward_weight', self.model,
                                                               'feedforward_weights', 0, 0, 0,
                                                               update_frequency=self.monitor_frequency)]

        self.monitors += [MonitorBuilder.create_weight_monitor('individual_second_layer_feedforward_weight', self.model,
                                                               'feedforward_weights', 1, 0, 0,
                                                               update_frequency=self.monitor_frequency)]

        self.monitors += [MonitorBuilder.create_weight_monitor('individual_predict_weight', self.model,
                                                               'predict_weights', 0, 0, 0,
                                                               update_frequency=self.monitor_frequency)]

        self.monitors += [MonitorBuilder.create_weight_monitor('individual_interneuron_weights', self.model,
                                                               'interneuron_weights', 0, 0, 0,
                                                               update_frequency=self.monitor_frequency)]

        self.monitors += [MonitorBuilder.create_weight_monitor('individual_feedback_weights', self.model,
                                                               'feedback_weights', 0, 0, 0,
                                                               update_frequency=self.monitor_frequency)]

        self.monitors += [MonitorBuilder.create_potential_monitor('individual_first_layer_pyramidal_basal_potential',
                                                                  self.model, 0, 'pyramidal_basal', 0,
                                                                  update_frequency=self.monitor_frequency)]
        self.monitors += [MonitorBuilder.create_potential_monitor('individual_first_layer_pyramidal_soma_potential',
                                                                  self.model, 0, 'pyramidal_soma', 0,
                                                                  update_frequency=self.monitor_frequency)]
        self.monitors += [MonitorBuilder.create_potential_monitor('individual_first_layer_pyramidal_apical_potential',
                                                                  self.model, 0, 'pyramidal_apical', 0,
                                                                  update_frequency=self.monitor_frequency)]

        self.monitors += [MonitorBuilder.create_potential_monitor('individual_first_layer_interneuron_basal_potential',
                                                                  self.model, 0, 'interneuron_basal', 0,
                                                                  update_frequency=self.monitor_frequency)]
        self.monitors += [MonitorBuilder.create_potential_monitor('individual_first_layer_interneuron_soma_potential',
                                                                  self.model, 0, 'interneuron_soma', 0,
                                                                  update_frequency=self.monitor_frequency)]

        # self.monitors += [MonitorBuilder.create_pyramidal_basal_soma_rate_diff_monitor('', self.model, 0, 0,
        #                                                                               self.dynamics,
        #                                                                               update_frequency=
        #                                                                               self.monitor_frequency)]

        self.monitors += [MonitorBuilder.create_potential_monitor('individual_second_layer_pyramidal_basal_potential',
                                                                  self.model, 1, 'pyramidal_basal', 0,
                                                                  update_frequency=self.monitor_frequency)]
        self.monitors += [MonitorBuilder.create_potential_monitor('individual_second_layer_pyramidal_soma_potential',
                                                                  self.model, 1, 'pyramidal_soma', 0,
                                                                  update_frequency=self.monitor_frequency)]

        self.monitors += [MonitorBuilder.create_data_monitor('individual_input', self.input_output_stream, 'input', 0,
                                                             update_frequency=self.monitor_frequency)]

        self.monitors += [MonitorBuilder.create_data_monitor('individual_target', self.input_output_stream, 'target', 0,
                                                             update_frequency=self.monitor_frequency)]

        self.monitors += [ExponentialAverageMonitor(
            MonitorBuilder.create_error_monitor('sum_squares_error', self.model, self.input_output_stream,
                                                'sum_squares_error', self.dynamics,
                                                update_frequency=self.monitor_frequency), 100000)]

    def start_experiment(self, num_epochs, num_epoch_iterations, test_phase_length, resume_from_epoch=None,
                         show_final_plots=False):
        state_save_folder = 'experiment_results/{}'.format(self.experiment_name)

        for i in range(num_epochs):
            epoch_index = i + 1
            if resume_from_epoch is not None and epoch_index <= resume_from_epoch:
                continue
            self.dynamics_simulator.run_simulation(num_epoch_iterations * epoch_index)
            new_state_save_location = state_save_folder+'/epoch_{}'.format(epoch_index)
            if epoch_index > 1:
                prev_state_save_location = state_save_folder+'/epoch_{}'.format(epoch_index - 1)
                self.save_state(new_state_save_location=new_state_save_location,
                                prev_state_save_location=prev_state_save_location)
                remove_directory(location=prev_state_save_location)
            else:
                self.save_state(new_state_save_location=new_state_save_location)
            gc.collect()

        self.dynamics_simulator.set_testing_phase(True)
        self.dynamics_simulator.run_simulation(num_epochs * num_epoch_iterations + test_phase_length)

        self.save_state(new_state_save_location=state_save_folder+'/test',
                        prev_state_save_location=state_save_folder+'/epoch_{}'.format(num_epochs),
                        show_generated_plots=show_final_plots)

    def save_state(self, new_state_save_location, prev_state_save_location=None,
                   show_generated_plots=False):
        monitor_save_location = '{}/monitors'.format(new_state_save_location)
        pathlib.Path(monitor_save_location).mkdir(parents=True, exist_ok=True)

        for monitor in self.monitors:
            monitor_name = monitor.get_name()

            if prev_state_save_location is not None:
                prev_monitor_save_location = '{}/monitors/{}.pkl'.format(prev_state_save_location, monitor_name)
                monitor.prepend_data(prev_monitor_save_location)

            monitor.plot_values(save_location=new_state_save_location)#, close_plot=not show_generated_plots)

            new_monitor_save_location = '{}/{}.pkl'.format(monitor_save_location, monitor_name)
            monitor.save_data(new_monitor_save_location)
            monitor.clear_data()

        self.dynamics_simulator.save_model(save_location=new_state_save_location, name=self.experiment_name)
        if show_generated_plots:
            show_plots()

    def load_state(self, resume_from_epoch, num_epoch_iterations):
        if resume_from_epoch is not None:
            state_folder = 'experiment_results/{}/epoch_{}'.format(self.experiment_name, resume_from_epoch)
            self.dynamics_simulator.set_iteration_number(resume_from_epoch * num_epoch_iterations)
            del self.model
            self.model = load_model(state_folder + '/' + self.experiment_name + '.pkl')
            self.initialise_dynamics_simulator()
            resume_from_iterations = resume_from_epoch * num_epoch_iterations
            self.dynamics_simulator.set_iteration_number(resume_from_iterations)


class ExperimentBuilder:
    @staticmethod
    def create_target_network_experiment(config, experiment_name, train_data_path, test_data_path, example_iterations,
                 self_predict_phase_length, training_phase_length, test_phase_length, model_file=None):
        experiment = Experiment(config=config, experiment_name=experiment_name, model_file=model_file)
        transfer_function = create_transfer_function(experiment.dynamics['transfer_function'])

        target_netork_forward_weights_list = []
        # forward_weights_list += [np.random.uniform(-1, 1, (30, 50))]
        # forward_weights_list += [np.random.uniform(-1, 1, (50, 10))]
        target_netork_forward_weights_list += [np.load('./target_network_weights/first_layer_feedforward_weights.npy').copy()]
        target_netork_forward_weights_list += [np.load('./target_network_weights/second_layer_feedforward_weights.npy').copy()]
        print("Target Network layer 1 weights: {}".format(target_netork_forward_weights_list[0]))
        print("Target Network layer 2 weights: {}".format(target_netork_forward_weights_list[1]))

        # input_sequence = np.random.uniform(-1, 1, (num_train_examples, self.input_size))
        input_sequence = np.load(train_data_path)
        output_sequence = compute_non_linear_transform(input_sequence, transfer_function,
                                                       target_netork_forward_weights_list)

        if test_phase_length > 0:
            # test_input_sequence = np.random.uniform(-1, 1, (num_test_examples, self.input_size))
            test_input_sequence = np.load(test_data_path)

            test_output_sequence = compute_non_linear_transform(test_input_sequence, transfer_function,
                                                                target_netork_forward_weights_list)

            input_stream = CompositeStream([CyclingStream((experiment.input_size, 1), input_sequence, example_iterations),
                                        CyclingStream((experiment.input_size, 1), input_sequence, example_iterations),
                                        CyclingStream((experiment.input_size, 1), test_input_sequence, example_iterations)],
                                       [0, self_predict_phase_length, self_predict_phase_length + training_phase_length])

            output_stream = CompositeStream([NoneStream((experiment.output_size, 1)),
                                             SmoothStream(CyclingStream((experiment.output_size, 1), output_sequence,
                                                                        example_iterations), 30),
                                             SmoothStream(CyclingStream((experiment.output_size, 1), test_output_sequence,
                                                                        example_iterations), 30)
                                             ],
                                            [0, self_predict_phase_length,
                                             self_predict_phase_length + training_phase_length])
        else:
            input_stream = CompositeStream([CyclingStream((experiment.input_size, 1), input_sequence, example_iterations),
                                            CyclingStream((experiment.input_size, 1), input_sequence, example_iterations)],
                                           [0, self_predict_phase_length])
            output_stream = CompositeStream([NoneStream((experiment.output_size, 1)),
                                            SmoothStream(CyclingStream((experiment.output_size, 1), output_sequence,
                                                                       example_iterations), 30)],
                                            [0, self_predict_phase_length])

        experiment.set_input_output_stream(InputOutputStream(input_stream, output_stream))
        experiment.initialise_dynamics_simulator()
        return experiment

    @staticmethod
    def create_xor_experiment(config, experiment_name, train_data_path, example_iterations,
                 self_predict_phase_length, model_file=None):
        experiment = Experiment(config=config, experiment_name=experiment_name, model_file=model_file)
        # input_sequence = [[0.1, 0.1, 0.8], [0.1, 0.8, 0.8], [0.8, 0.1, 0.8], [0.8, 0.8, 0.8]]
        # output_sequence = [0.1, 0.8, 0.8, 0.1]

        input_sequence = np.load(train_data_path + 'input_sequence')
        output_sequence = np.load(train_data_path + 'output_sequence')
        # input_stream = CompositeStream([CyclingStream((self.input_size, 1), input_sequence, example_iterations),

        random_input_sequence = np.random.uniform(-1.0, 1.0, (500, experiment.input_size))
        input_stream = CompositeStream([CyclingStream((experiment.input_size, 1), random_input_sequence, example_iterations),
                                        CyclingStream((experiment.input_size, 1), input_sequence, example_iterations)],
                                       [0, self_predict_phase_length])

        output_stream = CompositeStream([NoneStream((experiment.output_size, 1)),
                                         SmoothStream(CyclingStream((experiment.output_size, 1), output_sequence,
                                                                    example_iterations), 30)],
                                        [0, self_predict_phase_length])
        experiment.set_input_output_stream(InputOutputStream(input_stream, output_stream))
        experiment.initialise_dynamics_simulator()
        return experiment

    @staticmethod
    def create_mnist_experiment(config, experiment_name, train_data_path, training_phase_length, model_file=None):
        experiment = Experiment(config=config, experiment_name=experiment_name, model_file=model_file)

        experiment.set_input_output_stream(MNISTInputOutputStream(train_data_path+'train_images.idx3-ubyte',
                                                                  train_data_path+'train_labels.idx1-ubyte',
                                                                  training_phase_length))
        experiment.initialise_dynamics_simulator()
        return experiment


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    input_args = sys.argv[1:]

    parser.add_argument('experiment_name', help='Name of the experiment.')
    parser.add_argument('parameter_config_file', default=None,
                        help='Parameter configuration file.')

    parser.add_argument('num_epochs', type=int, help='Number of epochs to train for.')
    parser.add_argument('num_epoch_iterations', type=int,
                        help='Number of iterations to step through the dynamics per epoch.')

    parser.add_argument('-self_predict_phase_length', type=int, default=0,
                        help='Length of initial self-prediction stage where no output targets are given')
    parser.add_argument('-test_phase_length', type=int, default=0,
                        help='Length of the final test phase in iteration steps.')
    parser.add_argument('-example_iterations', type=int, default=1000,
                        help='Number of iterations that each training example is presented for.')

    parser.add_argument('-model_file', default=None, help='Model file to load initial state from.')
    parser.add_argument('-train_data_path', type=str, help='Location of the training data.')
    parser.add_argument('-test_data_path', type=str, default=None, help='Location of the test data.')
    parser.add_argument('-resume_from_epoch', type=int, default=None, help='Epoch to resume experiment from.')

    parser.add_argument('-show_final_plots', type=bool, default=False,
                        help='Show the final plots at the end.')

    args = parser.parse_args()

    if len(input_args) > 0:
        experiment_name = input_args[0]
    else:
        raise Exception("No experiment name provided.")

    if args.parameter_config_file is not None:
        parameter_config = ParameterConfig(args.parameter_config_file)
        print("Using parameter configuration: {}".format(args.parameter_config_file))
    else:
        print("Using default parameter configuration.")
        parameter_config = ParameterConfig()

    training_phase_length = args.num_epochs * args.num_epoch_iterations

    if experiment_name.startswith('xor'):
        experiment = ExperimentBuilder.create_xor_experiment(config=parameter_config,
                                                             experiment_name=experiment_name,
                                                             train_data_path=args.train_data_path,
                                                             example_iterations=args.example_iterations,
                                                             self_predict_phase_length=args.self_predict_phase_length,
                                                             model_file=args.model_file)

    elif experiment_name.startswith('target_network'):
        experiment = ExperimentBuilder.create_target_network_experiment(config=parameter_config,
                                                                        experiment_name=experiment_name,
                                                                        train_data_path=args.train_data_path,
                                                                        test_data_path=args.test_data_path,
                                                                        example_iterations=args.example_iterations,
                                                                        self_predict_phase_length=
                                                                        args.self_predict_phase_length,
                                                                        training_phase_length=training_phase_length,
                                                                        test_phase_length=args.test_phase_length,
                                                                        model_file=args.model_file)

    elif experiment_name.startswith('mnist'):
        experiment = ExperimentBuilder.create_mnist_experiment(config=parameter_config,
                                                               experiment_name=experiment_name,
                                                               train_data_path=args.train_data_path,
                                                               training_phase_length=training_phase_length,
                                                               model_file=args.model_file)
    else:
        raise Exception("Invalid experiment name: {}".format(experiment_name))

    if args.resume_from_epoch is not None:
        experiment.load_state(resume_from_epoch=args.resume_from_epoch,
                              num_epoch_iterations=args.num_epoch_iterations)
    experiment.add_monitors()
    experiment.start_experiment(num_epochs=args.num_epochs, num_epoch_iterations=args.num_epoch_iterations,
                                test_phase_length=args.test_phase_length,
                                resume_from_epoch=args.resume_from_epoch,
                                show_final_plots=args.show_final_plots)

