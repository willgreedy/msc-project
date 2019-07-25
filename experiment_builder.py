import sys
import pathlib

import numpy as np
from parameter_config import ParameterConfig
from models import MultiCompartmentModel
from helpers import UniformInitialiser, \
    show_plots, load_model, compute_non_linear_transform, create_transfer_function, remove_directory, \
    get_target_network_forward_weights_list, read_monitoring_values_config_file
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
        self.hidden_layer_sizes = self.network_architecture['hidden_layer_sizes']

        self.weight_inititalision = self.config['weight_intialisation']
        self.weight_inititalision_type = self.weight_inititalision['type']
        self.init_self_predicting_weights = self.weight_inititalision['self_predicting']

        if self.weight_inititalision_type == 'uniform':
            lower_bound = self.weight_inititalision['lower_bound']
            upper_bound = self.weight_inititalision['upper_bound']
            self.feedforward_weight_inititaliser = UniformInitialiser(lower_bound, upper_bound)
            if self.init_self_predicting_weights:
                self.predict_weight_inititaliser = None
                self.feedback_weight_inititaliser = None
                self.interneuron_weight_inititaliser = None
            else:
                self.predict_weight_inititaliser = UniformInitialiser(lower_bound, upper_bound)
                self.feedback_weight_inititaliser = UniformInitialiser(lower_bound, upper_bound)
                self.interneuron_weight_inititaliser = UniformInitialiser(lower_bound, upper_bound)
        else:
            raise Exception("Invalid weight initialisation type specified.")

        self.feedforward_learning_rates = self.config['feedforward_learning_rates']
        self.predict_learning_rates = self.config['predict_learning_rates']
        self.interneuron_learning_rates = self.config['interneuron_learning_rates']
        self.feedback_learning_rates = self.config['feedback_learning_rates']

        if model_file is None:
            self.model = MultiCompartmentModel(self.input_size,
                                               self.hidden_layer_sizes,
                                               self.output_size,
                                               self.feedforward_weight_inititaliser,
                                               self.predict_weight_inititaliser,
                                               self.interneuron_weight_inititaliser,
                                               self.feedback_weight_inititaliser,
                                               self.feedforward_learning_rates,
                                               self.predict_learning_rates,
                                               self.interneuron_learning_rates,
                                               self.feedback_learning_rates,
                                               self.init_self_predicting_weights)

            print("Created {}".format(str(self.model)))
        else:
            self.model = load_model(model_file)
            print("Loaded {}".format(str(self.model)))

        self.dynamics = self.config['dynamics']
        self.dynamics_simulator = None
        self.input_output_stream = None

        self.monitor_frequency = 100

        self.orig_model_file = model_file
        self.monitors = []

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

    def add_default_monitors(self, monitor_name):
        if monitor_name == 'layer_1_feedforward_predict_weight_difference':
            self.monitors += [
                MonitorBuilder.create_weight_diff_monitor('layer_1_feedforward_predict_weight_difference',
                                                          self.model, 0, 'feedforward_predict_diff',
                                                          update_frequency=self.monitor_frequency)]

        elif monitor_name == 'layer_1_feedback_interneuron_weight_difference':
            self.monitors += [
                MonitorBuilder.create_weight_diff_monitor('layer_1_feedback_interneuron_weight_difference',
                                                          self.model, 0, 'feedback_negative_interneuron_diff',
                                                          update_frequency=self.monitor_frequency)]

        elif monitor_name == 'layer_1_feedforward_feedback_weight_angle':
            self.monitors += [
                MonitorBuilder.create_weight_angle_monitor('layer_1_feedforward_feedback_weight_angle',
                                                           self.model, 0, 'feedforward_feedback_angle',
                                                           update_frequency=self.monitor_frequency)]

        elif monitor_name == 'layer_1_feedforward_predict_weight_angle':
            self.monitors += [
                MonitorBuilder.create_weight_angle_monitor('layer_1_feedforward_predict_weight_angle',
                                                           self.model, 0, 'feedforward_predict_angle',
                                                           update_frequency=self.monitor_frequency)]

        elif monitor_name == 'layer_1_feedback_interneuron_weight_angle':
            self.monitors += [
                MonitorBuilder.create_weight_angle_monitor('layer_1_feedback_interneuron_weight_angle',
                                                           self.model, 0, 'feedback_interneuron_angle',
                                                           update_frequency=self.monitor_frequency)]

        elif monitor_name == 'layer_2_feedforward_predict_weight_difference':
            self.monitors += [
                MonitorBuilder.create_weight_diff_monitor('layer_2_feedforward_predict_weight_difference',
                                                          self.model, 1, 'feedforward_predict_diff',
                                                          update_frequency=self.monitor_frequency)]

        elif monitor_name == 'layer_2_feedback_interneuron_weight_difference':
            self.monitors += [
                MonitorBuilder.create_weight_diff_monitor('layer_2_feedback_interneuron_weight_difference',
                                                          self.model, 1, 'feedback_interneuron_diff',
                                                          update_frequency=self.monitor_frequency)]

        elif monitor_name == 'layer_2_feedforward_feedback_weight_angle':
            self.monitors += [
                MonitorBuilder.create_weight_angle_monitor('layer_2_feedforward_feedback_weight_angle',
                                                           self.model, 1, 'feedforward_feedback_angle',
                                                           update_frequency=self.monitor_frequency)]

        elif monitor_name == 'layer_2_feedforward_predict_weight_angle':
            self.monitors += [
                MonitorBuilder.create_weight_angle_monitor('layer_2_feedforward_predict_weight_angle',
                                                           self.model, 1, 'feedforward_predict_angle',
                                                           update_frequency=self.monitor_frequency)]

        elif monitor_name == 'layer_2_feedback_interneuron_weight_angle':
            self.monitors += [
                MonitorBuilder.create_weight_angle_monitor('layer_2_feedback_interneuron_weight_angle',
                                                           self.model, 1, 'feedback_interneuron_angle',
                                                           update_frequency=self.monitor_frequency)]

        elif monitor_name == 'layer_1_mean_feedforward_weight_magnitudes':
            self.monitors += [
                MonitorBuilder.create_weight_layer_monitor('layer_1_mean_feedforward_weight_magnitudes',
                                                           self.model, 'feedforward_weights', 0,
                                                           operation='mean-magnitude', update_frequency=
                                                           self.monitor_frequency)]

        elif monitor_name == 'layer_2_mean_feedforward_weight_magnitudes':
            self.monitors += [
                MonitorBuilder.create_weight_layer_monitor('layer_2_mean_feedforward_weight_magnitudes',
                                                           self.model, 'feedforward_weights', 1,
                                                           operation='mean-magnitude', update_frequency=
                                                           self.monitor_frequency)]

        elif monitor_name == 'layer_3_mean_feedforward_weight_magnitudes':
            self.monitors += [
                MonitorBuilder.create_weight_layer_monitor('layer_3_mean_feedforward_weight_magnitudes',
                                                           self.model, 'feedforward_weights', 2,
                                                           operation='mean-magnitude', update_frequency=
                                                           self.monitor_frequency)]

        elif monitor_name == 'layer_1_std_feedforward_weight_magnitudes':
            self.monitors += [
                MonitorBuilder.create_weight_layer_monitor('layer_1_std_feedforward_weight_magnitudes',
                                                           self.model, 'feedforward_weights', 0,
                                                           operation='std-magnitude', update_frequency=
                                                           self.monitor_frequency)]

        elif monitor_name == 'layer_2_std_feedforward_weight_magnitudes':
            self.monitors += [
                MonitorBuilder.create_weight_layer_monitor('layer_2_std_feedforward_weight_magnitudes',
                                                           self.model, 'feedforward_weights', 1,
                                                           operation='std-magnitude', update_frequency=
                                                           self.monitor_frequency)]

        elif monitor_name == 'layer_3_std_feedforward_weight_magnitudes':
            self.monitors += [
                MonitorBuilder.create_weight_layer_monitor('layer_3_std_feedforward_weight_magnitudes',
                                                           self.model, 'feedforward_weights', 2,
                                                           operation='std-magnitude', update_frequency=
                                                           self.monitor_frequency)]

        elif monitor_name == 'layer_1_mean_feedforward_weight_magnitude_changes':
            if self.orig_model_file is None:
                raise Exception('Must provide original model file when using weight magnitude monitors')
            orig_model = load_model(self.orig_model_file)
            self.monitors += [
                MonitorBuilder.create_weight_layer_monitor('layer_1_mean_feedforward_weight_magnitude_changes',
                                                           self.model, 'feedforward_weights', 0,
                                                           operation='mean-magnitude-change', orig_model=orig_model,
                                                           update_frequency=self.monitor_frequency)]

        elif monitor_name == 'layer_2_mean_feedforward_weight_magnitude_changes':
            if self.orig_model_file is None:
                raise Exception('Must provide original model file when using weight magnitude monitors')
            orig_model = load_model(self.orig_model_file)
            self.monitors += [
                MonitorBuilder.create_weight_layer_monitor('layer_2_mean_feedforward_weight_magnitude_changes',
                                                           self.model, 'feedforward_weights', 1,
                                                           operation='mean-magnitude-change', orig_model=orig_model,
                                                           update_frequency=self.monitor_frequency)]

        elif monitor_name == 'layer_3_mean_feedforward_weight_magnitude_changes':
            if self.orig_model_file is None:
                raise Exception('Must provide original model file when using weight magnitude monitors')
            orig_model = load_model(self.orig_model_file)
            self.monitors += [
                MonitorBuilder.create_weight_layer_monitor('layer_3_mean_feedforward_weight_magnitude_changes',
                                                           self.model, 'feedforward_weights', 2,
                                                           operation='mean-magnitude-change', orig_model=orig_model,
                                                           update_frequency=self.monitor_frequency)]

        elif monitor_name == 'layer_1_individual_feedforward_weight':
            self.monitors += [
                MonitorBuilder.create_weight_monitor('layer_1_individual_feedforward_weight', self.model,
                                                     'feedforward_weights', 0, 0, 0,
                                                     update_frequency=self.monitor_frequency)]

        elif monitor_name == 'layer_2_individual_feedforward_weight':
            self.monitors += [
                MonitorBuilder.create_weight_monitor('layer_2_individual_feedforward_weight', self.model,
                                                     'feedforward_weights', 1, 0, 0,
                                                     update_frequency=self.monitor_frequency)]

        elif monitor_name == 'layer_1_individual_predict_weight':
            self.monitors += [
                MonitorBuilder.create_weight_monitor('layer_1_individual_predict_weight', self.model,
                                                     'predict_weights', 0, 0, 0,
                                                     update_frequency=self.monitor_frequency)]
        elif monitor_name == 'layer_1_individual_interneuron_weights':
            self.monitors += [
                MonitorBuilder.create_weight_monitor('layer_1_individual_interneuron_weights', self.model,
                                                     'interneuron_weights', 0, 0, 0,
                                                     update_frequency=self.monitor_frequency)]

        elif monitor_name == 'layer_1_individual_feedback_weights':
            self.monitors += [
                MonitorBuilder.create_weight_monitor('layer_1_individual_feedback_weights', self.model,
                                                     'feedback_weights', 0, 0, 0,
                                                     update_frequency=self.monitor_frequency)]

        elif monitor_name == 'layer_1_individual_pyramidal_basal_potential':
            self.monitors += [
                MonitorBuilder.create_potential_monitor('layer_1_individual_pyramidal_basal_potential',
                                                        self.model, 0, 'pyramidal_basal', 0,
                                                        update_frequency=self.monitor_frequency)]

        elif monitor_name == 'layer_1_individual_pyramidal_soma_potential':
            self.monitors += [
                MonitorBuilder.create_potential_monitor('layer_1_individual_pyramidal_soma_potential',
                                                        self.model, 0, 'pyramidal_soma', 0,
                                                        update_frequency=self.monitor_frequency)]

        elif monitor_name == 'layer_1_individual_pyramidal_apical_potential':
            self.monitors += [
                MonitorBuilder.create_potential_monitor('layer_1_individual_pyramidal_apical_potential',
                                                        self.model, 0, 'pyramidal_apical', 0,
                                                        update_frequency=self.monitor_frequency)]

        elif monitor_name == 'layer_1_individual_interneuron_basal_potential':
            self.monitors += [
                MonitorBuilder.create_potential_monitor('layer_1_individual_interneuron_basal_potential',
                                                        self.model, 0, 'interneuron_basal', 0,
                                                        update_frequency=self.monitor_frequency)]

        elif monitor_name == 'layer_1_individual_interneuron_soma_potential':
            self.monitors += [
                MonitorBuilder.create_potential_monitor('layer_1_individual_interneuron_soma_potential',
                                                        self.model, 0, 'interneuron_soma', 0,
                                                        update_frequency=self.monitor_frequency)]

        elif monitor_name == 'layer_2_individual_pyramidal_basal_potential':
            self.monitors += [
                MonitorBuilder.create_potential_monitor('layer_2_individual_pyramidal_basal_potential',
                                                        self.model, 1, 'pyramidal_basal', 0,
                                                        update_frequency=self.monitor_frequency)]

        elif monitor_name == 'layer_2_individual_pyramidal_soma_potential':
            self.monitors += [
                MonitorBuilder.create_potential_monitor('layer_2_individual_pyramidal_soma_potential',
                                                        self.model, 1, 'pyramidal_soma', 0,
                                                        update_frequency=self.monitor_frequency)]

        elif monitor_name == 'layer_2_individual_pyramidal_apical_potential':
            self.monitors += [
                MonitorBuilder.create_potential_monitor('layer_2_individual_pyramidal_apical_potential',
                                                        self.model, 1, 'pyramidal_apical', 0,
                                                        update_frequency=self.monitor_frequency)]

        elif monitor_name == 'layer_2_individual_interneuron_basal_potential':
            self.monitors += [
                MonitorBuilder.create_potential_monitor('layer_2_individual_interneuron_basal_potential',
                                                        self.model, 1, 'interneuron_basal', 0,
                                                        update_frequency=self.monitor_frequency)]

        elif monitor_name == 'layer_2_individual_interneuron_soma_potential':
            self.monitors += [
                MonitorBuilder.create_potential_monitor('layer_2_individual_interneuron_soma_potential',
                                                        self.model, 1, 'interneuron_soma', 0,
                                                        update_frequency=self.monitor_frequency)]

        elif monitor_name == 'layer_3_individual_pyramidal_basal_potential':
            self.monitors += [
                MonitorBuilder.create_potential_monitor('layer_3_individual_pyramidal_basal_potential',
                                                        self.model, 2, 'pyramidal_basal', 0,
                                                        update_frequency=self.monitor_frequency)]

        elif monitor_name == 'layer_3_individual_pyramidal_soma_potential':
            self.monitors += [
                MonitorBuilder.create_potential_monitor('layer_3_individual_pyramidal_soma_potential',
                                                        self.model, 2, 'pyramidal_soma', 0,
                                                        update_frequency=self.monitor_frequency)]

        elif monitor_name == 'layer_3_individual_pyramidal_apical_potential':
            self.monitors += [
                MonitorBuilder.create_potential_monitor('layer_3_individual_pyramidal_apical_potential',
                                                        self.model, 2, 'pyramidal_apical', 0,
                                                        update_frequency=self.monitor_frequency)]

        elif monitor_name == 'layer_3_individual_interneuron_basal_potential':
            self.monitors += [
                MonitorBuilder.create_potential_monitor('layer_3_individual_interneuron_basal_potential',
                                                        self.model, 2, 'interneuron_basal', 0,
                                                        update_frequency=self.monitor_frequency)]

        elif monitor_name == 'layer_3_individual_interneuron_soma_potential':
            self.monitors += [
                MonitorBuilder.create_potential_monitor('layer_3_individual_interneuron_soma_potential',
                                                        self.model, 2, 'interneuron_soma', 0,
                                                        update_frequency=self.monitor_frequency)]

        elif monitor_name == 'individual_input_value':
            self.monitors += [
                MonitorBuilder.create_data_monitor('individual_input_value', self.input_output_stream, 'input', 0,
                                                   update_frequency=self.monitor_frequency)]

        elif monitor_name == 'individual_target_value':
            self.monitors += [
                MonitorBuilder.create_data_monitor('individual_target_value', self.input_output_stream, 'target', 0,
                                                   update_frequency=self.monitor_frequency)]

        elif monitor_name == 'sum_squares_error':
            self.monitors += [ExponentialAverageMonitor(
                MonitorBuilder.create_error_monitor('sum_squares_error', self.model, self.input_output_stream,
                                                    'sum_squares_error', self.dynamics,
                                                    update_frequency=self.monitor_frequency), 100000)]
        elif monitor_name == 'sum_squares_potential_error':
            self.monitors += [ExponentialAverageMonitor(
                MonitorBuilder.create_error_monitor('sum_squares_potential_error', self.model, self.input_output_stream,
                                                    'sum_squares_potential_error', self.dynamics,
                                                    update_frequency=self.monitor_frequency), 100000)]
        else:
            Exception('Could not find default monitor {}'.format(monitor_name))

    def set_active_default_monitors(self, default_monitor_names):
        for default_monitor_name in default_monitor_names:
            self.add_default_monitors(default_monitor_name)

    def add_custom_monitor(self, monitor):
        self.monitors += [monitor]

    def start_experiment(self, num_epochs, num_epoch_iterations, test_phase_length, resume_from_epoch=None,
                         show_final_plots=False):
        state_save_folder = 'experiment_results/{}'.format(self.experiment_name)

        for i in range(num_epochs):
            epoch_index = i + 1
            if resume_from_epoch is not None and epoch_index <= resume_from_epoch:
                continue
            self.dynamics_simulator.run_simulation(num_epoch_iterations * epoch_index)
            new_state_save_location = state_save_folder + '/epoch_{}'.format(epoch_index)
            if epoch_index > 1:
                prev_state_save_location = state_save_folder + '/epoch_{}'.format(epoch_index - 1)
                self.save_state(new_state_save_location=new_state_save_location,
                                prev_state_save_location=prev_state_save_location)
                remove_directory(location=prev_state_save_location)
            else:
                self.save_state(new_state_save_location=new_state_save_location)

        self.dynamics_simulator.set_testing_phase(True)
        self.dynamics_simulator.run_simulation(num_epochs * num_epoch_iterations + test_phase_length)

        self.save_state(new_state_save_location=state_save_folder + '/test',
                        prev_state_save_location=state_save_folder + '/epoch_{}'.format(num_epochs),
                        show_generated_plots=show_final_plots)

    def save_state(self, new_state_save_location, prev_state_save_location=None,
                   show_generated_plots=False):
        monitor_save_location = '{}/monitors'.format(new_state_save_location)
        pathlib.Path(monitor_save_location).mkdir(parents=True, exist_ok=True)

        for monitor in self.monitors:
            if prev_state_save_location is not None:
                prev_monitor_save_location = '{}/monitors'.format(prev_state_save_location)
                monitor.prepend_data(prev_monitor_save_location)

            monitor.plot(save_location=new_state_save_location)

            monitor.save_data(monitor_save_location)
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
                                         self_predict_phase_length, training_phase_length, test_phase_length,
                                         target_network_weights_path,
                                         model_file=None):
        experiment = Experiment(config=config, experiment_name=experiment_name, model_file=model_file)
        transfer_function = create_transfer_function(experiment.dynamics['transfer_function'])

        target_network_forward_weights_list = get_target_network_forward_weights_list(target_network_weights_path)

        # input_sequence = np.random.uniform(-1, 1, (num_train_examples, self.input_size))
        input_sequence = np.load(train_data_path)
        output_sequence = compute_non_linear_transform(input_sequence, transfer_function,
                                                       target_network_forward_weights_list)

        if test_phase_length > 0:
            # test_input_sequence = np.random.uniform(-1, 1, (num_test_examples, self.input_size))
            test_input_sequence = np.load(test_data_path)

            test_output_sequence = compute_non_linear_transform(test_input_sequence, transfer_function,
                                                                target_network_forward_weights_list)

            input_stream = CompositeStream(
                [CyclingStream((experiment.input_size, 1), input_sequence, example_iterations),
                 CyclingStream((experiment.input_size, 1), input_sequence, example_iterations),
                 CyclingStream((experiment.input_size, 1), test_input_sequence, example_iterations)],
                [0, self_predict_phase_length, self_predict_phase_length + training_phase_length])

            output_stream = CompositeStream([NoneStream((experiment.output_size, 1)),
                                             SmoothStream(CyclingStream((experiment.output_size, 1), output_sequence,
                                                                        example_iterations), 30),
                                             SmoothStream(
                                                 CyclingStream((experiment.output_size, 1), test_output_sequence,
                                                               example_iterations), 30)
                                             ],
                                            [0, self_predict_phase_length,
                                             self_predict_phase_length + training_phase_length])
        else:
            input_stream = CompositeStream(
                [CyclingStream((experiment.input_size, 1), input_sequence, example_iterations),
                 CyclingStream((experiment.input_size, 1), input_sequence, example_iterations)],
                [0, self_predict_phase_length])
            output_stream = CompositeStream([NoneStream((experiment.output_size, 1)),
                                             SmoothStream(CyclingStream((experiment.output_size, 1), output_sequence,
                                                                        example_iterations), 30)],
                                            [0, self_predict_phase_length])

        experiment.set_input_output_stream(InputOutputStream(input_stream, output_stream))
        experiment.initialise_dynamics_simulator()
        #experiment.dynamics_simulator.save_model(save_location='./saved_models', name='')
        return experiment

    @staticmethod
    def create_xor_experiment(config, experiment_name, train_data_path, example_iterations,
                              self_predict_phase_length, model_file=None):
        experiment = Experiment(config=config, experiment_name=experiment_name, model_file=model_file)
        # input_sequence = [[0.1, 0.1, 0.8], [0.1, 0.8, 0.8], [0.8, 0.1, 0.8], [0.8, 0.8, 0.8]]
        # output_sequence = [0.1, 0.8, 0.8, 0.1]

        input_sequence = np.load(train_data_path + '/train_input_sequence_size{}.npy'.format(experiment.input_size))
        output_sequence = np.load(train_data_path + '/train_output_sequence_size{}.npy'.format(experiment.output_size))
        # input_stream = CompositeStream([CyclingStream((self.input_size, 1), input_sequence, example_iterations),

        random_input_sequence = np.random.uniform(-1.0, 1.0, (500, experiment.input_size))
        input_stream = CompositeStream(
            [CyclingStream((experiment.input_size, 1), random_input_sequence, example_iterations),
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
    def create_single_input_output_experiment(config, experiment_name, example_iterations,
                                              self_predict_phase_length, model_file=None):
        experiment = Experiment(config=config, experiment_name=experiment_name, model_file=model_file)

        input_sequence = np.array([1.0, -1.0])
        output_sequence = np.array([2.0])
        # input_stream = CompositeStream([CyclingStream((self.input_size, 1), input_sequence, example_iterations),

        random_input_sequence = np.random.uniform(-1.0, 1.0, (500, experiment.input_size))
        input_stream = CompositeStream(
            [CyclingStream((experiment.input_size, 1), random_input_sequence, example_iterations),
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

        experiment.set_input_output_stream(MNISTInputOutputStream(train_data_path + 'train_images.idx3-ubyte',
                                                                  train_data_path + 'train_labels.idx1-ubyte',
                                                                  training_phase_length))
        experiment.initialise_dynamics_simulator()
        return experiment


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    input_args = sys.argv[1:]

    parser.add_argument('experiment_name', help='Name of the experiment.')
    parser.add_argument('parameter_config_file', default=None,
                        help='Parameter configuration file.')
    parser.add_argument('-monitored_values_config_file', default=None,
                        help='Monitored values configuration file.')

    parser.add_argument('num_epochs', type=int, help='Number of epochs to train for.')
    parser.add_argument('num_epoch_iterations', type=int,
                        help='Number of iterations to step through the dynamics per epoch.')

    parser.add_argument('-self_predict_phase_length', type=int, default=0,
                        help='Length of initial self-prediction stage where no output targets are given')
    parser.add_argument('-test_phase_length', type=int, default=0,
                        help='Length of the final test phase in iteration steps.')
    parser.add_argument('-example_iterations', type=int, default=1000,
                        help='Number of iterations that each training example is presented for.')

    parser.add_argument('-target_network_weights_path', default='./target_network_weights/3_layer_sf_2x10/',
                        help='Location of stored target network weight matrices.')

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
                                                                        target_network_weights_path=
                                                                        args.target_network_weights_path,
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

    if args.monitored_values_config_file is not None:
        print("Using custom monitoring values from {}".format(args.monitored_values_config_file))
        monitor_name_list = read_monitoring_values_config_file(args.monitored_values_config_file)
    else:
        print("Using default monitoring values")
        monitor_name_list = read_monitoring_values_config_file('./monitored_values_configurations/default.txt')

    experiment.set_active_default_monitors(monitor_name_list)

    experiment.start_experiment(num_epochs=args.num_epochs, num_epoch_iterations=args.num_epoch_iterations,
                                test_phase_length=args.test_phase_length,
                                resume_from_epoch=args.resume_from_epoch,
                                show_final_plots=args.show_final_plots)
