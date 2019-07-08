import sys
import numpy as np
from parameter_config import ParameterConfig
from models import MultiCompartmentModel
from helpers import UniformInitialiser, ConstantInitialiser, create_plot, create_diff_plot, visualise_MNIST, \
    show_plots, load_model, compute_non_linear_transform, create_transfer_function
from dynamics_simulator import StandardDynamicsSimulator, SimplifiedDynamicsSimulator
from data_streams import InputOutputStream, CompositeStream, ConstantStream, CyclingStream, MNISTInputOutputStream, \
    SmoothStream, NoneStream
from monitors import MonitorBuilder, ExponentialAverageMonitor


class ExperimentBuilder:
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
        self.monitors = []
        self.monitor_frequency = 25

    def add_monitors(self):
        self.monitors += [MonitorBuilder.create_feedforward_predict_weight_diff_monitor(self.model, 1, 0,
                                                                                        update_frequency
                                                                                        =self.monitor_frequency)]
        self.monitors += [MonitorBuilder.create_weight_angle_monitor(self.model, 0, 'feedforward_feedback_angle',
                                                                     update_frequency=self.monitor_frequency)]

        self.monitors += [MonitorBuilder.create_weight_angle_monitor(self.model, 0, 'feedforward_predict_angle',
                                                                     update_frequency=self.monitor_frequency)]
        self.monitors += [MonitorBuilder.create_weight_angle_monitor(self.model, 0, 'feedback_interneuron_angle',
                                                                     update_frequency=self.monitor_frequency)]

        self.monitors += [MonitorBuilder.create_weight_monitor(self.model, "feedforward_weights", 1, 0, 0,
                                                               update_frequency=self.monitor_frequency)]
        self.monitors += [MonitorBuilder.create_weight_monitor(self.model, "predict_weights", 0, 0, 0,
                                                               update_frequency=self.monitor_frequency)]
        self.monitors += [MonitorBuilder.create_weight_monitor(self.model, "feedback_weights", 0, 0, 0,
                                                               update_frequency=self.monitor_frequency)]

        self.monitors += [MonitorBuilder.create_potential_monitor(self.model, 0, "pyramidal_basal", 0,
                                                                  update_frequency=self.monitor_frequency)]
        self.monitors += [MonitorBuilder.create_potential_monitor(self.model, 0, "pyramidal_soma", 0,
                                                                  update_frequency=self.monitor_frequency)]
        self.monitors += [MonitorBuilder.create_potential_monitor(self.model, 0, "pyramidal_apical", 0,
                                                                  update_frequency=self.monitor_frequency)]

        self.monitors += [MonitorBuilder.create_potential_monitor(self.model, 0, "interneuron_basal", 0,
                                                                  update_frequency=self.monitor_frequency)]
        self.monitors += [MonitorBuilder.create_potential_monitor(self.model, 0, "interneuron_soma", 0,
                                                                  update_frequency=self.monitor_frequency)]

        self.monitors += [MonitorBuilder.create_pyramidal_basal_soma_rate_diff_monitor(self.model, 0, 0, self.dynamics,
                                                                                       update_frequency=self.monitor_frequency)]

        self.monitors += [MonitorBuilder.create_potential_monitor(self.model, 1, "pyramidal_basal", 0,
                                                                  update_frequency=self.monitor_frequency)]
        self.monitors += [MonitorBuilder.create_potential_monitor(self.model, 1, "pyramidal_soma", 0,
                                                                  update_frequency=self.monitor_frequency)]

    def initialise_xor_experiment(self, example_iterations, self_predict_phase_length):
        input_sequence = [[0.1, 0.1, 0.8], [0.1, 0.8, 0.8], [0.8, 0.1, 0.8], [0.8, 0.8, 0.8]]
        # input_stream = CompositeStream([CyclingStream((self.input_size, 1), input_sequence, example_iterations),
        random_input_sequence = np.random.random((50, 3))
        input_stream = CompositeStream([CyclingStream((self.input_size, 1), random_input_sequence, example_iterations),
                                        CyclingStream((self.input_size, 1), input_sequence, example_iterations)],
                                       [0, self_predict_phase_length])

        output_sequence = [0.1, 0.8, 0.8, 0.1]
        output_stream = CompositeStream([NoneStream((self.output_size, 1)),
                                         SmoothStream(CyclingStream((self.output_size, 1), output_sequence,
                                                                    example_iterations), 30)],
                                        [0, self_predict_phase_length])
        input_output_stream = InputOutputStream(input_stream, output_stream)

        self.monitors += [MonitorBuilder.create_data_monitor(input_output_stream, 'input', 0,
                                                             update_frequency=self.monitor_frequency)]

        self.monitors += [MonitorBuilder.create_data_monitor(input_output_stream, 'target', 0,
                                                             update_frequency=self.monitor_frequency)]

        self.monitors += [ExponentialAverageMonitor(
            MonitorBuilder.create_error_monitor(self.model, input_output_stream, 'sum_squares_error', self.dynamics,
                                                update_frequency=self.monitor_frequency), 25000)]

        if self.dynamics['type'] == 'standard':
            self.dynamics_simulator = StandardDynamicsSimulator(self.model, input_output_stream, self.dynamics,
                                                                self.monitors)
        elif self.dynamics['type'] == 'simplified':
            self.dynamics_simulator = SimplifiedDynamicsSimulator(self.model, input_output_stream, self.dynamics,
                                                                  self.monitors)
        else:
            raise Exception("Invalid dynamics type: {}".format({self.dynamics['type']}))

    def initialise_target_network_experiment(self, num_examples, example_iterations, self_predict_phase_length):
        input_sequence = np.random.uniform(-1, 1, (num_examples, self.input_size))
        input_stream = CompositeStream([CyclingStream((self.input_size, 1), input_sequence, example_iterations),
                                        CyclingStream((self.input_size, 1), input_sequence, example_iterations)],
                                       [0, self_predict_phase_length])

        # print(input_sequence)
        transfer_function = create_transfer_function(self.dynamics['transfer_function'])

        forward_weights_list = []
        # forward_weights_list += [np.random.uniform(-1, 1, (30, 50))]
        # forward_weights_list += [np.random.uniform(-1, 1, (50, 10))]
        forward_weights_list += [np.load('./target_network_weights/first_layer_feedforward_weights.npy').copy()]
        forward_weights_list += [np.load('./target_network_weights/second_layer_feedforward_weights.npy').copy()]
        print("Layer 1 weights: {}".format(forward_weights_list[0]))
        print("Layer 2 weights: {}".format(forward_weights_list[1]))
        output_sequence = compute_non_linear_transform(input_sequence, transfer_function, forward_weights_list)

        output_stream = CompositeStream([NoneStream((self.output_size, 1)),
                                         SmoothStream(CyclingStream((self.output_size, 1), output_sequence,
                                                                    example_iterations), 30)],
                                        [0, self_predict_phase_length])

        input_output_stream = InputOutputStream(input_stream, output_stream)

        self.monitors += [MonitorBuilder.create_data_monitor(input_output_stream, 'input', 0,
                                                             update_frequency=self.monitor_frequency)]

        self.monitors += [MonitorBuilder.create_data_monitor(input_output_stream, 'target', 0,
                                                             update_frequency=self.monitor_frequency)]

        self.monitors += [ExponentialAverageMonitor(
            MonitorBuilder.create_error_monitor(self.model, input_output_stream, 'sum_squares_error', self.dynamics,
                                                update_frequency=self.monitor_frequency), 25000)]

        if self.dynamics['type'] == 'standard':
            self.dynamics_simulator = StandardDynamicsSimulator(self.model, input_output_stream, self.dynamics,
                                                                self.monitors)
        elif self.dynamics['type'] == 'simplified':
            self.dynamics_simulator = SimplifiedDynamicsSimulator(self.model, input_output_stream, self.dynamics,
                                                                  self.monitors)
        else:
            raise Exception("Invalid dynamics type: {}".format({self.dynamics['type']}))

    def initialise_mnist_experiment(self):
        input_output_stream = MNISTInputOutputStream('mnist/train_images.idx3-ubyte', 'train_labels.idx1-ubyte', 55000)

        transfer_function = create_transfer_function(self.dynamics['transfer_function'])

        self.monitors += [MonitorBuilder.create_data_monitor(input_output_stream, 'input', 0,
                                                             update_frequency=self.monitor_frequency)]

        self.monitors += [MonitorBuilder.create_data_monitor(input_output_stream, 'target', 0,
                                                             update_frequency=self.monitor_frequency)]

        self.monitors += [ExponentialAverageMonitor(
            MonitorBuilder.create_error_monitor(self.model, input_output_stream, 'sum_squares_error',
                                                update_frequency=self.monitor_frequency), 50000)]

        if self.dynamics['type'] == 'standard':
            self.dynamics_simulator = StandardDynamicsSimulator(self.model, input_output_stream, self.dynamics,
                                                                self.monitors)
        elif self.dynamics['type'] == 'simplified':
            self.dynamics_simulator = SimplifiedDynamicsSimulator(self.model, input_output_stream, self.dynamics,
                                                                  self.monitors)
        else:
            raise Exception("Invalid dynamics type: {}".format({self.dynamics['type']}))

    def start_experiment(self):
        num_epoch_iterations = 500000
        num_test_iterations = 500000
        num_epochs = 8

        for i in range(num_epochs):
            self.dynamics_simulator.run_simulation(num_epoch_iterations * (i + 1))
            self.plot_monitors(show=False, sub_directory='epoch_{}'.format(i + 1))

        self.dynamics_simulator.set_testing_phase(True)
        self.dynamics_simulator.run_simulation(num_epochs * num_epoch_iterations + num_test_iterations)

        self.dynamics_simulator.save_model(self.experiment_name)
        self.plot_monitors(show=True, sub_directory='test')

    def plot_monitors(self, show=False, sub_directory=None):
        for monitor in self.monitors:
            create_plot(monitor, show, save_location=self.experiment_name, sub_directory=sub_directory)

        show_plots()


if __name__ == '__main__':
    input_args = sys.argv[1:]

    model_file = None
    if len(input_args) > 0:
        experiment_name = input_args[0]
    else:
        raise Exception("No experiment name provided.")

    if len(input_args) > 1:
        config_file = input_args[1]
        parameter_config = ParameterConfig(config_file)
        print("Using configuration: {}".format(config_file))
    else:
        print("Using default configuration.")
        parameter_config = ParameterConfig()

    experiment = ExperimentBuilder(parameter_config, experiment_name, model_file=model_file)
    experiment.add_monitors()

    if experiment_name.startswith('xor'):
        experiment.initialise_xor_experiment(example_iterations=1000, self_predict_phase_length=5000000)
    elif experiment_name.startswith('target_network'):
        experiment.initialise_target_network_experiment(num_examples=1000, example_iterations=1000,
                                                        self_predict_phase_length=500000)

    # import atexit
    # atexit.register(experiment.plot_monitors)

    experiment.start_experiment()
