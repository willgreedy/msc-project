import sys
import numpy as np
from parameter_config import ParameterConfig
from models import MultiCompartmentModel
from helpers import UniformInitialiser, ConstantInitialiser, create_plot, create_diff_plot, visualise_MNIST, show_plots, load_model, compute_non_linear_transform, create_transfer_function
from dynamics_simulator import StandardDynamicsSimulator, SimplifiedDynamicsSimulator
from data_streams import InputOutputStream, CompositeStream, ConstantStream, CyclingStream, MNISTInputOutputStream, SmoothStream, NoneStream
from monitors import MonitorBuilder, ExponentialAverageMonitor


class ExperimentBuilder:
    def __init__(self, args, model_file=None):
        if len(args) > 0:
            config_file = args[0]
            self.config = ParameterConfig(config_file)
            print("Using configuration: {}".format(config_file))
        else:
            print("Using default configuration.")
            self.config = ParameterConfig()

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

        if model_file == None:
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
        self.monitors = []
        self.monitor_frequency = 1

    def add_monitors(self):
        self.monitors += [MonitorBuilder.create_feedforward_predict_weight_diff_monitor(self.model, 1, 0, update_frequency=self.monitor_frequency)]
        self.monitors += [MonitorBuilder.create_weight_angle_monitor(self.model, 0, 'feedforward_feedback_angle',
                                                                     update_frequency=self.monitor_frequency)]

        self.monitors += [MonitorBuilder.create_weight_angle_monitor(self.model, 0, 'feedforward_predict_angle',
                                                                     update_frequency=self.monitor_frequency)]
        self.monitors += [MonitorBuilder.create_weight_angle_monitor(self.model, 0, 'feedback_interneuron_angle',
                                                                     update_frequency=self.monitor_frequency)]

        self.monitors += [MonitorBuilder.create_weight_monitor(self.model, "feedforward_weights", 1, 0, 0, update_frequency=self.monitor_frequency)]
        self.monitors += [MonitorBuilder.create_weight_monitor(self.model, "predict_weights", 0, 0, 0, update_frequency=self.monitor_frequency)]
        self.monitors += [MonitorBuilder.create_weight_monitor(self.model, "feedback_weights", 0, 0, 0, update_frequency=self.monitor_frequency)]

        self.monitors += [MonitorBuilder.create_potential_monitor(self.model, 0, "pyramidal_basal", 0, update_frequency=self.monitor_frequency)]
        self.monitors += [MonitorBuilder.create_potential_monitor(self.model, 0, "pyramidal_soma", 0, update_frequency=self.monitor_frequency)]
        self.monitors += [MonitorBuilder.create_potential_monitor(self.model, 0, "pyramidal_apical", 0, update_frequency=self.monitor_frequency)]

        self.monitors += [MonitorBuilder.create_potential_monitor(self.model, 0, "interneuron_basal", 0, update_frequency=self.monitor_frequency)]
        self.monitors += [MonitorBuilder.create_potential_monitor(self.model, 0, "interneuron_soma", 0, update_frequency=self.monitor_frequency)]

        self.monitors += [MonitorBuilder.create_basal_soma_rate_diff_monitor(self.model, 0, 0, self.dynamics, update_frequency=self.monitor_frequency)]

        self.monitors += [MonitorBuilder.create_potential_monitor(self.model, 1, "pyramidal_basal", 0, update_frequency=self.monitor_frequency)]
        self.monitors += [MonitorBuilder.create_potential_monitor(self.model, 1, "pyramidal_soma", 0, update_frequency=self.monitor_frequency)]

    def initialise_xor_experiment(self):
        #input_sequence = [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
        input_sequence = [[0.0, 0.0, 0.04], [0.0, 0.04, 0.04], [0.04, 0.0, 0.04], [0.04, 0.04, 0.04]]
        input_stream = CompositeStream([CyclingStream((self.input_size, 1), [0.02, 0.04, 0.0], 250000),
                                        CyclingStream((self.input_size, 1), input_sequence, 20000)],
                                       [0, 200000])

        output_sequence = [0.0, 1.00, 1.00, 0.0]
        output_stream = CompositeStream([NoneStream((self.output_size, 1)),
                                         SmoothStream(CyclingStream((self.output_size, 1), output_sequence, 20000), 30)],
                                                     [0, 200000]
        )
        input_output_stream = InputOutputStream(input_stream, output_stream)

        self.monitors += [MonitorBuilder.create_data_monitor(input_output_stream, 'target', 0, update_frequency=self.monitor_frequency)]

        if self.dynamics['type'] == 'standard':
            self.dynamics_simulator = StandardDynamicsSimulator(self.model, input_output_stream, self.dynamics, self.monitors)
        elif self.dynamics['type'] == 'simplified':
            self.dynamics_simulator = SimplifiedDynamicsSimulator(self.model, input_output_stream, self.dynamics, self.monitors)
        else:
            raise Exception("Invalid dynamics type: {}".format({self.dynamics['type']}))

    def initialise_target_network_experiment(self):
        num_examples = 50
        input_sequence = np.random.random((num_examples, self.input_size))
        input_stream = CompositeStream([CyclingStream((self.input_size, 1), input_sequence, 10000),
                                        CyclingStream((self.input_size, 1), input_sequence, 10000)],
                                       [0, 1])

        print(input_sequence)
        transfer_function = create_transfer_function(self.dynamics['transfer_function'])

        forward_weights_list = []
        #forward_weights_list += [(np.random.random((30, 20)) * 2) - 1]
        #forward_weights_list += [(np.random.random((20, 10)) * 2) - 1]
        forward_weights_list += [np.load('./first_layer_feedforward_weights.npy').copy().T]
        forward_weights_list += [np.load('./second_layer_feedforward_weights.npy').copy().T]
        output_sequence = compute_non_linear_transform(input_sequence, transfer_function, forward_weights_list)

        output_stream = CompositeStream([NoneStream((self.output_size, 1)),
                                         SmoothStream(CyclingStream((self.output_size, 1), output_sequence, 10000),
                                                      30)],
                                        [0, 1])

        input_output_stream = InputOutputStream(input_stream, output_stream)

        self.monitors += [MonitorBuilder.create_data_monitor(input_output_stream, 'input', 0,
                                                             update_frequency=self.monitor_frequency)]

        self.monitors += [MonitorBuilder.create_data_monitor(input_output_stream, 'target', 0,
                                                             update_frequency=self.monitor_frequency)]

        self.monitors += [ExponentialAverageMonitor(MonitorBuilder.create_error_monitor(self.model, output_stream, 'sum_squares_error', update_frequency=self.monitor_frequency), 50000)]

        if self.dynamics['type'] == 'standard':
            self.dynamics_simulator = StandardDynamicsSimulator(self.model, input_output_stream, self.dynamics,
                                                                self.monitors)
        elif self.dynamics['type'] == 'simplified':
            self.dynamics_simulator = SimplifiedDynamicsSimulator(self.model, input_output_stream, self.dynamics,
                                                                  self.monitors)
        else:
            raise Exception("Invalid dynamics type: {}".format({self.dynamics['type']}))

    def start_experiment(self):
        num_iterations = 5000000
        self.dynamics_simulator.run_simulation(num_iterations)
        experiment_name = "target_network_symmetric"
        self.dynamics_simulator.save_model(experiment_name)
        self.plot_monitors()

    def plot_monitors(self):
        for monitor in self.monitors:
            create_plot(monitor)
        show_plots()

if __name__ == '__main__':
    input_args = sys.argv[1:]
    experiment = ExperimentBuilder(input_args, "target_network_symmetric")
    print("Layer 1 weights: {}".format(experiment.model.get_layers()[0][1].get_feedforward_weights()))
    print("Layer 2 weights: {}".format(experiment.model.get_layers()[1][1].get_feedforward_weights()))
    experiment.add_monitors()
    experiment.initialise_target_network_experiment()

    import atexit
    atexit.register(experiment.plot_monitors)
    experiment.start_experiment()


