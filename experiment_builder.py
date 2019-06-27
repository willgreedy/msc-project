import sys
import numpy as np
from parameter_config import ParameterConfig
from models import MultiCompartmentModel
from helpers import UniformInitialiser, create_plot, create_diff_plot, visualise_MNIST, show_plots, load_model
from dynamics_simulator import DynamicsSimulator
from data_streams import InputOutputStream, ConstantStream, CyclingStream, MNISTInputOutputStream, SmoothStream
from monitors import MonitorBuilder


class ExperimentBuilder:
    def __init__(self, args):
        if len(args) > 0:
            config_file = args[0]
            self.config = ParameterConfig(config_file)
            print("Using configuration: {}".format(config_file))
        else:
            print("Using default configuration.")
            self.config = ParameterConfig()

    def initialise_xor_experiment(self, model_file=None):
        network_architecture = self.config['network_architecture']
        input_size = network_architecture['input_size']
        output_size = network_architecture['output_size']
        layer_sizes = network_architecture['layer_sizes']

        weight_inititalision = self.config['weight_intialisation']
        weight_inititalision_type = weight_inititalision['type']

        if weight_inititalision_type == 'uniform':
            lower_bound = weight_inititalision['lower_bound']
            upper_bound = weight_inititalision['upper_bound']
            weight_inititaliser = UniformInitialiser(lower_bound, upper_bound)
        else:
            raise Exception("Invalid weight initialisation type specified.")

        feedforward_learning_rates = self.config['feedforward_learning_rates']
        predict_learning_rates = self.config['predict_learning_rates']
        interneuron_learning_rates = self.config['interneuron_learning_rates']
        feedback_learning_rates = self.config['feedback_learning_rates']

        if model_file == None:
            model = MultiCompartmentModel(input_size,
                                      layer_sizes,
                                      output_size,
                                      weight_inititaliser,
                                      weight_inititaliser,
                                      weight_inititaliser,
                                      weight_inititaliser,
                                      feedforward_learning_rates,
                                      predict_learning_rates,
                                      interneuron_learning_rates,
                                      feedback_learning_rates)
            print("Created {}".format(str(model)))
        else:
            model = load_model(model_file)
            print("Loaded {}".format(str(model)))



        monitor_frequency = 1
        self.monitors = []

        self.monitors += [MonitorBuilder.create_feedforward_predict_weight_diff_monitor(model, 1, 0, update_frequency=monitor_frequency)]
        self.monitors += [MonitorBuilder.create_weight_angle_monitor(model, 0, 'feedforward_feedback_angle',
                                                                     update_frequency=monitor_frequency)]

        self.monitors += [MonitorBuilder.create_weight_monitor(model, "feedforward_weights", 0, 0, 0, update_frequency=monitor_frequency)]
        #self.monitors += [MonitorBuilder.create_weight_monitor(model, "feedforward_weights", 1, 0, 0, update_frequency=monitor_frequency)]

        self.monitors += [MonitorBuilder.create_weight_monitor(model, "feedforward_weights", 0, 1, 0, update_frequency=monitor_frequency)]
        self.monitors += [MonitorBuilder.create_weight_monitor(model, "feedforward_weights", 0, 0, 1, update_frequency=monitor_frequency)]
        self.monitors += [MonitorBuilder.create_weight_monitor(model, "feedforward_weights", 0, 1, 1, update_frequency=monitor_frequency)]

        #self.monitors += [MonitorBuilder.create_weight_monitor(model, "feedback_weights", 0, 0, 0, update_frequency=monitor_frequency)]

        self.monitors += [MonitorBuilder.create_weight_monitor(model, "predict_weights", 0, 0, 0, update_frequency=monitor_frequency)]

        self.monitors += [MonitorBuilder.create_potential_monitor(model, 0, "pyramidal_basal", 0, update_frequency=monitor_frequency)]
        self.monitors += [MonitorBuilder.create_potential_monitor(model, 0, "pyramidal_soma", 0, update_frequency=monitor_frequency)]
        self.monitors += [MonitorBuilder.create_potential_monitor(model, 0, "pyramidal_apical", 0, update_frequency=monitor_frequency)]

        self.monitors += [MonitorBuilder.create_potential_monitor(model, 0, "interneuron_basal", 0, update_frequency=monitor_frequency)]
        self.monitors += [MonitorBuilder.create_potential_monitor(model, 0, "interneuron_soma", 0, update_frequency=monitor_frequency)]

        #self.monitors += [MonitorBuilder.create_potential_monitor(model, 1, "pyramidal_basal", 0, update_frequency=monitor_frequency)]
        #self.monitors += [MonitorBuilder.create_potential_monitor(model, 1, "pyramidal_soma", 0, update_frequency=monitor_frequency)]
        #self.monitors += [MonitorBuilder.create_potential_monitor(model, 1, "pyramidal_apical", 0, update_frequency=monitor_frequency)]

        #self.monitors += [MonitorBuilder.create_potential_monitor(model, 1, "interneuron_basal", 0, update_frequency=monitor_frequency)]
        #self.monitors += [MonitorBuilder.create_potential_monitor(model, 1, "interneuron_soma", 0, update_frequency=monitor_frequency)]

        self.monitors += [MonitorBuilder.create_basal_soma_rate_diff_monitor(model, 0, 0, self.config, update_frequency=monitor_frequency)]

        self.monitors += [MonitorBuilder.create_potential_monitor(model, 1, "pyramidal_basal", 0, update_frequency=monitor_frequency)]
        self.monitors += [MonitorBuilder.create_potential_monitor(model, 1, "pyramidal_soma", 0, update_frequency=monitor_frequency)]

        #input_output_stream = MNISTInputOutputStream('datasets/mnist/train_images.idx3-ubyte',
        #                                'datasets/mnist/train_labels.idx1-ubyte',
        #                                1000)

        #visualise_MNIST(input_output_stream.get_inputs(2001))
        #print(input_output_stream.get_output_targets(2001))

        #input_sequence = [0.1, 0.4]
        #input_sequence = [0.1]
        input_sequence = [[0.1, 0.1, 0.4], [0.1, 0.4, 0.4], [0.4, 0.1, 0.1], [0.4, 0.4, 0.4]]
        input_stream = CyclingStream((input_size, 1), input_sequence, 10000)

        output_sequence = [0.01, 0.05, 0.05, 0.01]
        #output_sequence = [0.0]
        output_stream = SmoothStream(CyclingStream((output_size, 1), output_sequence, 10000), 30)
        #input_stream = ConstantStream((input_size, 1), 0.02)
        #output_stream = ConstantStream((output_size, 1), 0.02)

        input_output_stream = InputOutputStream(input_stream, output_stream)

        self.monitors += [MonitorBuilder.create_data_monitor(input_output_stream, 'target', 0, update_frequency=monitor_frequency)]

        self.dynamics_simulator = DynamicsSimulator(model, input_output_stream, self.config, self.monitors)
        #show_plots()

    def start_experiment(self):
        num_iterations = 5000000
        self.dynamics_simulator.run_simulation(num_iterations)
        experiment_name = "2_layer_xor"
        self.dynamics_simulator.save_model(experiment_name)
        self.plot_monitors()

    def plot_monitors(self):
        for monitor in self.monitors:
            create_plot(monitor)
        create_diff_plot(self.monitors[-1], self.monitors[-2])
        show_plots()

if __name__ == '__main__':
    input_args = sys.argv[1:]
    experiment = ExperimentBuilder(input_args)
    experiment.initialise_xor_experiment()

    import atexit
    atexit.register(experiment.plot_monitors)
    experiment.start_experiment()


