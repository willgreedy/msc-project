import sys
from parameter_config import ParameterConfig
from models import MultiCompartmentModel
from helpers import UniformInitialiser, create_plot, show_plots
from dynamics_simulator import DynamicsSimulator
from monitors import DataMonitor, CellMonitor


class ExperimentBuilder:
    def __init__(self, args):
        if len(args) > 0:
            config_file = args[0]
            config = ParameterConfig(config_file)
            print("Using configuration: {}".format(config_file))
        else:
            print("Using default configuration.")
            config = ParameterConfig()

        network_architecture = config['network_architecture']
        input_size = network_architecture['input_size']
        output_size = network_architecture['output_size']
        layer_sizes = network_architecture['layer_sizes']

        weight_inititalision = config['weight_intialisation']
        weight_inititalision_type = weight_inititalision['type']

        if weight_inititalision_type == 'uniform':
            lower_bound = weight_inititalision['lower_bound']
            upper_bound = weight_inititalision['upper_bound']
            weight_inititaliser = UniformInitialiser(lower_bound, upper_bound)
        else:
            raise Exception("Invalid weight initialisation type specified.")

        feedforward_learning_rate = config['feedforward_learning_rate']
        predict_learning_rate = config['predict_learning_rate']
        interneuron_learning_rate = config['interneuron_learning_rate']
        feedback_learning_rate = config['feedback_learning_rate']

        model = MultiCompartmentModel(input_size,
                                      layer_sizes,
                                      output_size,
                                      weight_inititaliser,
                                      weight_inititaliser,
                                      weight_inititaliser,
                                      weight_inititaliser,
                                      feedforward_learning_rate,
                                      predict_learning_rate,
                                      interneuron_learning_rate,
                                      feedback_learning_rate)

        print("Created {}".format(str(model)))

        self.monitors = []
        self.monitors += [CellMonitor(model, 0, "pyramidal_soma", 0)]

        self.dynamics_simulator = DynamicsSimulator(model, config, self.monitors)

    def start_experiment(self):
        self.dynamics_simulator.run_simulation(5000)
        for monitor in self.monitors:
            create_plot(monitor)
        show_plots()


if __name__ == '__main__':
    input_args = sys.argv[1:]
    experiment = ExperimentBuilder(input_args)
    experiment.start_experiment()
