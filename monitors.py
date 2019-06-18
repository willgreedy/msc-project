from abc import ABC, abstractmethod
import numpy as np


class Monitor(ABC):
    def __init__(self, var_name):
        self.var_name = var_name

        self.iter_numbers = []
        self.values = []

    def get_var_name(self):
        return self.var_name

    def get_values(self):
        return self.iter_numbers, self.values

    @abstractmethod
    def update(self, iter_number):
        pass


class DataMonitor(Monitor):
    def __init__(self, data_stream, data_type, data_location):
        valid_data_types = ["input", "target"]

        if data_type not in valid_data_types:
            raise Exception("Invalid data type given to monitor {}. Must be one of: {}".format(data_type,
                                                                                               valid_data_types))

        var_name = "{} location {}".format(data_type, data_stream)
        super().__init__(var_name)
        self.data_stream = data_stream
        self.data_type = data_type
        self.data_location = data_location

    def update(self, iter_number):
        if self.data_type == "input":
            inputs = self.data_stream.get_inputs(iter_number)
            value = float(inputs[self.data_location])
        elif self.data_type == "output_target":
            output_targets = self.data_stream.get_output_targets(iter_number)
            value = float(output_targets[self.data_location])
        else:
            raise Exception("Fatal Error")

        self.iter_numbers += [iter_number]
        self.values += [value]


class CellMonitor(Monitor):
    def __init__(self, model, layer_num, cell_type, cell_location):

        valid_cell_types = ["pyramidal_basal", "pyramidal_soma", "pyramidal_apical", "interneuron_basal",
                            "interneuron_soma"]

        if cell_type not in valid_cell_types:
            raise Exception("Invalid cell type given to monitor {}. Must be one of: {}".format(cell_type,
                                                                                               valid_cell_types))

        var_name = "Layer {}, {} at location {}".format(layer_num, cell_type, cell_location)
        super().__init__(var_name)
        self.model = model
        self.layer_num = layer_num
        self.cell_type = cell_type
        self.cell_location = cell_location

    def update(self, iter_number):
        layers = self.model.get_layers()
        _, layer = layers[self.layer_num]

        if self.cell_type == "pyramidal_basal":
            pyramidal_basals = layer.get_pyramidal_basal_potentials()
            value = float(pyramidal_basals[self.cell_location])
        elif self.cell_type == "pyramidal_soma":
            pyramidal_somas = layer.get_pyramidal_somatic_potentials()
            value = float(pyramidal_somas[self.cell_location])
        elif self.cell_type == "pyramidal_apical":
            pyramidal_apical = layer.get_pyramidal_apical_potentials()
            value = float(pyramidal_apical[self.cell_location])
        elif self.cell_type == "interneuron_basal":
            interneuron_basals = layer.get_interneuron_basal_potentials()
            value = float(interneuron_basals[self.cell_location])
        elif self.cell_type == "interneuron_soma":
            interneuron_somas = layer.get_interneuron_somatic_potentials()
            value = float(interneuron_somas[self.cell_location])
        else:
            raise Exception("Fatal Error")

        self.iter_numbers += [iter_number]
        self.values += [value]
