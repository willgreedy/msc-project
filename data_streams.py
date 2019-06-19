from abc import ABC, abstractmethod
import numpy as np

from mlxtend.data import loadlocal_mnist


class Stream(ABC):
    def __init__(self, shape):
        self.shape = shape

    @abstractmethod
    def get(self, iteration_num):
        pass


class ConstantStream(Stream):
    def __init__(self, shape, constant_value):
        super().__init__(shape)
        self.constant_value = constant_value

    def get(self, iteration_num):
        return np.ones(self.shape) * self.constant_value


class CyclingStream(Stream):
    def __init__(self, shape, values, num_iterations):
        super().__init__(shape)
        self.values = values
        self.num_iterations = num_iterations

    def get(self, iteration_num):
        curr_index = int(iteration_num / self.num_iterations) % len(self.values)
        return np.ones(self.shape) * self.values[curr_index]


class DataStream(Stream):
    def __init__(self, data, num_iterations):
        self.data = data
        super().__init__((len(data), 1))
        self.num_iterations = num_iterations

    def get(self, iteration_num):
        curr_index = int(iteration_num / self.num_iterations) % len(self.data)
        return self.data[curr_index]


class InputOutputStream:
    def __init__(self, input_stream, output_target_stream):
        self.input_stream = input_stream
        self.output_target_stream = output_target_stream

    def get_inputs(self, iteration_num):
        return self.input_stream.get(iteration_num)

    def get_output_targets(self, iteration_num):
        return self.output_target_stream.get(iteration_num)


class MNISTInputOutputStream(InputOutputStream):
    def __init__(self, images_path, labels_path, num_iterations):
        images, labels = loadlocal_mnist(images_path=images_path, labels_path=labels_path)

        images = images / 255.0

        labels = labels.reshape(-1)
        one_hot_labels = np.eye(10)[labels]

        input_stream = DataStream(images, num_iterations)
        output_target_stream = DataStream(one_hot_labels, num_iterations)
        super().__init__(input_stream, output_target_stream)

