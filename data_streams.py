from abc import ABC, abstractmethod
import numpy as np

from mlxtend.data import loadlocal_mnist


class Stream(ABC):
    def __init__(self, shape):
        self.shape = shape

    def get_shape(self):
        return self.shape

    @abstractmethod
    def get(self, iteration_num):
        pass


class CompositeStream(Stream):
    def __init__(self, stream_list, start_indices):
        shape = stream_list[0].shape
        for stream in stream_list[1:]:
            if stream.shape != shape:
                raise Exception("All streams in list must have the same shape.")

        super().__init__(shape)
        if len(stream_list) != len(start_indices):
            raise Exception("List of streams and indices are not the same length.")

        if start_indices[0] != 0:
            raise Exception("First starting index in list must be 0.")

        for i in range(len(start_indices) - 1):
            if start_indices[i] > start_indices[i+1]:
                raise Exception("List of start indices must be ascending.")

        self.stream_list = stream_list
        self.start_indices = start_indices

    def get(self, iteration_num):
        # Loop backwards through the list of starting indices
        for i, start_index in list(zip(range(len(self.start_indices)), self.start_indices))[::-1]:
            if iteration_num >= start_index:
                stream = self.stream_list[i]
                return stream.get(iteration_num - start_index)
        raise Exception("Stream index out of bounds: {}".format(iteration_num))


class SmoothStream(Stream):
    def __init__(self, orig_stream, time_constant_iters):
        super().__init__(orig_stream.get_shape())
        self.orig_stream = orig_stream
        self.time_constant_iters = time_constant_iters
        self.decay_factor = np.exp(-1.0 / time_constant_iters)

        self.last_smoothed_value = orig_stream.get(0)
        self.last_retrieved_iteration = 0

    def get(self, iteration_num):
        if iteration_num < self.last_retrieved_iteration:
            self.last_smoothed_value = self.orig_stream.get(0)
            self.last_retrieved_iteration = 0

        curr_value = self.last_smoothed_value.copy()
        for i in range(self.last_retrieved_iteration, iteration_num):
            curr_value *= self.decay_factor
            curr_value += (1 - self.decay_factor) * self.orig_stream.get(i)
            self.last_smoothed_value = curr_value
        self.last_retrieved_iteration = iteration_num
        return self.last_smoothed_value


class NoneStream(Stream):
    def __init__(self, shape):
        super().__init__(shape)

    def get(self, iteration_num):
        return None


class ConstantStream(Stream):
    def __init__(self, shape, constant_value):
        super().__init__(shape)
        self.constant_value = constant_value

    def get(self, iteration_num):
        return np.ones(self.shape) * self.constant_value


class CyclingStream(Stream):
    def __init__(self, shape, value_sequence, num_iterations):
        super().__init__(shape)
        value_sequence = np.array(value_sequence)

        if len(value_sequence.shape) == 1:
            self.value_sequence = np.dot(value_sequence.reshape((-1, 1)), np.ones((1, shape[0])))
        elif len(value_sequence.shape) == 2:
            if value_sequence.shape[1] != shape[0]:
                raise Exception('Values of input stream incorrect size. Expected: {}, Received: {}'.format(
                    value_sequence.shape[1], shape[0]))
            else:
                self.value_sequence = value_sequence
        else:
            raise Exception('Invalid input sequence. Must be 1-d or 2-d. Dimensions provided: {}'.format(
                len(value_sequence.shape)))

        self.num_iterations = num_iterations

    def get(self, iteration_num):
        curr_index = int(iteration_num / self.num_iterations) % self.value_sequence.shape[0]
        return (self.value_sequence[curr_index, :]).reshape((-1, 1))


class DataStream(Stream):
    def __init__(self, data, num_examples):
        self.data = data
        super().__init__((len(data), 1))
        self.num_examples = num_examples

    def get(self, iteration_num):
        curr_index = int(iteration_num / self.num_examples) % len(self.data)
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
    def __init__(self, images_path, labels_path, num_examples):
        images, labels = loadlocal_mnist(images_path=images_path, labels_path=labels_path)

        images = images / 255.0

        labels = labels.reshape(-1)
        one_hot_labels = np.eye(10)[labels]
        # Re-scale to range {0.1, 0.8}
        one_hot_labels = one_hot_labels * 0.7 + 0.1

        input_stream = DataStream(images, num_examples)
        output_target_stream = DataStream(one_hot_labels, num_examples)
        super().__init__(input_stream, output_target_stream)

