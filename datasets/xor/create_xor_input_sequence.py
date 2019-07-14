import numpy as np

input_size = 3
num_examples = 4

#train_input_sequence = np.random.uniform(-1, 1, (num_train_examples, input_size))
#test_input_sequence = np.random.uniform(-1, 1, (num_test_examples, input_size))

train_input_sequence = np.array([[0.1, 0.1, 0.8], [0.1, 0.8, 0.8], [0.8, 0.1, 0.8], [0.8, 0.8, 0.8]])
train_output_sequence = np.array([0.1, 0.8, 0.8, 0.1])

np.save('train_input_sequence_size{}'.format(input_size, num_examples), train_input_sequence)
np.save('train_output_sequence_size{}'.format(1, num_examples), train_output_sequence)