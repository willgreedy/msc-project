import numpy as np

num_train_examples = 500
num_test_examples = 500

input_size = 30

train_input_sequence = np.random.uniform(-1, 1, (num_train_examples, input_size))
test_input_sequence = np.random.uniform(-1, 1, (num_test_examples, input_size))

np.save('train_input_sequence_size{}_examples{}'.format(input_size, num_train_examples), train_input_sequence)
np.save('test_input_sequence_size{}_examples{}'.format(input_size, num_test_examples), test_input_sequence)