import torch
from torch import nn, optim
from torch.nn.functional import mse_loss
from torch.utils.data import Dataset
from collections import OrderedDict
from helpers import create_transfer_function, get_target_network_forward_weights_list, compute_non_linear_transform, \
    visualise_transfer_function
import numpy as np

transfer_function_config = {'type': 'soft-rectify',
                            'gamma': 0.1,
                            'beta': 1.0,
                            'theta': 1.0}


class SoftRectifyTransferFunction(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gamma = config['gamma'] if 'gamma' in config else 1.0
        self.beta = config['beta'] if 'beta' in config else 1.0
        self.theta = config['theta'] if 'theta' in config else 1.0

    def forward(self, u):
        inner_vals = self.beta * (u - self.theta)

        large_val_indices = (inner_vals >= 500.0)
        result = self.gamma * torch.log(1.0 + torch.exp(inner_vals))
        result[large_val_indices] = self.gamma * inner_vals[large_val_indices]
        return result

    def backward(self, ctx, grad_output):
        input, = ctx.saved_tensors

        inner_vals = self.beta * (input - self.theta)
        non_large_neg_val_indices = (inner_vals >= -500.0)
        grads = torch.zeros(input.dims)
        grads[non_large_neg_val_indices] = self.gamma * self.beta / (1.0 + torch.exp(-inner_vals[non_large_neg_val_indices]))

        grad_input = grad_output * grads
        return grad_input


class TargetNetworkDataset(Dataset):
    def __init__(self, input_sequence_data_path, target_network_weights_path, transfer_function):
        target_network_forward_weights_list = get_target_network_forward_weights_list(target_network_weights_path)
        numpy_input_sequence = np.load(input_sequence_data_path)
        self.input_sequence = torch.tensor(numpy_input_sequence).float()
        self.output_sequence = torch.tensor(compute_non_linear_transform(numpy_input_sequence,
                                                           transfer_function,
                                                           target_network_forward_weights_list)
        ).float()

    def __len__(self):
        return len(self.input_sequence)

    def __getitem__(self, index):
        return self.input_sequence[index], self.output_sequence[index]


class MSERateLoss(nn.Module):
    def __init__(self, activation_function, reduction='mean'):
        super(MSERateLoss, self).__init__()
        self.reduction = reduction
        self.activation_function = activation_function

    def forward(self, input, target):
        return mse_loss(self.activation_function.forward(input),
                        self.activation_function(target),
                        reduction=self.reduction)


transfer_function = create_transfer_function(config=transfer_function_config)

target_network_train_dataset = TargetNetworkDataset(
    input_sequence_data_path='./datasets/target_network/train_input_sequence_size20_examples500.npy',
    target_network_weights_path='./target_network_weights/4_layer_sf_2x6x10/',
    transfer_function=transfer_function)

target_network_test_dataset = TargetNetworkDataset(
    input_sequence_data_path='./datasets/target_network/test_input_sequence_size20_examples500.npy',
    target_network_weights_path='./target_network_weights/4_layer_sf_2x6x10/',
    transfer_function=transfer_function)

activation_function = SoftRectifyTransferFunction(config=transfer_function_config)

#model = nn.Sequential(OrderedDict([
#                      ('fc1', nn.Linear(30, 50, bias=False)),
#                      ('activation1', activation_function),
#                      ('fc2', nn.Linear(50, 10, bias=False))]))

model = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(20, 20, bias=False)),
                      ('activation1', activation_function),
                      ('fc2', nn.Linear(20, 20, bias=False)),
                      ('activation2', activation_function),
                      ('fc3', nn.Linear(20, 10, bias=False))]))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.uniform_(m.weight, -1.0, 1.0)

model.apply(init_weights)

device = torch.device("cpu")
eval_criterion = MSERateLoss(activation_function)
train_criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.009)

train_loss_list = []
test_loss_list = []

best_test_loss = -np.inf
best_test_loss_epoch = None
best_test_loss_model = None

model.eval()
train_eval_loss = 0.0
for _, data in enumerate(target_network_train_dataset, 0):
    inputs, labels = data
    outputs = model(inputs)
    eval_loss = eval_criterion(outputs, labels)
    train_eval_loss += eval_loss.item()
average_train_loss = train_eval_loss / len(target_network_train_dataset)

test_loss = 0.0
for _, data in enumerate(target_network_test_dataset, 0):
    inputs, labels = data
    outputs = model(inputs)
    eval_loss = eval_criterion(outputs, labels)
    test_loss += eval_loss.item()

average_test_loss = test_loss / len(target_network_test_dataset)
print('Initial train_loss={}, test_loss={}'.format(average_train_loss, average_test_loss))

num_epochs = 200
for epoch in range(num_epochs):  # loop over the dataset multiple times
    train_eval_loss = 0.0
    model.train()
    for _, data in enumerate(target_network_train_dataset, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)

        train_loss = train_criterion(outputs, labels)
        train_loss.backward()
        eval_loss = eval_criterion(outputs, labels)
        optimizer.step()

        # print statistics
        train_eval_loss += eval_loss.item()
    average_train_eval_loss = train_eval_loss / len(target_network_train_dataset)
    train_loss_list += [average_train_eval_loss]

    model.eval()
    test_loss = 0.0

    first_outputs = []
    first_targets = []
    for _, data in enumerate(target_network_test_dataset, 0):
        inputs, labels = data

        outputs = model(inputs)
        first_outputs += [outputs[0]]
        first_targets += [labels[0]]
        eval_loss = eval_criterion(outputs, labels)

        test_loss += eval_loss.item()

    average_test_loss = test_loss / len(target_network_test_dataset)
    test_loss_list += [average_test_loss]

    if average_test_loss >= best_test_loss:
        best_test_loss = average_test_loss
        best_test_loss_epoch = epoch
        torch.save(model, './saved_models/standard_neural_network.pkl')

    if epoch % 1 == 0:
        print('Epoch {} train_loss={}, test_loss={}'.format(epoch + 1,
                                                            average_train_eval_loss,
                                                            average_test_loss))

print('Finished Training')
import matplotlib.pyplot as plt

plt.figure()
plt.plot(range(num_epochs), train_loss_list)
plt.plot(range(num_epochs), test_loss_list)
plt.figure()
plt.plot(range(500), first_outputs, label='output')
plt.plot(range(500), first_targets, label='output')
plt.show()
#print(loss)