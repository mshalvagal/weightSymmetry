import torch
import torch.nn as nn
import torch.nn.functional as F


class simpleFCNet(nn.Module):

    def __init__(self, num_neurons=5, device='cpu'):
        super(simpleFCNet, self).__init__()

        self.num_neurons = num_neurons
        self.device = device
        
        self.dense_1 = nn.Linear(784, num_neurons)
        self.dense_out = nn.Linear(num_neurons, 10)
        self.to(device)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.dense_1(x))
        x = F.softmax(self.dense_out(x), dim=1)
        return x
    
    def grow_network(self, mode='simple'):
        
        permutation_array = torch.arange(self.num_neurons*2)
        # permutation_array[self.num_neurons:] = torch.randint(self.num_neurons, (self.num_neurons, ))
        permutation_array[self.num_neurons:] = torch.randperm(self.num_neurons)

        new_dense_1 = nn.Linear(784, self.num_neurons*2)
        new_dense_1.weight.data = self.dense_1.weight.data[permutation_array]
        new_dense_1.bias.data = self.dense_1.bias.data[permutation_array]

        new_dense_out = nn.Linear(self.num_neurons*2, 10)
        new_dense_out.weight.data = self.dense_out.weight.data[:, permutation_array]
        new_dense_out.bias.data = self.dense_out.bias.data

        new_dense_out.weight.data[:self.num_neurons] *= 2
        new_dense_out.bias.data[:self.num_neurons] *= 2
        new_dense_out.weight.data[self.num_neurons:] *= -1
        new_dense_out.bias.data[self.num_neurons:] *= -1

        new_dense_1.to(self.device)
        new_dense_out.to(self.device)

        self.dense_1 = new_dense_1
        self.dense_out = new_dense_out
