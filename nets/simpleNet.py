import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.net2net import widen_v1, widen_v2

class simpleFCNet(nn.Module):

    def __init__(self, num_neurons=5, device='cpu'):
        super(simpleFCNet, self).__init__()

        self.device = device
        self.num_neurons = num_neurons

        self.dense_1 = nn.Linear(784, num_neurons)
        self.dense_out = nn.Linear(num_neurons, 10)
        self.to(device)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.dense_1(x))
        x = F.softmax(self.dense_out(x), dim=1)
        return x
    
    def grow_network(self, symmetry_break_method='simple', noise_var=0.5, weight_norm=False):
        
        if symmetry_break_method == 'v2':
            self.dense_1, self.dense_out, _ = widen_v2(self.dense_1, self.dense_out, self.num_neurons*2)
        else:
            self.dense_1, self.dense_out = widen_v1(self.dense_1, self.dense_out, symmetry_break_method, noise_var, weight_norm)

        self.num_neurons *= 2
        self.to(self.device)


class FCNet(nn.Module):

    def __init__(self, num_neurons=5, device='cpu'):
        super(FCNet, self).__init__()

        self.device = device
        self.num_neurons = num_neurons
        
        self.dense_1 = nn.Linear(784, num_neurons)
        self.dense_2 = nn.Linear(num_neurons, num_neurons)
        self.dense_out = nn.Linear(num_neurons, 10)
        self.to(device)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = F.softmax(self.dense_out(x), dim=1)
        return x
    
    def grow_network(self, symmetry_break_method='simple'):
        
        if symmetry_break_method == 'v2':
            self.dense_1, self.dense_2, _ = widen_v2(self.dense_1, self.dense_2, self.num_neurons*2)
            self.dense_2, self.dense_out, _ = widen_v2(self.dense_2, self.dense_out, self.num_neurons*2)
        else:
            self.dense_1, self.dense_2 = widen_v1(self.dense_1, self.dense_2, symmetry_break_method)
            self.dense_2, self.dense_out = widen_v1(self.dense_2, self.dense_out, symmetry_break_method)

        self.num_neurons *= 2

        self.to(self.device)
