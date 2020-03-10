import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.net2net import widen_v1, widen_v2

class simpleFCNet(nn.Module):

    def __init__(self, num_neurons=5, device='cpu', dropout=False):
        super(simpleFCNet, self).__init__()

        self.device = device
        self.num_neurons = num_neurons
        self.dropout = dropout
        
        self.dropout_layer = nn.Dropout(p=0.5)
        self.dense_1 = nn.Linear(784, num_neurons)
        self.dense_out = nn.Linear(num_neurons, 10)
        self.to(device)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.dense_1(x))
        if self.dropout:
            x = self.dropout_layer(x)
        x = F.softmax(self.dense_out(x), dim=1)
        return x
    
    def grow_network(self, symmetry_break_method='simple', noise_var=0.5, weight_norm=False):
        
        self.dense_1, self.dense_out = widen_v1(self.dense_1, self.dense_out, symmetry_break_method, noise_var, weight_norm)

        self.num_neurons *= 2
        self.to(self.device)


class FCNet(nn.Module):

    def __init__(self, num_neurons=5, device='cpu', dropout=False, trained_net_file=None, freeze_feature_extraction=True):
        super(FCNet, self).__init__()

        self.device = device
        self.num_neurons = num_neurons
        self.dropout = dropout
        
        self.dropout_layer = nn.Dropout(p=0.5)
        if trained_net_file is not None:
            trained_net = torch.load(trained_net_file)
            dict_trained_params = dict(trained_net.dense_1.named_parameters())
            for name, param in self.dense_1.named_parameters():
                if name in dict_trained_params:
                    print('Copying pretrained ' + name)
                    param.requires_grad = False
                    param.copy_(dict_trained_params[name].data)
        self.dense_1 = nn.Linear(784, 20)
        self.dense_2 = nn.Linear(20, num_neurons)
        self.dense_out = nn.Linear(num_neurons, 10)
        self.to(device)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.dense_1(x))
        if self.dropout:
            x = self.dropout_layer(x)
        x = F.relu(self.dense_2(x))
        if self.dropout:
            x = self.dropout_layer(x)
        x = F.softmax(self.dense_out(x), dim=1)
        return x
    
    def grow_network(self, symmetry_break_method='simple'):
        
        self.dense_1, self.dense_2 = widen_v1(self.dense_1, self.dense_2, symmetry_break_method)
        self.dense_2, self.dense_out = widen_v1(self.dense_2, self.dense_out, symmetry_break_method)

        self.num_neurons *= 2

        self.to(self.device)
