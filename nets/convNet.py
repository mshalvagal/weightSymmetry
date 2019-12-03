import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.net2net import widen_v1, widen_v2

class simpleConvNet(nn.Module):

    def __init__(self, num_neurons=40, device='cpu'):
        super(simpleConvNet, self).__init__()

        self.device = device
        self.num_neurons = num_neurons
        
        self.conv1 = nn.Conv2d(1, num_neurons, kernel_size=5)
        self.conv2 = nn.Conv2d(num_neurons, 4, kernel_size=5)
        self.fc1 = nn.Linear(64, 10)
        self.fc2 = nn.Linear(10, 10)

        self.to(device)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)
    
    def grow_network(self, symmetry_break_method='noise'):
        
        self.conv1, self.conv2, _ = widen_v2(self.conv1, self.conv2, self.num_neurons*2)
        # self.fc1, self.fc2 = widen_v2(self.fc1, self.fc2, self.num_neurons*2)
        self.num_neurons *= 2
        self.to(self.device)


class CIFARConvNet(nn.Module):
    def __init__(self, device='cpu', big_net=False):
        super(CIFARConvNet, self).__init__()

        self.device = device

        filter_sizes = [8, 16, 32]
        if big_net:
            filter_sizes = [12, 24, 48]

        self.conv1 = nn.Conv2d(3, filter_sizes[0], 3, padding=1)
        self.bn1 = nn.BatchNorm2d(filter_sizes[0])
        self.pool1 = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(filter_sizes[0], filter_sizes[1], 3, padding=1)
        self.bn2 = nn.BatchNorm2d(filter_sizes[1])
        self.pool2 = nn.MaxPool2d(3, 2)
        self.conv3 = nn.Conv2d(filter_sizes[1], filter_sizes[2], 3, padding=1)
        self.bn3 = nn.BatchNorm2d(filter_sizes[2])
        self.pool3 = nn.AvgPool2d(5, 1)
        self.fc1 = nn.Linear(filter_sizes[2] * 3 * 3, 10)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, np.sqrt(2. / n))
        #         m.bias.data.fill_(0.0)
        #     if isinstance(m, nn.Linear):
        #         m.bias.data.fill_(0.0)

        self.to(device)

    def forward(self, x):
        try:
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.pool2(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = F.relu(x)
            x = self.pool3(x)
            x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
            x = self.fc1(x)
            return F.log_softmax(x)
        except RuntimeError:
            print(x.size())

    def grow_network(self):
        self.conv1, self.conv2, _ = widen_v2(self.conv1, self.conv2, 12,
                                             bnorm=self.bn1, weight_norm=True)
        self.conv2, self.conv3, _ = widen_v2(self.conv2, self.conv3, 24,
                                             bnorm=self.bn2, weight_norm=True)
        self.conv3, self.fc1, _ = widen_v2(self.conv3, self.fc1, 48,
                                           bnorm=self.bn3, weight_norm=True)

        self.to(self.device)