import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np
import argparse
import os

from nets.simpleNet import simpleFCNet

def train(args, net, train_loader, criterion, optimizer, scheduler=None, device='cpu', test_loader=None):
    losses = np.zeros(args.epochs)
    training_acc = np.zeros(args.epochs)

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        
        if scheduler is not None:
            scheduler.step()

        for batch_idx, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            losses[epoch] += loss.item()        
            acc = torch.sum(torch.argmax(outputs, dim=1)==labels).item()/train_loader.batch_size

            training_acc[epoch] += acc

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTraining accuracy: {:.2f}%'.format(
                    epoch + 1, batch_idx * train_loader.batch_size, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(), acc*100))
        
        if test_loader is not None:
            val_loss, val_acc = test(net, test_loader, criterion)
            print('End of epoch {}: Validation loss: {:.6f}\tValidation accuracy: {:.2f}%'.format(
                epoch + 1, val_loss, val_acc*100))

    losses /= len(train_loader)
    training_acc /= len(train_loader)
    
    return losses, training_acc

def test(net, test_loader, criterion, device='cpu'):
    val_acc = 0.0
    val_loss = 0.0

    for batch_idx, data in enumerate(test_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
#         inputs = inputs.view(inputs.shape[0], -1)
        outputs = net(inputs)
        
        batch_loss = criterion(outputs, labels)

        val_loss += batch_loss.item()
        val_acc += torch.sum(torch.argmax(outputs, dim=1)==labels).item()/test_loader.batch_size
    
    val_loss /= len(test_loader)
    val_acc /= len(test_loader)
    
    return val_loss, val_acc

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Net2WiderNet')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--num-runs=5', type=int, default=10, metavar='N',
                        help='number of repeated runs (default: 5)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--num-hidden-neurons', type=int, default=10, metavar='N',
                        help='number of neurons in hidden layer (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--logdir', type=str, default='demo',
                        help='directory to store training curves')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--smart-init', action='store_true', default=False,
                        help='whether to start with weights from a smaller trained network')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    if args.smart_init:
        args.epochs = int(args.epochs/2)
        model = simpleFCNet(num_neurons=int(args.num_hidden_neurons/2), device=device)
    else:
        model = simpleFCNet(num_neurons=args.num_hidden_neurons, device=device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    losses, training_acc = train(args, model, train_loader, criterion, optimizer, device=device, test_loader=test_loader)

    if args.smart_init:
        print("Growing network and continuing training")
        model.grow_network()
        loss_new, acc_new = train(args, model, train_loader, criterion, optimizer, device=device, test_loader=test_loader)

        losses = np.append(losses, loss_new)
        training_acc = np.append(training_acc, acc_new)

    save_dir = os.path.join('logs', str(args.num_hidden_neurons) + ' neurons')
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'loss_curve'), losses)
    np.save(os.path.join(save_dir, 'acc_curve'), training_acc)


if __name__ == '__main__':
    main()