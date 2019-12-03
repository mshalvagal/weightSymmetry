import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np
import argparse
import os
import copy

from nets.convNet import CIFARConvNet

def train(args, net, train_loader, criterion, optimizer, scheduler=None, device='cpu', test_loader=None, start_epoch=0, save_dir='demo'):
    losses = []
    training_acc = []

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        
        if scheduler is not None:
            scheduler.step()

        running_loss = 0.0
        running_acc = 0.0        

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

            # print('Gradients for new weights')
            # print(torch.norm(net.dense_1.weight.grad[10:]))
            # print('Gradients for old weights')
            # print(torch.norm(net.dense_1.weight.grad[:10]))

            # print statistics
            running_loss += loss.item()
            acc = torch.sum(torch.argmax(outputs, dim=1)==labels).item()/train_loader.batch_size
            running_acc += acc

            if batch_idx % args.log_interval == (args.log_interval - 1):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTraining accuracy: {:.2f}%'.format(
                    epoch + start_epoch + 1, batch_idx * train_loader.batch_size, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(), acc*100))

                losses.append(running_loss/(args.log_interval))
                training_acc.append(100.0*running_acc/args.log_interval)
                running_loss = 0.0
                running_acc = 0.0
        
        if test_loader is not None:
            val_loss, val_acc = test(net, test_loader, criterion, device=device)
            print('End of epoch {}: Validation loss: {:.6f}\tValidation accuracy: {:.2f}%'.format(
                epoch + start_epoch + 1, val_loss, val_acc*100))

        torch.save(net, os.path.join(save_dir, 'epoch_'+ str(epoch + start_epoch +  1) + '.pt'))

    losses = np.array(losses)
    training_acc = np.array(training_acc)
    
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

def teacher_exists(save_dir, i, epochs):

    if not os.path.isfile(os.path.join(save_dir, 'teacher', 'loss_curve_' + str(i) + '.npy')):
        return False
    if not os.path.isfile(os.path.join(save_dir, 'teacher', 'acc_curve_' + str(i) + '.npy')):
        return False
    if not os.path.isfile(os.path.join(save_dir, 'teacher', 'trained_network_' + str(i) + '.pt')):
        return False

    # losses = np.load(os.path.join(save_dir, 'teacher', 'loss_curve_' + str(i) + '.npy'))
    # training_acc = np.load(os.path.join(save_dir, 'teacher', 'acc_curve_' + str(i) + '.npy'))
    # if len(losses) != epochs or len(training_acc) != epochs:
    #     return False
    
    return True

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Net2WiderNet')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--num-runs', type=int, default=1, metavar='N',
                        help='number of repeated runs (default: 1)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--num-hidden-neurons', type=int, default=10, metavar='N',
                        help='number of neurons in hidden layer (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--logdir', type=str, default='logs/cifar10',
                        help='parent directory to store logs')
    parser.add_argument('--smart-init', action='store_true', default=False,
                        help='whether to start with weights from a smaller trained network')
    parser.add_argument('--big-net', action='store_true', default=False,
                        help='Flag to train big network from scratch')
    parser.add_argument('--symmetry-break-method', type=str, default='v2',
                        help='which method to use to break symmetry in new network')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print('GPU accelerated training enabled')

    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data/cifar10', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data/cifar10', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    if args.smart_init:
        args.epochs = int(args.epochs/2)

    net_definition = CIFARConvNet

    parent_dir = args.logdir
    save_dir = 'teacher'
    if args.big_net:
        save_dir = 'train_from_scratch'
    elif args.smart_init:
        save_dir = 'net2net'
    save_dir = os.path.join(args.logdir, save_dir)
    os.makedirs(save_dir, exist_ok=True)

    for i in range(args.num_runs):

        print('Beginning run ' + str(i))

        model = net_definition(device=device)
        if args.big_net:
            model = net_definition(device=device, big_net=True)
        print('Model definition')
        print(model)
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

        if not args.smart_init:
            losses, training_acc = train(args, model, train_loader, criterion, optimizer, device=device, test_loader=test_loader, 
                save_dir=os.path.join(save_dir, 'weight_history', 'run_' + str(i)))
        elif teacher_exists(parent_dir, i, args.epochs):
            print("Teacher exists, continuing training")
            losses = np.load(os.path.join(parent_dir, 'teacher', 'loss_curve_' + str(i) + '.npy'))
            training_acc = np.load(os.path.join(parent_dir, 'teacher', 'acc_curve_' + str(i) + '.npy'))

            num_batches = losses.shape[0]
            losses = losses[:num_batches//2]
            training_acc = training_acc[:num_batches//2]

            model = torch.load(os.path.join(parent_dir, 'teacher', 'trained_network_' + str(i) + '.pt'))
        else:
            print("Teacher does not exist, training from start")
            losses, training_acc = train(args, model, train_loader, criterion, optimizer, device=device, test_loader=test_loader, 
                save_dir=os.path.join(save_dir, 'weight_history', 'run_' + str(i)))
            
            os.makedirs(os.path.join(parent_dir, 'teacher'), exist_ok=True)
            np.save(os.path.join(parent_dir, 'teacher', 'loss_curve_' + str(i)), losses)
            np.save(os.path.join(parent_dir, 'teacher', 'acc_curve_' + str(i)), training_acc)
            torch.save(model, os.path.join(parent_dir, 'teacher', 'trained_network_' + str(i) + '.pt'))

        if args.smart_init:
            print("Growing network and continuing training")

            model_ = net_definition(device=device)
            model_ = copy.deepcopy(model)

            del model
            model = model_
            model.grow_network()
            print('New model definition')
            print(model)

            val_loss, val_acc = test(model, test_loader, criterion, device=device)
            print('After growing: Validation loss: {:.6f}\tValidation accuracy: {:.2f}%'.format(
                val_loss, val_acc*100))

            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
            loss_new, acc_new = train(args, model, train_loader, criterion, optimizer, device=device, test_loader=test_loader, 
                start_epoch = args.epochs, save_dir=os.path.join(save_dir, 'weight_history', 'run_' + str(i)))

            losses = np.append(losses, loss_new)
            training_acc = np.append(training_acc, acc_new)
        
        np.save(os.path.join(save_dir, 'loss_curve_' + str(i)), losses)
        np.save(os.path.join(save_dir, 'acc_curve_' + str(i)), training_acc)


if __name__ == '__main__':
    main()