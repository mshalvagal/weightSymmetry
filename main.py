import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import os
import copy
import argparse
import yaml
import shutil
import numpy as np

from nets.simpleNet import simpleFCNet, FCNet
from nets.convNet import simpleConvNet

from utils.utils import gradient_projection_norms, pairwise_cos_dist
from metrics.metrics import Metrics, CosineStats

class Experiment():

    def __init__(self, settings):

        self.logparams = settings['log-params']
        self.hyperparams = settings['hyperparams']
        self.network_config = settings['network-config']
        self.experiment_params = settings['experiment-params']

        self.num_runs = settings['num-runs']
        self.num_epochs = settings['epochs']

        self._generate_output_directory()
        self._setup_metrics()
        
        # Create data loaders
        use_cuda = torch.cuda.is_available() and not settings['no-cuda']
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        if use_cuda:
            print('GPU accelerated training enabled')
        self.train_loader, self.test_loader = self._create_data_loaders(use_cuda)

        # Don't judge me for this :P
        self.network_generator = FCNet if self.network_config['two-hidden-layers'] else simpleFCNet

    def _create_data_loaders(self, use_cuda):
        '''
            Creates the training and test data loaders
            Nothing to see here
        '''
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/mnist', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=self.hyperparams['batch-size'], shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/mnist', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=self.hyperparams['test-batch-size'], shuffle=True, **kwargs)
        
        return train_loader, test_loader
    
    def _generate_output_directory(self):
        '''
            Function to generate the output folder name for the logs
            Obviously, I have tried and failed to keep this sane and manageable
            And then created this function to hide the madness
        '''
        parent_dir = os.path.join(self.logparams['logdir'], '2 hidden layers' if self.network_config['two-hidden-layers'] else '1 hidden layer')
        parent_dir = os.path.join(parent_dir, str(self.network_config['num-hidden-neurons']) + ' neurons')

        description = 'train_from_scratch'
        suffix = ''
        if self.experiment_params['network-growth']['flag']:
            if self.experiment_params['network-growth']['at-start-flag']:
                suffix = '_net2net_at_start'
            else:
                description = 'smart_init'
                suffix = '_' + self.experiment_params['network-growth']['method'] + '_' + str(self.experiment_params['network-growth']['noise-var'])
        
        if self.experiment_params['regularizer']['ortho-reg']:
            description += '_' + 'ortho_reg'
        if self.experiment_params['regularizer']['dropout']:
            description += '_' + 'dropout'

        self.logdir = os.path.join(parent_dir, description + suffix)
        os.makedirs(self.logdir, exist_ok=True)

    def _setup_metrics(self):
        self.metrics_list = {}
        if self.logparams['metrics']['loss']:
            self.metrics_list['loss'] = Metrics('loss_curve')
        if self.logparams['metrics']['accuracy']:
            self.metrics_list['accuracy'] = Metrics('acc_curve')
        if self.logparams['metrics']['cosine-dists']:
            if not self.logparams['metrics']['cosine-dists']['stats-only']:
                self.metrics_list['cosine_dists'] = Metrics('cos_dists')
            self.metrics_list['cosine_dists_hist'] = Metrics('cosine_dists_hist')
            self.metrics_list['cosine_dists_diff'] = Metrics('cosine_dists_diff')
            self.metrics_list['cosine_dists_mean'] = Metrics('cosine_dists_mean')
        if self.logparams['metrics']['gradient-projections']:
            self.metrics_list['mean_grad'] = Metrics('mean_grad')
            self.metrics_list['diff_grad'] = Metrics('diff_grad')
        if self.logparams['metrics']['test-accuracy']:
            self.metrics_list['test_accuracy'] = Metrics('test_accuracy')
        if self.logparams['metrics']['weights']:
            for i in range(self.num_runs):
                os.makedirs(os.path.join(self.logdir, 'weight_history', 'run_' + str(i)), exist_ok=True)

    def run_experiment(self):

        for i in range(self.num_runs):

            self.current_run = i
            print('Beginning run ' + str(i))

            if self.experiment_params['network-growth']['flag']:
                self.net = self.network_generator(num_neurons=self.network_config['num-hidden-neurons']//2, device=self.device,\
                    dropout=self.experiment_params['regularizer']['dropout'])
            else:
                trained_net_file = self.experiment_params['trained-feature-extractor']['path'] if self.experiment_params['trained-feature-extractor']['flag'] else None
                self.net = self.network_generator(num_neurons=self.network_config['num-hidden-neurons'], device=self.device,\
                    dropout=self.experiment_params['regularizer']['dropout'], trained_net_file=trained_net_file)
            print('Model definition')
            print(self.net)
            self.criterion = nn.CrossEntropyLoss()

            self.optimizer = optim.SGD(self.net.parameters(), lr=self.hyperparams['learning-rate'], momentum=self.hyperparams['momentum'])
            # self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

            if not self.experiment_params['network-growth']['flag'] and not self.experiment_params['network-growth']['at-start-flag']:
                self._train(self.num_epochs)
            elif self._teacher_exists(i, self.experiment_params['network-growth']['epochs-before-growth']):
                print("Teacher exists, loading teacher")
                teacher_dir = os.path.join(os.path.dirname(self.logdir), 'teacher_' + \
                    str(self.experiment_params['network-growth']['epochs-before-growth']) + '_epochs')
                for metric in self.metrics_list:
                    self.metrics_list[metric].load_from_disk(teacher_dir, i)
                self.net = torch.load(os.path.join(teacher_dir, 'trained_network_' + str(i) + '.pt'))
                self.net.num_neurons = self.network_config['num-hidden-neurons']//2
            else:
                print("Teacher does not exist, training from start")
                if self.experiment_params['network-growth']['at-start-flag']:
                    self.experiment_params['network-growth']['epochs-before-growth'] = 0
                self._train(self.experiment_params['network-growth']['epochs-before-growth'], teacher_training=True)
                teacher_dir = os.path.join(os.path.dirname(self.logdir), 'teacher_' + \
                    str(self.experiment_params['network-growth']['epochs-before-growth']) + '_epochs')
                os.makedirs(teacher_dir, exist_ok=True)
                for metric in self.metrics_list:
                    self.metrics_list[metric].save_to_disk(teacher_dir, i)
                torch.save(self.net, os.path.join(teacher_dir, 'trained_network_' + str(i) + '.pt'))

            if self.experiment_params['network-growth']['flag']:
                print("Growing network and continuing training")

                net_ = self.network_generator(num_neurons=self.network_config['num-hidden-neurons'], device=self.device,\
                    dropout=self.experiment_params['regularizer']['dropout'])
                net_ = copy.deepcopy(self.net)

                del self.net
                self.net = net_
                self.net.grow_network(symmetry_break_method=self.experiment_params['network-growth']['method'], \
                    noise_var=self.experiment_params['network-growth']['noise-var'],
                    weight_norm = self.experiment_params['network-growth']['weight-norm'])
                print('New model definition')

                if self.experiment_params['regularizer']['dropout']:
                    self.net.dropout = True

                print(self.net)

                val_loss, val_acc = self._test()
                print('After growing: Validation loss: {:.6f}\tValidation accuracy: {:.2f}%'.format(
                    val_loss, val_acc*100))

                self._reset_metrics_on_growth()
                self.optimizer = optim.SGD(self.net.parameters(), lr=self.hyperparams['learning-rate'], momentum=self.hyperparams['momentum'])
                self._train(self.num_epochs - self.experiment_params['network-growth']['epochs-before-growth'], \
                    start_epoch=self.experiment_params['network-growth']['epochs-before-growth'])

            for metric in self.metrics_list:
                self.metrics_list[metric].save_to_disk(self.logdir, i)

    def _train(self, num_epochs, start_epoch=0, teacher_training=False):
        self.test_acc = []
        self.cosine_dist_hists = []

        log_interval = self.logparams['log-interval']
        self.net.train()

        layer_of_interest = self.net.dense_2 if self.network_config['two-hidden-layers'] else self.net.dense_1
        if self.logparams['metrics']['cosine-dists']['flag']:
            cosine_stats = CosineStats(layer_of_interest, self.logparams['metrics']['cosine-dists']['population-size'], teacher_training)
            hist, cd = cosine_stats.initial_stats()
            self.metrics_list['cosine_dists_hist'].log_vals(hist)
            if not teacher_training:
                self.metrics_list['cosine_dists_diff'].log_vals(cd)

        for epoch in range(num_epochs):  # loop over the dataset multiple times

            # if self.scheduler is not None:
            #     self.scheduler.step()

            running_loss = 0.0
            running_acc = 0.0
            if self.logparams['metrics']['cosine-dists']['flag'] and not self.logparams['metrics']['cosine-dists']['stats-only']:
                self.metrics_list['cosine_dists'].log_vals(cosine_stats.cosine_dists)
            if self.logparams['metrics']['weights'] and not teacher_training:
                torch.save(self.net, os.path.join(self.logdir, 'weight_history', 'run_' + str(self.current_run), 'epoch_'+ str(epoch + start_epoch +  1) + '.pt'))

            for batch_idx, data in enumerate(self.train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()

                if self.experiment_params['regularizer']['ortho-reg']:
                    w_in = self.net.dense_1.weight.data.clone()
                    w_in.requires_grad = False
                    w_in = w_in/w_in.norm(dim=1, keepdim=True)
                    regularizer_weight = torch.tensor([10.0]).to(w_in.device)
                    theta = torch.exp(regularizer_weight * torch.mm(w_in, w_in.transpose(0, 1)))
                    theta.diagonal(dim1=-2, dim2=-1).zero_()
                    theta = regularizer_weight * theta / (theta + torch.exp(regularizer_weight))

                    self.net.dense_1.weight.grad += 0.1*torch.mm(theta, w_in)

                self.optimizer.step()

                # print('Gradients for new weights')
                # print(torch.norm(net.dense_1.weight.grad[10:]))
                # print('Gradients for old weights')
                # print(torch.norm(net.dense_1.weight.grad[:10]))

                # print statistics
                running_loss += loss.item()
                acc = torch.sum(torch.argmax(outputs, dim=1)==labels).item()/self.train_loader.batch_size
                running_acc += acc

                if batch_idx % log_interval == (log_interval - 1):
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTraining accuracy: {:.2f}%'.format(
                        epoch + start_epoch + 1, batch_idx * self.train_loader.batch_size, len(self.train_loader.dataset),
                        100. * batch_idx / len(self.train_loader), loss.item(), acc*100))

                    if self.logparams['metrics']['loss']:
                        self.metrics_list['loss'].log_vals(loss.item())
                    if self.logparams['metrics']['accuracy']:
                        self.metrics_list['accuracy'].log_vals(100.0*acc)
                    if self.logparams['metrics']['gradient-projections']:
                        mean_grad, diff_grad = gradient_projection_norms(layer_of_interest)
                        self.metrics_list['mean_grad'].log_vals(mean_grad)
                        self.metrics_list['mean_grad'].log_vals(diff_grad)
                    if self.logparams['metrics']['cosine-dists']['flag']:
                        hist, cd, cm = cosine_stats.compute_stats()
                        self.metrics_list['cosine_dists_hist'].log_vals(hist)
                        if not teacher_training:
                            self.metrics_list['cosine_dists_diff'].log_vals(cd)
                            self.metrics_list['cosine_dists_mean'].log_vals(cm)

                    running_loss = 0.0
                    running_acc = 0.0

            if self.test_loader is not None:
                val_loss, val_acc = self._test()
                print('End of epoch {}: Validation loss: {:.6f}\tValidation accuracy: {:.2f}%'.format(
                    epoch + start_epoch + 1, val_loss, val_acc*100))
                
                if self.logparams['metrics']['test-accuracy']:
                    self.metrics_list['test_accuracy'].log_vals(val_acc*100)

        # mean_grad_proj = np.vstack([np.array(mean_grad_proj).mean(axis=0), np.array(mean_grad_proj).std(axis=0)])
        # diff_grad_proj = np.vstack([np.array(diff_grad_proj).mean(axis=0), np.array(diff_grad_proj).std(axis=0)])

    def _test(self):
        val_acc = 0.0
        val_loss = 0.0

        self.net.eval()

        for batch_idx, data in enumerate(self.test_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            # inputs = inputs.view(inputs.shape[0], -1)
            outputs = self.net(inputs)
            
            batch_loss = self.criterion(outputs, labels)

            val_loss += batch_loss.item()
            val_acc += torch.sum(torch.argmax(outputs, dim=1)==labels).item()/self.test_loader.batch_size
        
        val_loss /= len(self.test_loader)
        val_acc /= len(self.test_loader)
        
        return val_loss, val_acc

    def _teacher_exists(self, i, trained_epochs):

        for metric in self.metrics_list:
            if not os.path.isfile(os.path.join(os.path.dirname(self.logdir), 'teacher_' + str(trained_epochs) + '_epochs', \
                self.metrics_list[metric].f_name + '_' + str(i) + '.npy')):
                return False

        if not os.path.isfile(os.path.join(os.path.dirname(self.logdir), 'teacher_' + str(trained_epochs) + '_epochs', \
            'trained_network_' + str(i) + '.pt')):
            return False

        return True

    def _reset_metrics_on_growth(self):
        # Flush cosine similarity values (because the size changes on growing the network)
        if self.logparams['metrics']['cosine-dists']:
            if not self.logparams['metrics']['cosine-dists']['stats-only']:
                self.metrics_list['cosine_dists'].reset()
            self.metrics_list['cosine_dists_hist'].reset()
            self.metrics_list['cosine_dists_diff'].reset()
            self.metrics_list['cosine_dists_mean'].reset()
        if self.logparams['metrics']['gradient-projections']:
            self.metrics_list['mean_grad'].reset()
            self.metrics_list['diff_grad'].reset()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Weight Symmetry experiment settings')
    parser.add_argument('--experiment-file', type=str, default='experiment_settings.yaml',
                        help='settings file for experiment')
    args = parser.parse_args()

    with open(args.experiment_file) as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    torch.manual_seed(settings['random-seed'])
    experiment = Experiment(settings)
    experiment.run_experiment()
    shutil.copy2(args.experiment_file, experiment.logdir)
