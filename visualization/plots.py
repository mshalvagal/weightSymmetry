import matplotlib.pyplot as plt
import os
import numpy as np

def plot_learning_curves(parent_dir, mode='mean', log_interval=20, teacher_dir=None):

    plt.figure(figsize=(15, 20))
    for method in os.listdir(parent_dir):
        if method == 'teacher':
            continue
        method_dir = os.path.join(parent_dir, method)
        losses = []
        acc = []
        for f_name in os.listdir(method_dir):
            if os.path.isdir(os.path.join(method_dir, f_name)):
                # skip directories
                continue
            if f_name.startswith('loss'):
                losses.append(np.load(os.path.join(method_dir, f_name)))
            elif f_name.startswith('acc'):
                acc.append(np.load(os.path.join(method_dir, f_name)))
        losses = np.array(losses)
        acc = np.array(acc)
        
        plt.subplot(2, 1, 1)
        plt.xlabel('Number of minibatches passed', fontsize=10)
        plt.ylabel('Training Loss', fontsize=10)
        if mode=='mean':
            plt.plot(np.arange(losses.shape[1])*log_interval, np.mean(losses, 0), label=method)
            se = np.std(losses,0)/np.sqrt(losses.shape[0])
            plt.fill_between(np.arange(losses.shape[1])*log_interval, np.mean(losses, 0)-se, np.mean(losses, 0)+se, alpha=0.5)
        else:
            plt.plot(np.arange(losses.shape[1])*log_interval, losses[0], label=method)

        plt.subplot(2, 1, 2)
        plt.xlabel('Number of minibatches passed', fontsize=10)
        plt.ylabel('Training Accuracy', fontsize=10)
        if mode=='mean':
            plt.plot(np.arange(acc.shape[1])*log_interval, np.mean(acc, 0), label=method)
            se = np.std(acc,0)/np.sqrt(acc.shape[0])
            plt.fill_between(np.arange(acc.shape[1])*log_interval, np.mean(acc, 0)-se, np.mean(acc, 0)+se, alpha=0.5)
        else:
            plt.plot(np.arange(losses.shape[1])*log_interval, losses[0], label=method)
    
    if teacher_dir is not None:
        losses = []
        acc = []
        for f_name in os.listdir(teacher_dir):
            if os.path.isdir(os.path.join(teacher_dir, f_name)):
                # skip directories
                continue
            if f_name.startswith('loss'):
                losses.append(np.load(os.path.join(teacher_dir, f_name)))
            elif f_name.startswith('acc'):
                acc.append(np.load(os.path.join(teacher_dir, f_name)))
        losses = np.array(losses)
        acc = np.array(acc)
        
        plt.subplot(2, 1, 1)
        plt.xlabel('Number of minibatches passed', fontsize=10)
        plt.ylabel('Training Loss', fontsize=10)
        if mode=='mean':
            plt.plot(np.arange(losses.shape[1])*log_interval, np.mean(losses, 0), label='teacher')
            se = np.std(losses,0)/np.sqrt(losses.shape[0])
            plt.fill_between(np.arange(losses.shape[1])*log_interval, np.mean(losses, 0)-se, np.mean(losses, 0)+se, alpha=0.5)
        else:
            plt.plot(np.arange(losses.shape[1])*log_interval, losses[0], label='teacher')

        plt.subplot(2, 1, 2)
        plt.xlabel('Number of minibatches passed', fontsize=10)
        plt.ylabel('Training Accuracy', fontsize=10)
        if mode=='mean':
            plt.plot(np.arange(acc.shape[1])*log_interval, np.mean(acc, 0), label='teacher')
            se = np.std(acc,0)/np.sqrt(acc.shape[0])
            plt.fill_between(np.arange(acc.shape[1])*log_interval, np.mean(acc, 0)-se, np.mean(acc, 0)+se, alpha=0.5)
        else:
            plt.plot(np.arange(losses.shape[1])*log_interval, losses[0], label='teacher')

    plt.subplot(2, 1, 1)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.legend()