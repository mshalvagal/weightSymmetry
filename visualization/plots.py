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
            if f_name.startswith('loss') and f_name.endswith('.npy'):
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
            plt.fill_between(np.arange(losses.shape[1])*log_interval, np.mean(losses, 0)-se, np.mean(losses, 0)+se, alpha=0.25)
        else:
            plt.plot(np.arange(losses.shape[1])*log_interval, losses[0], label=method)

        plt.subplot(2, 1, 2)
        plt.xlabel('Number of minibatches passed', fontsize=10)
        plt.ylabel('Training Accuracy', fontsize=10)
        if mode=='mean':
            plt.plot(np.arange(acc.shape[1])*log_interval, np.mean(acc, 0), label=method)
            se = np.std(acc,0)/np.sqrt(acc.shape[0])
            plt.fill_between(np.arange(acc.shape[1])*log_interval, np.mean(acc, 0)-se, np.mean(acc, 0)+se, alpha=0.25)
        else:
            plt.plot(np.arange(losses.shape[1])*log_interval, losses[0], label=method)
    
    if teacher_dir is not None:
        losses = []
        acc = []
        for f_name in os.listdir(teacher_dir):
            if os.path.isdir(os.path.join(teacher_dir, f_name)):
                # skip directories
                continue
            if f_name.startswith('loss') and f_name.endswith('.npy'):
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
            plt.fill_between(np.arange(losses.shape[1])*log_interval, np.mean(losses, 0)-se, np.mean(losses, 0)+se, alpha=0.25)
        else:
            plt.plot(np.arange(losses.shape[1])*log_interval, losses[0], label='teacher')

        plt.subplot(2, 1, 2)
        plt.xlabel('Number of minibatches passed', fontsize=10)
        plt.ylabel('Training Accuracy', fontsize=10)
        if mode=='mean':
            plt.plot(np.arange(acc.shape[1])*log_interval, np.mean(acc, 0), label='teacher')
            se = np.std(acc,0)/np.sqrt(acc.shape[0])
            plt.fill_between(np.arange(acc.shape[1])*log_interval, np.mean(acc, 0)-se, np.mean(acc, 0)+se, alpha=0.25)
        else:
            plt.plot(np.arange(losses.shape[1])*log_interval, losses[0], label='teacher')

    plt.subplot(2, 1, 1)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.legend()


def scrollable_loss_vs_hist(parent_dir, run_num, idxes=None, ylim=None, net2net=False, steps=1):
    
    x = np.load(os.path.join(parent_dir, 'cos_dists_' + str(run_num) + '.npy'))
    
    offset = 0

    loss_curve = np.load(parent_dir + 'loss_curve_' + str(run_num) + '.npy')
    
    num_neurons = x.shape[1]
    iu2 = np.triu_indices(num_neurons, k=1)
    
    if idxes is None:
        if net2net:
            idxes = np.arange(40)
            offset = 40
        else:
            idxes = np.arange(80)

    batch_idxes = [46*(i + offset) for i in idxes]

    # now the real code :) 
    global curr_pos
    curr_pos = 0

    def key_event(e):
        global curr_pos

        if e.key == "right":
            curr_pos = curr_pos + steps
        elif e.key == "left":
            curr_pos = curr_pos - steps
        else:
            return
        curr_pos = curr_pos % len(x)


        ax1.cla()
        ax1.plot(loss_curve)
        ax1.plot(curr_pos + offset*46, loss_curve[curr_pos + offset*46], marker='o', label='batch ' + str(curr_pos))
        ax1.legend()

        ax2.cla()
        plots = x[curr_pos]
        ax2.hist(plots[iu2], bins=20)
        ax2.set_xlim([-1, 1])
        if ylim is not None:
            ax2.set_ylim([0, ylim])
        fig.canvas.draw()

    fig = plt.figure()
    fig.canvas.mpl_connect('key_press_event', key_event)
    ax1 = fig.add_subplot(211)
    ax1.plot(loss_curve)
    ax1.plot(1 + offset*46, loss_curve[1 + offset*46], marker='o')

    ax2 = fig.add_subplot(212)
    plots = x[0]
    ax2.hist(plots[iu2], bins=20)
    ax2.set_xlim([-1, 1])
    if ylim is not None:
        ax2.set_ylim([0, ylim])
    plt.show()   