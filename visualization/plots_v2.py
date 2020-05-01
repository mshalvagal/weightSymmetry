import matplotlib.pyplot as plt
import os
import numpy as np
from visualization.plot_utils import sorted_subset_weight_pairs, numpy_to_mpl_hist

def scrollable_loss_vs_hist(parent_dir, run_num, ylim=None, net2net=False, steps=1):
    
    x = np.load(os.path.join(parent_dir, 'cosine_dists_hist_' + str(run_num) + '.npy'))
    loss_curve = np.load(parent_dir + 'loss_curve_' + str(run_num) + '.npy')

    offset = 0
    if net2net:
        offset = len(loss_curve)//2 - 1
    
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
        ax1.loglog(loss_curve)
        ax1.loglog(curr_pos  + offset, loss_curve[curr_pos + offset], marker='o', label='batch ' + str(curr_pos))
        ax1.legend()

        ax2.cla()
        hist, center, width = numpy_to_mpl_hist(x[curr_pos])

        ax2.bar(center, hist, align='center', width=width)
        ax2.set_xlim([-1, 1])
        if ylim is not None:
            ax2.set_ylim([0, ylim])
        fig.canvas.draw()

    fig = plt.figure()
    fig.canvas.mpl_connect('key_press_event', key_event)
    ax1 = fig.add_subplot(211)
    ax1.loglog(loss_curve)
    ax1.loglog(1 + offset, loss_curve[1 + offset], marker='o')

    ax2 = fig.add_subplot(212)
    hist, center, width = numpy_to_mpl_hist(x[0])
    ax2.bar(center, hist, align='center', width=width)
    ax2.set_xlim([-1, 1])
    if ylim is not None:
        ax2.set_ylim([0, ylim])
    plt.show()

def initial_vs_final_overlap(parent_dir, run_num, subset_plot=None):
    cos_dists = np.load(os.path.join(parent_dir, 'cos_dists_' + str(run_num) + '.npy'))
    num_neurons = cos_dists.shape[1]
    iu2 = np.triu_indices(num_neurons, k=1)

    x = cos_dists[0][iu2]
    plt.figure(figsize=(8,5))
    plt.tick_params(labelsize=20)

    if subset_plot is None:
        plt.scatter(x, cos_dists[-1][iu2])
        plt.xlabel('Initial overlap', fontsize=25)
        plt.ylabel('Final overlap', fontsize=25)
    else:
        low_overlap_pair_indices, high_overlap_pair_indices = sorted_subset_weight_pairs(x, iu2, num_pairs=subset_plot)
        
        plt.scatter(x[high_overlap_pair_indices], cos_dists[-1][iu2][high_overlap_pair_indices])
        plt.scatter(x[low_overlap_pair_indices], cos_dists[-1][iu2][low_overlap_pair_indices])
        plt.xlabel('Initial overlap', fontsize=25)
        plt.ylabel('Final overlap', fontsize=25)

def MSD_plot(parent_dir, run_num, max_time_lag, start_batch=0, end_batch=-1):
    
    cd_ranked = np.load(os.path.join(parent_dir, 'cosine_dists_diff_' + str(run_num) + '.npy'))[start_batch:end_batch]
    
    msd_low = np.zeros(max_time_lag)
    msd_high = np.zeros(max_time_lag)
    se_low = np.zeros(max_time_lag)
    se_high = np.zeros(max_time_lag)

    for i in range(1,max_time_lag):
        sq_diff = ((cd_ranked[i:] - cd_ranked[:-i])**2).mean(axis=0)
        # sq_diff = ((cd_ranked[i:] - cd_ranked[0])**2).mean(axis=0)

        n_pairs = len(sq_diff)//2
        msd_low[i] = np.mean(sq_diff[:n_pairs])
        se_low[i] = np.std(sq_diff[:n_pairs])/(np.sqrt(n_pairs))
        msd_high[i] = np.mean(sq_diff[n_pairs:])
        se_high[i] = np.std(sq_diff[n_pairs:])/(np.sqrt(n_pairs))
    
    plt.figure(figsize=(8,5))

    plt.plot(np.arange(max_time_lag), msd_low, label='low_overlap_pairs', linewidth=3)
    plt.fill_between(np.arange(max_time_lag), msd_low-se_low, msd_low+se_low, alpha=0.25)

    plt.plot(np.arange(max_time_lag), msd_high, label='high_overlap_pairs', linewidth=3)
    plt.fill_between(np.arange(max_time_lag), msd_high-se_high, msd_high+se_high, alpha=0.25)

    plt.title('Mean MSD')
    plt.xlabel('Time Lag', fontsize=25)
    plt.ylabel('Mean MSD', fontsize=25)
    plt.tick_params(labelsize=20)
    plt.legend(fontsize=20)
