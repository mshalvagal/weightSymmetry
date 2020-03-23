import os
import numpy as np
import torch

from utils.utils import pairwise_cos_dist
from visualization.plot_utils import sorted_subset_weight_pairs
from sklearn.metrics.pairwise import cosine_similarity

class Metrics():
    def __init__(self, f_name):
        self.f_name = f_name
        self.values = []

    def log_vals(self, value):
        self.values.append(value)

    def reset(self):
        self.values = []

    def save_to_disk(self, parent_dir, run_num):
        np.save(os.path.join(parent_dir, self.f_name + '_' + str(run_num)), np.array(self.values))

    def load_from_disk(self, parent_dir, run_num):
        self.values = list(np.load(os.path.join(parent_dir, self.f_name + '_' + str(run_num) + '.npy')))

class CosineStats():
    def __init__(self, layer, population_size, basic_mode=False):
        self.num_neurons = layer.out_features
        self.cosine_dists = pairwise_cos_dist(layer)
        self.layer = layer # Pointer to layer of interest, no need to pass new weights every time step
        self.iu2 = np.triu_indices(self.num_neurons, k=1)

        self.old_w = np.array(torch.cat((self.layer.weight.data, self.layer.bias.data.unsqueeze(1)), dim=1).clone().cpu())
        
        self.basic_mode = basic_mode
        if not self.basic_mode:
            self.low_overlap_pair_indices, self.high_overlap_pair_indices = sorted_subset_weight_pairs(self.cosine_dists[self.iu2], self.iu2, num_pairs=population_size)
            self.low_overlap_x = self.iu2[0][self.low_overlap_pair_indices]
            self.low_overlap_y = self.iu2[1][self.low_overlap_pair_indices]
            self.high_overlap_x = self.iu2[0][self.high_overlap_pair_indices]
            self.high_overlap_y = self.iu2[1][self.high_overlap_pair_indices]
            # self.old_mean_w = self._compute_mean_w_population(w)

    def initial_stats(self):
        hist = self._compute_hist()
        if not self.basic_mode:
            cd = self._cos_dist_subpopulation()
        else:
            cd = []

        return hist, cd

    def _compute_mean_w_population(self, w):
        low_overlap_means = 0.5*(w[self.low_overlap_x] + w[self.low_overlap_y])
        high_overlap_means = 0.5*(w[self.high_overlap_x] + w[self.high_overlap_y])
        return np.concatenate((low_overlap_means, high_overlap_means))

    def _compute_hist(self):
        return np.histogram(self.cosine_dists[self.iu2], bins=50, range=(-1,1))

    def _cos_dist_subpopulation(self):
        cd_low = self.cosine_dists[self.iu2][self.low_overlap_pair_indices]
        cd_high = self.cosine_dists[self.iu2][self.high_overlap_pair_indices]
        cd = np.concatenate((cd_low, cd_high))
        return cd

    def compute_stats(self):
        self.cosine_dists = pairwise_cos_dist(self.layer)

        hist = self._compute_hist()
        if not self.basic_mode:
            cd = self._cos_dist_subpopulation()

            w = np.array(torch.cat((self.layer.weight.data, self.layer.bias.data.unsqueeze(1)), dim=1).clone().cpu())
            mean_w = self._compute_mean_w_population(w)
            old_mean_w = self._compute_mean_w_population(self.old_w)
            cm = cosine_similarity(old_mean_w, mean_w).diagonal()
            # print(np.mean((mean_w-old_mean_w)**2))
            self.old_w = w
        else:
            cd = []
            cm = []

        return hist, cd, cm
