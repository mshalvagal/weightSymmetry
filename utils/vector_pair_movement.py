import torch
import numpy as np
import os
from metrics.metrics import CosineStats

def calculate_epochwise_changes(parent_dir, run_num, time_points):
    w = torch.load(os.path.join(parent_dir, 'weight_history/run_' + str(run_num), 'epoch_' + str(time_points[0]) + '.pt'))
    w_ref = torch.load(os.path.join(parent_dir, 'weight_history/run_' + str(run_num), 'epoch_800.pt'))
    cosine_stats = CosineStats(w.dense_2, 100, reference_layer=w_ref.dense_2)

    cd_long = []
    cm_long = []

    _, cd, cm = cosine_stats.initial_stats()
    cd_long.append(cd)
    cm_long.append(cm)

    for t in time_points[1:]:
        w_next = torch.load(os.path.join(parent_dir, 'weight_history/run_' + str(run_num), 'epoch_' + str(t) + '.pt'))
        w.dense_2.weight.data = w_next.dense_2.weight.data
        w.dense_2.bias.data = w_next.dense_2.bias.data
        _, cd, cm = cosine_stats.compute_stats()
        cd_long.append(cd)
        cm_long.append(cm)

    cd_long = np.clip(np.array(cd_long), -1, 1)
    cm_long = np.clip(np.array(cm_long), -1, 1)
    sd_long = np.sqrt(1-cd_long**2)
    sm_long = np.sqrt(1-cm_long**2)

    return cd_long, cm_long, sd_long, sm_long
