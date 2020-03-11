import numpy as np

def msd_bin(cos_dists, delta, population_size=100):
    num_neurons = cos_dists.shape[1]
    iu2 = np.triu_indices(num_neurons, k=1)

    x = cos_dists[0][iu2]
    low_overlap_pair_indices, high_overlap_pair_indices = sorted_subset_weight_pairs(x, iu2, num_pairs=population_size)

    y_low = []
    y_high = []

    for i in range(len(cos_dists)-delta):
        temp_1_low = cos_dists[i][iu2][low_overlap_pair_indices]
        temp_2_low = cos_dists[i+delta][iu2][low_overlap_pair_indices]
        
        temp_1_high = cos_dists[i][iu2][high_overlap_pair_indices]
        temp_2_high = cos_dists[i+delta][iu2][high_overlap_pair_indices]

        y_low.append(np.square(temp_2_low-temp_1_low))
        y_high.append(np.square(temp_2_high-temp_1_high))
        
    y_low = np.array(y_low)
    y_high = np.array(y_high)

    msd_low = np.mean(y_low)
    std_msd_low = np.std(y_low)
    msd_high = np.mean(y_high)
    std_msd_high = np.std(y_high)
    
    return msd_low, std_msd_low, msd_high, std_msd_high


def sorted_subset_weight_pairs(x, iu2, num_pairs=100):
    sort_indices = np.argsort(np.abs(x))
    pairs = np.array(iu2)[:, sort_indices]

    high_overlap_pair_indices = []
    low_overlap_pair_indices = []

    i = 0
    j = 0

    list_neuron_idx = []

    while i < num_pairs:
        if pairs[0][-1-j] not in list_neuron_idx and pairs[1][-1-j] not in list_neuron_idx:
            high_overlap_pair_indices.append(sort_indices[-1-j])
            list_neuron_idx.append(pairs[0][-1-j])
            list_neuron_idx.append(pairs[1][-1-j])
            i += 1
        j += 1

    i = 0
    j = 0
    while i < num_pairs:
        if pairs[0][j] not in list_neuron_idx and pairs[1][j] not in list_neuron_idx:
            low_overlap_pair_indices.append(sort_indices[j])
            list_neuron_idx.append(pairs[0][j])
            list_neuron_idx.append(pairs[1][j])
            i += 1
        j += 1

    return low_overlap_pair_indices, high_overlap_pair_indices


def numpy_to_mpl_hist(np_hist_output):
    hist, bins = np_hist_output
    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    return hist, center, width
