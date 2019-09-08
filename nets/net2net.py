import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def widen_v1(m_in, m_out, symmetry_break_method):
    
    w_in = m_in.weight.data
    b_in = m_in.bias.data

    w_out = m_out.weight.data

    old_width = w_in.shape[0]
    assert old_width == w_out.shape[1], "Module weights are not compatible"

    nw_in = w_in.repeat(2, 1)
    nb_in = b_in.repeat(2)
    nw_out = w_out.repeat(1, 2)

    permutation_array = torch.randperm(old_width)

    nw_in[old_width:] = w_in[permutation_array]
    nb_in[old_width:] = b_in[permutation_array]
    nw_out[:, old_width:] = w_out[:, permutation_array]

    if symmetry_break_method == 'simple':
        nw_out[:old_width] *= 2.0
        nw_out[old_width:] *= -1.0
    elif symmetry_break_method == 'noise':
        nw_out /= 2.0
        nw_out[:, old_width:] += np.sqrt(2.0/(2*old_width))*torch.randn_like(w_out)
        nw_in[old_width:] += np.sqrt(2.0/w_in.shape[1])*torch.randn_like(w_in)
    elif symmetry_break_method == 'random_init':
        nw_out[:, old_width:] = np.sqrt(2.0/(2*old_width))*torch.randn_like(w_out)
        nw_in[old_width:] = np.sqrt(2.0/w_in.shape[1])*torch.randn_like(w_in)

    m_in.weight.data = nw_in
    m_in.bias.data = nb_in
    m_out.weight.data = nw_out

    return m_in, m_out
