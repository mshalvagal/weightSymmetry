import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from collections import Counter

def widen_v1(m_in, m_out, symmetry_break_method):
    
    w_in = m_in.weight.data
    b_in = m_in.bias.data

    w_out = m_out.weight.data
    b_out = m_out.bias.data

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
        nw_out[:, :old_width] *= 2.0
        nw_out[:, old_width:] *= -1.0
    if symmetry_break_method == 'convex_comb':
        weights = torch.rand(old_width).to(w_in.device)
        nw_out[:, :old_width] *= weights
        nw_out[:, old_width:] *= 1.0-weights
    elif symmetry_break_method == 'noise':
        nw_out /= 2.0
        nw_out[:, old_width:] += np.sqrt(2.0/(2*old_width))*torch.randn_like(w_out)
        nw_in[old_width:] += np.sqrt(2.0/w_in.shape[1])*torch.randn_like(w_in)
    elif symmetry_break_method == 'noise_unscaled':
        nw_out /= 2.0
        nw_out[:, old_width:] += torch.randn_like(w_out)
        nw_in[old_width:] += torch.randn_like(w_in)
    elif symmetry_break_method == 'random_init':
        nw_out[:, old_width:] = np.sqrt(2.0/(2*old_width))*torch.randn_like(w_out)
        nw_in[old_width:] = np.sqrt(2.0/w_in.shape[1])*torch.randn_like(w_in)

    nm_in = nn.Linear(w_in.shape[1], 2*old_width)
    nm_out = nn.Linear(2*old_width, w_out.shape[0])

    nm_in.weight.data = nw_in
    nm_in.bias.data = nb_in
    nm_out.weight.data = nw_out
    nm_out.bias.data = b_out

    return nm_in, nm_out


def widen_v2(m1, m2, new_width, bnorm=None, out_size=None, noise=True,
            random_init=False, weight_norm=False):
    """
    Convert m1 layer to its wider version by adapthing next weight layer and
    possible batch norm layer in btw.
    Args:
        m1 - module to be wider
        m2 - follwing module to be adapted to m1
        new_width - new width for m1.
        bn (optional) - batch norm layer, if there is btw m1 and m2
        out_size (list, optional) - necessary for m1 == conv3d and m2 == linear. It
            is 3rd dim size of the output feature map of m1. Used to compute
            the matching Linear layer size
        noise (bool, True) - add a slight noise to break symmetry btw weights.
        random_init (optional, True) - if True, new weights are initialized
            randomly.
        weight_norm (optional, True) - If True, weights are normalized before
            transfering.
    """

    w1 = m1.weight.data
    w2 = m2.weight.data
    b1 = m1.bias.data

    if "Conv" in m1.__class__.__name__ or "Linear" in m1.__class__.__name__:
        # Convert Linear layers to Conv if linear layer follows target layer
        if "Conv" in m1.__class__.__name__ and "Linear" in m2.__class__.__name__:
            assert w2.size(1) % w1.size(0) == 0, "Linear units need to be multiple"
            if w1.dim() == 4:
                factor = int(np.sqrt(w2.size(1) // w1.size(0)))
                w2 = w2.view(w2.size(0), w2.size(1)//factor**2, factor, factor)
            elif w1.dim() == 5:
                assert out_size is not None,\
                       "For conv3d -> linear out_size is necessary"
                factor = out_size[0] * out_size[1] * out_size[2]
                w2 = w2.view(w2.size(0), w2.size(1)//factor, out_size[0],
                             out_size[1], out_size[2])
        else:
            assert w1.size(0) == w2.size(1), "Module weights are not compatible"
        assert new_width > w1.size(0), "New size should be larger"

        old_width = w1.size(0)
        nw1 = m1.weight.data.clone()
        nw2 = w2.clone()

        if nw1.dim() == 4:
            nw1.resize_(new_width, nw1.size(1), nw1.size(2), nw1.size(3))
            nw2.resize_(nw2.size(0), new_width, nw2.size(2), nw2.size(3))
        elif nw1.dim() == 5:
            nw1.resize_(new_width, nw1.size(1), nw1.size(2), nw1.size(3), nw1.size(4))
            nw2.resize_(nw2.size(0), new_width, nw2.size(2), nw2.size(3), nw2.size(4))
        else:
            nw1.resize_(new_width, nw1.size(1))
            nw2.resize_(nw2.size(0), new_width)

        if b1 is not None:
            nb1 = m1.bias.data.clone()
            nb1.resize_(new_width)

        if bnorm is not None:
            nrunning_mean = bnorm.running_mean.clone().resize_(new_width)
            nrunning_var = bnorm.running_var.clone().resize_(new_width)
            if bnorm.affine:
                nweight = bnorm.weight.data.clone().resize_(new_width)
                nbias = bnorm.bias.data.clone().resize_(new_width)

        w2 = w2.transpose(0, 1)
        nw2 = nw2.transpose(0, 1)

        nw1.narrow(0, 0, old_width).copy_(w1)
        nw2.narrow(0, 0, old_width).copy_(w2)
        nb1.narrow(0, 0, old_width).copy_(b1)

        if bnorm is not None:
            nrunning_var.narrow(0, 0, old_width).copy_(bnorm.running_var)
            nrunning_mean.narrow(0, 0, old_width).copy_(bnorm.running_mean)
            if bnorm.affine:
                nweight.narrow(0, 0, old_width).copy_(bnorm.weight.data)
                nbias.narrow(0, 0, old_width).copy_(bnorm.bias.data)

        # TEST:normalize weights
        if weight_norm:
            for i in range(old_width):
                norm = w1.select(0, i).norm()
                w1.select(0, i).div_(norm)

        # select weights randomly
        tracking = dict()
        for i in range(old_width, new_width):
            idx = np.random.randint(0, old_width)
            try:
                tracking[idx].append(i)
            except:
                tracking[idx] = [idx]
                tracking[idx].append(i)

            # TEST:random init for new units
            if random_init:
                n = m1.kernel_size[0] * m1.kernel_size[1] * m1.out_channels
                if m2.weight.dim() == 4:
                    n2 = m2.kernel_size[0] * m2.kernel_size[1] * m2.out_channels
                elif m2.weight.dim() == 5:
                    n2 = m2.kernel_size[0] * m2.kernel_size[1] * m2.kernel_size[2] * m2.out_channels
                elif m2.weight.dim() == 2:
                    n2 = m2.out_features * m2.in_features
                nw1.select(0, i).normal_(0, np.sqrt(2./n))
                nw2.select(0, i).normal_(0, np.sqrt(2./n2))
            else:
                nw1.select(0, i).copy_(w1.select(0, idx).clone())
                nw2.select(0, i).copy_(w2.select(0, idx).clone())
            nb1[i] = b1[idx]

        if bnorm is not None:
            nrunning_mean[i] = bnorm.running_mean[idx]
            nrunning_var[i] = bnorm.running_var[idx]
            if bnorm.affine:
                nweight[i] = bnorm.weight.data[idx]
                nbias[i] = bnorm.bias.data[idx]
            bnorm.num_features = new_width

        if not random_init:
            for idx, d in tracking.items():
                for item in d:
                    nw2[item].div_(len(d))

        w2.transpose_(0, 1)
        nw2.transpose_(0, 1)

        m1.out_channels = new_width
        m2.in_channels = new_width
        m1.out_features = new_width
        m2.in_features = new_width

        if noise:
            noise = np.random.normal(scale=5e-2 * nw1.cpu().std(),
                                     size=list(nw1.cpu().size()))
            nw1 += torch.FloatTensor(noise).type_as(nw1)

        m1.weight.data = nw1

        if "Conv" in m1.__class__.__name__ and "Linear" in m2.__class__.__name__:
            if w1.dim() == 4:
                m2.weight.data = nw2.view(m2.weight.size(0), new_width*factor**2)
                m2.in_features = new_width*factor**2
            elif w2.dim() == 5:
                m2.weight.data = nw2.view(m2.weight.size(0), new_width*factor)
                m2.in_features = new_width*factor
        else:
            m2.weight.data = nw2

        m1.bias.data = nb1

        if bnorm is not None:
            bnorm.running_var = nrunning_var
            bnorm.running_mean = nrunning_mean
            if bnorm.affine:
                bnorm.weight.data = nweight
                bnorm.bias.data = nbias
        return m1, m2, bnorm
