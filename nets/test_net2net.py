import pytest
from net2net import widen_v1

import torch
import torch.nn as nn
import torch.nn.functional as F

def test_widen_v1_shapes():

    demo_m1 = nn.Linear(10, 4)
    demo_m2 = nn.Linear(4, 3)
    
    out_m1, out_m2 = widen_v1(demo_m1, demo_m2, 'simple')

    assert list(out_m1.weight.data.shape) == [8, 10], "Input weight transformed shape incorrect"
    assert list(out_m2.weight.data.shape) == [3, 8], "Output weight transformed shape incorrect"

def test_widen_v1_function_preserve():

    demo_m1 = nn.Linear(10, 4)
    demo_m2 = nn.Linear(4, 3)
    out_m1, out_m2 = widen_v1(demo_m1, demo_m2, 'simple')
    out_m1_2, out_m2_2 = widen_v1(demo_m1, demo_m2, 'convex_comb')

    old_net = nn.Sequential(
        demo_m1,
        nn.ReLU(),
        demo_m2,
        nn.Softmax(dim=0)
    )

    new_net = nn.Sequential(
        out_m1,
        nn.ReLU(),
        out_m2,
        nn.Softmax(dim=0)
    )

    new_net_2 = nn.Sequential(
        out_m1_2,
        nn.ReLU(),
        out_m2_2,
        nn.Softmax(dim=0)
    )

    demo_input = torch.ones(10)

    assert torch.equal(old_net(demo_input), new_net(demo_input)), "Function not preserved under basic widening"
    assert torch.equal(old_net(demo_input), new_net_2(demo_input)), "Function not preserved under widening with convex combination of output weights. Error: {}".format(torch.norm(old_net(demo_input) - new_net_2(demo_input)))