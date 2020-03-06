import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

def gradient_projection_norms(layer):
    grad = layer.weight.grad.clone().cpu()
    
    num_neurons = grad.shape[0]
    iu2 = np.triu_indices(num_neurons, k=1)
    
    diff_grad = torch.norm(grad[iu2[0]] - grad[iu2[1]], dim=1).numpy()
    mean_grad = torch.norm(0.5*(grad[iu2[0]] + grad[iu2[1]]), dim=1).numpy()
    
    return mean_grad, diff_grad

def pairwise_cos_dist(net):
    w = net.dense_2.weight.data.cpu()
    b = net.dense_2.bias.data.cpu()
    w1 = torch.cat((w,b.unsqueeze(1)), dim=1)

    cd = cosine_similarity(w1,w1)
    
    return cd