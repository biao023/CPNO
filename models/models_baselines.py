from torch.nn import functional as F
import torch
import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import numpy as np
import time

torch.set_default_dtype(torch.float)
torch.manual_seed(123456)
np.random.seed(123456)



def truncated_normal_(tensor,mean=0,std=0.09):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size+(4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor

class FCN(nn.Module):
    ##Neural Network
    def __init__(self, layers):
        super().__init__()  # call __init__ from parent
        self.layers = layers
        self.activation = nn.GELU()
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        self.iter = 0
        for i in range(len(layers) - 1):
            self.linears[i].weight = truncated_normal_(self.linears[i].weight, mean=0, std=0.1)
            nn.init.zeros_(self.linears[i].bias.data)

    def forward(self, x):
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x)
        a = x.float()
        for i in range(len(self.layers) - 2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a


class HyperPINN(torch.nn.Module):
    '''
    this class use full rank hyper net for pinns, here we have single hyper network
    '''
    def __init__(self, para_layer, Ns):
        super(HyperPINN, self).__init__()
        self.para_net = FCN(para_layer)
        self.Ns = Ns
        self.activation =  nn.GELU()

    def forward(self, params, x):
        #we use 8 layer in pinns, input dimenssion 2, hidden layer 6, node Ns, output dimenssion 1
        x = torch.unsqueeze(x, 1)
        latent = self.para_net(params)

        Ns = self.Ns
        weight_1 = latent[:, 0 : 2*Ns].reshape(-1, 2, Ns)
        weight_2 = latent[:, Ns * 2 : Ns * 2 + Ns * Ns].reshape(-1, Ns, Ns)
        weight_3 = latent[:, Ns * 2 + Ns * Ns : Ns * 2 + Ns * Ns * 2].reshape(-1, Ns, Ns)
        weight_4 = latent[:, Ns * 2 + Ns * Ns * 2 : Ns * 2 + Ns * Ns * 3].reshape(-1, Ns, Ns)
        weight_5 = latent[:, Ns * 2 + Ns * Ns * 3 : Ns * 2 + Ns * Ns * 4].reshape(-1, Ns, Ns)
        weight_6 = latent[:, Ns * 2 + Ns * Ns * 4 : Ns * 3 + Ns * Ns * 4].reshape(-1, Ns, 1)

        bias1 = torch.unsqueeze(latent[:, Ns * 3 + Ns * Ns * 4: Ns * 4 + Ns * Ns * 4], 1)
        bias2 = torch.unsqueeze(latent[:, Ns * 4 + Ns * Ns * 4: Ns * 5 + Ns * Ns * 4], 1)
        bias3 = torch.unsqueeze(latent[:, Ns * 5 + Ns * Ns * 4: Ns * 6 + Ns * Ns * 4], 1)
        bias4 = torch.unsqueeze(latent[:, Ns * 6 + Ns * Ns * 4: Ns * 7 + Ns * Ns * 4], 1)
        bias5 = torch.unsqueeze(latent[:, Ns * 7 + Ns * Ns * 4: Ns * 8 + Ns * Ns * 4], 1)

        out = self.activation(torch.matmul(x, weight_1) + bias1)
        out = self.activation(torch.matmul(out, weight_2) + bias2)
        out = self.activation(torch.matmul(out, weight_3) + bias3)
        out = self.activation(torch.matmul(out, weight_4) + bias4)
        out = self.activation(torch.matmul(out, weight_5) + bias5)
        out = torch.matmul(out, weight_6)

        out = torch.squeeze(out, 2)

        return out


class PI_Deeponet(torch.nn.Module):
    '''
    The PI-deeponet module as baseline
    '''
    def __init__(self, para_layer, truck_layer):
        super(PI_Deeponet, self).__init__()
        self.para_net = FCN(para_layer)
        self.truck_net = FCN(truck_layer)

    def forward(self, params, x):
        B = self.para_net(params)
        T = self.truck_net(x)
        out =  torch.unsqueeze( torch.sum(B * T, dim=1), 1)
        return out

class MAD(torch.nn.Module):
    '''
    The MAD module as baseline
    '''
    def __init__(self, para_layer, pinns_layer ):
        super(MAD, self).__init__()
        self.latent_net = FCN(para_layer)
        self.pinns_net = FCN(pinns_layer)

    def forward(self, params, x):
        latent = self.latent_net(params)
        input = torch.cat((x, latent), dim=1)
        out = self.pinns_net(input)
        return out

