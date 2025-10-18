import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils import spectral_norm as sn
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint
from functools import partial
from torch import optim
import math
import numbers
import functools
from scipy import ndimage
from torch.nn import Parameter

from utils import *
from torch.quasirandom import SobolEngine


#################################################################################
# Credit for Swish                                                              #
#                                                                               #
# Projet: https://github.com/rtqichen/residual-flows                            #
# Copyright (c) 2019 Ricky Tian Qi Chen                                         #
# Licence (MIT): https://github.com/rtqichen/residual-flows/blob/master/LICENSE #
#################################################################################

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))
    def forward(self, x):
        return (x * torch.sigmoid_(x * F.softplus(self.beta))).div_(1.1)

class Sinus(nn.Module):
    def forward(self, input):
        return torch.sinus(input)
    
class LinearEstimator(nn.Module):
    def __init__(self, in_c, out_c, factor=1.0):
        super().__init__()
        self.factor = factor
        self.net = nn.Linear(in_c, out_c, bias=False)

    def forward(self, x):
        return self.net(x) * self.factor

class MLPEstimator(nn.Module):
    def __init__(self, in_c, out_c, hidden, factor):
        super().__init__()
        bias = True
        self.net = nn.Sequential(
            nn.Linear(in_c, hidden, bias=bias),
            Swish(),
            nn.Linear(hidden, hidden, bias=bias),
            Swish(),
            nn.Linear(hidden, hidden, bias=bias),
            Swish(),
            nn.Linear(hidden, out_c, bias=bias),
        )
        self.factor = factor
    
    def forward(self, x):
        flatten = x.dim() == 3
        if flatten:
            batch_size, nc, T = x.shape
            x = x.transpose(2, 1)
        x = self.net(x) * self.factor
        if flatten:
            batch_size, nc, T = x.shape
            x = x.transpose(2, 1)
        return x

#####################################################################################
# Credit for SpectralConv2d_fast, FNO2d                                             #
#                                                                                   #
# Projet: https://github.com/zongyi-li/fourier_neural_operator                      #
# Copyright (c) 2020 Zongyi Li                                                      #
# Licence: https://github.com/zongyi-li/fourier_neural_operator/blob/master/LICENSE #
#####################################################################################
class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        # self.scale = (1 / (in_channels * out_channels))
        self.scale = 1 / math.sqrt(in_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(3, self.width)
        # input channel is 3: (w(t, x, y),  x, y)
        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.bn0, self.bn1, self.bn2, self.bn3 = [nn.Identity() for _ in range(4)]
        self.a0 = Swish()
        self.a1 = Swish()
        self.a2 = Swish()
        self.a3 = Swish()

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x - (batchsize, nc, h, w)
        batchsize = x.shape[0]
        size_x, size_y = x.shape[2], x.shape[3] 
        x = x.permute(0, 2, 3, 1)
        # x - (batchsize, h, w, nc)
        grid = self.get_grid(batchsize, size_x, size_y, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        # x - (batchsize, width, h, w)
        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn0(x1 + x2)
        x = self.a0(x)
        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn1(x1 + x2)
        x = self.a1(x)
        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn2(x1 + x2)
        x = self.a2(x)
        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn3(x1 + x2)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.a3(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x

    def get_grid(self, batchsize, size_x, size_y, device):
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

class ConvNetEstimator(nn.Module):
    def __init__(self, in_c, out_c, hidden, factor, net_type):
        super().__init__()

        kernel_size = 3
        padding = kernel_size // 2

        self.out_c = out_c
        self.factor = factor

        if net_type == 'conv':
            bias = True
            self.res = nn.Sequential(
                nn.Conv2d(in_c, hidden, kernel_size=kernel_size, padding=padding, padding_mode='circular', bias=bias),
                Swish(),
                nn.Conv2d(hidden, hidden, kernel_size=kernel_size, padding=padding, padding_mode='circular', bias=bias),
                Swish(),
                nn.Conv2d(hidden, hidden, kernel_size=kernel_size, padding=padding, padding_mode='circular', bias=bias),
                Swish(),
                nn.Conv2d(hidden, out_c, kernel_size=kernel_size, padding=padding, padding_mode='circular'),
            )
        elif net_type == 'fno':
            self.res = FNO2d(6, 6, 10)
        else:
            raise NotImplementedError

    def forward(self, x):
        flatten = x.dim() == 5
        if flatten:
            batch_size, nc, T, h, w = x.shape
            x = x.transpose(2, 1)
            x = x.reshape(batch_size * T, nc, h, w)

        x = self.res(x) * self.factor

        if flatten:
            x = x.reshape(batch_size, T, self.out_c, h, w)
            x = x.transpose(2, 1).contiguous()
        return x

#########################################################################
#########################################################################
#########################################################################
#########################################################################

nls = {'relu': partial(nn.ReLU),
       'sigmoid': partial(nn.Sigmoid),
       'tanh': partial(nn.Tanh),
       'selu': partial(nn.SELU),
       'softplus': partial(nn.Softplus),
       'swish': partial(Swish),
       'sinus': partial(Sinus),
       'elu': partial(nn.ELU)}


class GroupSwish(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.5 for _ in range(groups)]))
        self.groups = groups

    def forward(self, x):
        n_ch_group = x.size(1) // self.groups
        t = x.shape[2:]
        x = x.reshape(-1, self.groups, n_ch_group, *t)
        beta = self.beta.view(1, self.groups, 1, *[1 for _ in t])
        return (x * torch.sigmoid_(x * F.softplus(beta))).div_(1.1).reshape(-1, self.groups * n_ch_group, *t)


class GroupActivation(nn.Module):
    def __init__(self, nl, groups=1):
        super().__init__()
        self.groups = groups
        if nl == 'swish':
            self.activation = GroupSwish(groups)
        else:
            self.activation = nls[nl]()

    def forward(self, x):
        return self.activation(x)


def generate_mask(net_a, mask_type="layer", layers=[0]):
    n_params_tot = count_parameters(net_a)
    if mask_type == "layer":
        mask_w = torch.zeros(n_params_tot)
        count = 0
        for name, pa in net_a.named_parameters():
            if any(f"net.{layer}" in name for layer in layers):
                mask_w[count: count + pa.numel()] = 1.
            count += pa.numel()
    elif mask_type == "full":
        mask_w = torch.ones(n_params_tot)
    else:
        raise Exception(f"Unknown mask {mask_type}")
    return mask_w

import copy 

class HyperEnvNet(nn.Module):
    def __init__(self, net_a, ghost_structure, hypernet, codes, logger, net_mask=None, device="cuda:0"):
        super().__init__()
        self.net_a = net_a
        self.codes = codes
        self.n_env = codes.size(0)
        self.hypernet = hypernet
        self.nets = {'ghost_structure': ghost_structure, "mask": net_mask}  # , "ghost": ghost_structure}
        self.logger = logger
        self.device = device

    def update_ghost(self):
        net_ghost = copy.deepcopy(self.nets['ghost_structure'])
        set_requires_grad(net_ghost, False)
        self.nets["ghost"] = net_ghost
        param_hyper = self.hypernet(self.codes)
        count_f = 0
        count_p = 0
        param_mask = self.nets["mask"]
        for pa, pg in zip(self.net_a.parameters(), self.nets["ghost"].parameters()):
            phypers = []
            if param_mask is None:
                pmask_sum = int(pa.numel())
            else:
                pmask = param_mask[count_f: count_f + pa.numel()].reshape(*pa.shape)
                pmask_sum = int(pmask.sum())
            if pmask_sum == int(pa.numel()):
                for e in range(self.n_env):
                    phypers.append(param_hyper[e, count_p: count_p + pmask_sum].reshape(*pa.shape))
            else:
                for e in range(self.n_env):
                    phyper = torch.zeros(*pa.shape).to(self.device)
                    if pmask_sum != 0:
                        phyper[pmask == 1] = param_hyper[count_p:count_p + pmask_sum]
                    phypers.append(phyper)
            count_p += pmask_sum
            count_f += int(pa.numel())

            phyper = torch.cat(phypers, dim=0)
            pa_new = torch.cat([pa] * self.n_env, dim=0)
            pg.copy_(pa_new + phyper)

    def forward(self, *input, **kwargs):
        return self.nets["ghost"](*input, **kwargs)

class GroupConvMLP(nn.Module):
    def __init__(self, state_c, hidden_c=64, groups=1, factor=1.0, nl="swish"):
        super().__init__()
        self.factor = factor
        self.net = nn.Sequential(
            nn.Conv1d(state_c * groups, hidden_c * groups, 1, groups=groups),
            GroupActivation(nl, groups=groups),
            nn.Conv1d(hidden_c * groups, hidden_c * groups, 1, groups=groups),
            GroupActivation(nl, groups=groups),
            nn.Conv1d(hidden_c * groups, hidden_c * groups, 1, groups=groups),
            GroupActivation(nl, groups=groups),
            nn.Conv1d(hidden_c * groups, state_c * groups, 1, groups=groups),
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        return self.net(x).squeeze(-1) * self.factor


class GroupConv(nn.Module):
    def __init__(self, state_c, hidden_c=64, groups=1, factor=1.0, nl="swish", size=64, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.out_c = state_c
        self.factor = factor
        self.hidden_c = hidden_c
        self.size = size
        self.net = nn.Sequential(
            nn.Conv2d(state_c * groups, hidden_c * groups, kernel_size=kernel_size, padding=padding, padding_mode='circular', groups=groups),
            GroupActivation(nl, groups=groups),
            nn.Conv2d(hidden_c * groups, hidden_c * groups, kernel_size=kernel_size, padding=padding, padding_mode='circular', groups=groups),
            GroupActivation(nl, groups=groups),
            nn.Conv2d(hidden_c * groups, hidden_c * groups, kernel_size=kernel_size, padding=padding, padding_mode='circular', groups=groups),
            GroupActivation(nl, groups=groups),
            nn.Conv2d(hidden_c * groups, state_c * groups, kernel_size=kernel_size, padding=padding, padding_mode='circular', groups=groups)
        )

    def forward(self, x):
        x = self.net(x) 
        # print(x.abs().mean())
        x = x * self.factor
        return x


class GroupSpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, groups=1):
        super().__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        self.groups = groups
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(groups * in_channels, out_channels, self.modes1, self.modes2, 2))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(groups * in_channels, out_channels, self.modes1, self.modes2, 2))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, env, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, env, out_channel, x,y)
        return torch.einsum("beixy,eioxy->beoxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)
        x_ft = x_ft.reshape(batchsize, self.groups, self.in_channels, x.size(-2), x.size(-1) // 2 + 1)
        # Multiply relevant Fourier modes
        weights1 = self.weights1.reshape(self.groups, self.in_channels, self.out_channels, self.modes1, self.modes2, 2)
        weights2 = self.weights2.reshape(self.groups, self.in_channels, self.out_channels, self.modes1, self.modes2, 2)
        out_ft = torch.zeros(batchsize, self.groups, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :, :self.modes1, :self.modes2], torch.view_as_complex(weights1))
        out_ft[:, :, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :, -self.modes1:, :self.modes2], torch.view_as_complex(weights2))
        # Return to physical space
        out_ft = out_ft.reshape(batchsize, self.groups * self.out_channels, x.size(-2), x.size(-1) // 2 + 1)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class GroupFNO2d(nn.Module):
    def __init__(self, state_c, modes1=12, modes2=12, width=10, groups=1, nl='swish'):
        super().__init__()
        self.width = width
        self.groups = groups
        self.fc0 = nn.Conv2d((state_c + 2) * self.groups, self.width * self.groups, 1, groups=groups)
        self.conv0 = GroupSpectralConv2d(self.width, self.width, modes1, modes2, groups)
        self.conv1 = GroupSpectralConv2d(self.width, self.width, modes1, modes2, groups)
        self.conv2 = GroupSpectralConv2d(self.width, self.width, modes1, modes2, groups)
        self.conv3 = GroupSpectralConv2d(self.width, self.width, modes1, modes2, groups)
        self.w0 = nn.Conv2d(self.width * self.groups, self.width * self.groups, 1, groups=groups)
        self.w1 = nn.Conv2d(self.width * self.groups, self.width * self.groups, 1, groups=groups)
        self.w2 = nn.Conv2d(self.width * self.groups, self.width * self.groups, 1, groups=groups)
        self.w3 = nn.Conv2d(self.width * self.groups, self.width * self.groups, 1, groups=groups)
        self.a0 = GroupActivation(nl, groups=groups)
        self.a1 = GroupActivation(nl, groups=groups)
        self.a2 = GroupActivation(nl, groups=groups)
        self.a3 = GroupActivation(nl, groups=groups)
        self.fc1 = nn.Conv2d(self.width * self.groups, 128 * self.groups, 1, groups=groups)
        self.fc2 = nn.Conv2d(128 * self.groups, state_c * self.groups, 1, groups=groups)

    def forward(self, x):
        # x:  batchsize x n_env * c x h x w
        minibatch_size = x.shape[0]
        x = batch_transform_inverse(x, self.groups)
        batchsize = x.shape[0]
        size_x, size_y = x.shape[-2], x.shape[-1]
        grid = self.get_grid(batchsize, size_x, size_y, x.device)
        x = torch.cat((x, grid), dim=1)
        x = batch_transform(x, minibatch_size)

        # Lift with P
        x = self.fc0(x)
        # Fourier layer 0
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = self.a0(x1 + x2)
        # Fourier layer 1
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = self.a1(x1 + x2)
        # Fourier layer 2
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = self.a2(x1 + x2)
        # Fourier layer 3
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # Projection with Q
        x = self.a3(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_grid(self, batchsize, size_x, size_y, device):
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)
