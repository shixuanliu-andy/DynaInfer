# MIT License
#
# Copyright (c) 2022 Matthieu Kirchmeyer & Yuan Yin

import numpy as np
from torchdiffeq import odeint
from networks import *
from utils import *

class Derivative(nn.Module):
    def __init__(self, state_c, hidden_c, code_c, n_env, factor, nl, dataset, is_ode, size=64, is_layer=False,
                 layers=[0], logger=None, mask=None, codes_init=None, device="cuda", htype="hyper", **kwargs):
        super().__init__()
        self.is_ode = is_ode
        self.size = size
        self.is_layer = is_layer
        self.logger = logger
        self.codes = nn.Parameter(0. * torch.ones(n_env, code_c)) if codes_init is None else codes_init
        # Bias
        if self.is_ode:
            self.net_root = GroupConvMLP(state_c, hidden_c, groups=1, factor=factor, nl=nl)
        elif dataset == "gs" or dataset == "wave":
            self.net_root = GroupConv(state_c, hidden_c, groups=1, factor=factor, nl=nl, size=size)
        elif dataset == "ns":
            self.net_root = GroupFNO2d(state_c, nl=nl, groups=1)
        n_param_tot = count_parameters(self.net_root)
        n_param_mask = n_param_tot if not is_layer else get_n_param_layer(self.net_root, layers)
        n_param_hypernet = n_param_mask
        if logger:
            self.logger.info(f"Params: n_mask {n_param_mask} / n_tot {n_param_tot} / n_hypernet {n_param_hypernet}")
        # Hypernet
        self.net_hyper = nn.Linear(code_c, n_param_hypernet, bias=False)

        # Ghost
        if self.is_ode:
            self.ghost_structure = GroupConvMLP(state_c, hidden_c, groups=n_env, factor=factor, nl=nl)
        elif dataset == "gs" or dataset == "wave":
            self.ghost_structure = GroupConv(state_c, hidden_c, groups=n_env, factor=factor, nl=nl, size=size)
        elif dataset == "ns":
            self.ghost_structure = GroupFNO2d(state_c, nl=nl, groups=n_env)
        else:
            raise Exception(f"{dataset} net not implemented")
        set_requires_grad(self.ghost_structure, False)
        # Mask
        if is_layer and mask is None:
            self.mask = {"mask": generate_mask(self.net_root, "layer", layers)}
        else:
            self.mask = {"mask": mask}
        # Total
        self.net_leaf = HyperEnvNet(self.net_root, self.ghost_structure, 
                                    self.net_hyper, self.codes, logger, self.mask["mask"], device)
        
    def update_ghost(self):
        self.net_leaf.update_ghost()

    def forward(self, t, u):
        return self.net_leaf(u)


class Forecaster(nn.Module):
    def __init__(self, args, n_env, options=None, nl="swish", mask=None, htype='hyper', **kwargs):
        super().__init__()
        state_c = args.var_dim
        hidden_c = args.hidden
        code_c = args.dim_context
        factor = args.factor
        device = args.device
        logger = args.logger
        codes_init = args.codes_init
        self.dataset = args.dataset
        is_ode = self.dataset in ["lv", "go"]
        size = 0 if is_ode else 32
        layers = [-1]
        is_layer = (layers[0] != -1)

        self.method = args.method
        self.options = options
        self.is_layer = is_layer
        self.int_ = odeint
        self.is_ode = is_ode
        
        self.logger = logger
        self.derivative = Derivative(state_c, hidden_c, code_c, n_env, factor, nl, 
                                     self.dataset, is_ode, size, is_layer,
                                     layers, logger, mask, codes_init, device, htype, **kwargs)

    def forward(self, y, t, epsilon=0):
        if epsilon < 1e-3:
            epsilon = 0

        # y = y.permute(2, 0, 1) if self.is_ode else y.permute(2, 0, 1, 3, 4)
        if epsilon == 0:
            # res = self.int_(self.derivative, y0=y[0], t=t, method=self.method, options=self.options)
            res = self.int_(self.derivative, y0=y[:,:,0], t=t, method=self.method, options=self.options)
        else:
            eval_points = np.random.random(len(t)) < epsilon; eval_points[-1] = False
            res = []; start_i = 0
            for i, eval_point in enumerate(eval_points[1:]):
                if eval_point == True:
                    # end_i = i + 1
                    # t_seg = t[start_i:end_i + 1]
                    y0 = y[:,:,start_i]; t_seg = t[start_i:i+2]
                    # res_seg = self.int_(self.derivative, y0=y[start_i], t=t_seg,
                    #                     method=self.method, options=self.options)
                    res_seg = self.int_(self.derivative, y0=y0, t=t_seg,
                                         method=self.method, options=self.options)
                    # if len(res) == 0:
                    #     res.append(res_seg)
                    # else:
                    #     res.append(res_seg[1:])
                    res.append(res_seg if len(res) == 0 else res_seg[1:])
                    start_i = i+1
            # t_seg = t[start_i:]
            res_seg = self.int_(self.derivative, y0=y[:,:,start_i], t=t[start_i:], method=self.method, options=self.options)
            # res_seg = self.int_(self.derivative, y0=y[start_i], t=t_seg, method=self.method,
            #                     options=self.options)
            # if len(res) == 0:
            #     res.append(res_seg)
            # else:
            #     res.append(res_seg[1:])
            res.append(res_seg if len(res) == 0 else res_seg[1:])
            res = torch.cat(res, dim=0)
        dims = [1,2,0] + list(range(y.dim()))[3:]
        # return res.permute(1, 2, 0) if self.is_ode else res.permute(1, 2, 0, 3, 4)
        return res.permute(*dims)
