import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint
from networks import *

class DerivativeEstimatorMultiEnv(nn.Module):
    def __init__(self, left_model, right_model, n_env, decomp_type, ignore_right=False):
        super().__init__()
        self.left_model = left_model
        self.right_model = right_model
        self.decomp_type = decomp_type
        self.env = None
        self.enable_right = None
        self.ignore_right = ignore_right
        self.n_env = n_env

    def set_env(self, env):
        self.env = env

    def forward(self, t, u):
        left_res, right_res = None, None
        if self.decomp_type == 'leads_decomp':
            # 1 model on the left, n_env models on the right
            assert len(self.left_model) == 1
            assert self.env is not None
            left_res = self.left_model[0](u)
            right_res = self.right_model[self.env](u)
        elif self.decomp_type == 'one_for_all':
            # 1 model on the left, 1 model on the right
            # This case is adapted to experments with changing environments
            assert len(self.left_model) == len(self.right_model)
            change_every = self.n_env // len(self.left_model)
            left_res  = self.left_model[self.env // change_every](u)
            right_res = self.right_model[self.env // change_every](u) 
        elif self.decomp_type == 'one_per_env':
            # n_env models on the left, n_env models on the right
            assert len(self.left_model) == len(self.right_model)
            assert self.env is not None
            left_res = self.left_model[self.env](u)
            right_res = self.right_model[self.env](u)
        else:
            change_every = len(self.right_model) // len(self.left_model)
            left_res = self.left_model[self.env // change_every](u)
            right_res = self.right_model[self.env](u)

        if right_res is not None and (self.enable_right and not self.ignore_right):
            return left_res + right_res
        else:
            return left_res

class Forecaster(nn.Module):
    def __init__(self, args, n_env, n_left=None, n_right=None, options=None, decomp_type=None, ignore_right=False):
        super().__init__()
        in_c = out_c = args.var_dim
        if decomp_type == 'leads_decomp':
            n_left = 1; n_right = n_env
        elif decomp_type == 'one_for_all':
            if n_left is None and n_right is None:
                n_left = n_right = 1
            else:
                n_left = n_right
        elif decomp_type == 'one_per_env':
            n_left = n_right = n_env
        else:
            n_left = n_left; n_right = n_right

        if args.net_type == 'mlp':
            self.left_model  = nn.ModuleList([MLPEstimator(in_c=in_c, out_c=out_c, hidden=args.hidden, factor=args.factor) for _ in range(n_left)])
            self.right_model = nn.ModuleList([MLPEstimator(in_c=in_c, out_c=out_c, hidden=args.hidden, factor=args.factor) for _ in range(n_right)])
        elif args.net_type == 'linear':
            self.left_model  = nn.ModuleList([Linear(in_c=in_c, out_c=out_c, factor=args.factor) for _ in range(n_left)])
            self.right_model = nn.ModuleList([Linear(in_c=in_c, out_c=out_c, factor=args.factor) for _ in range(n_right)])
        elif args.net_type in ['conv', 'fno']:
            self.left_model  = nn.ModuleList([ConvNetEstimator(in_c=in_c, out_c=out_c, hidden=args.hidden, factor=args.factor, net_type=args.net_type) for _ in range(n_left)])
            self.right_model = nn.ModuleList([ConvNetEstimator(in_c=in_c, out_c=out_c, hidden=args.hidden, factor=args.factor, net_type=args.net_type) for _ in range(n_right)])
        else:
            raise NotImplementedError
        self.derivative_estimator = DerivativeEstimatorMultiEnv(self.left_model, self.right_model, n_env=n_env, decomp_type=decomp_type, ignore_right=ignore_right)
        self.method = args.method
        self.options = options
        self.odeint = odeint
        self.net_type = args.net_type
        
    def forward(self, y, t, env, enable_right=True, epsilon=None):
        self.derivative_estimator.enable_right = enable_right
        self.derivative_estimator.env = env
        if epsilon is None:
            ret = self.odeint(self.derivative_estimator, y0=y[:,:,0], t=t, method=self.method, options=self.options)
        else:
            eval_points = np.random.random(len(t)) < epsilon; eval_points[-1] = False
            res = []; start_i = 0
            for i, eval_point in enumerate(eval_points[1:]):
                if eval_point:
                    y0 = y[:,:,start_i]; t_seg = t[start_i:i+2]
                    res_seg = self.odeint(self.derivative_estimator, y0=y0, t=t_seg, method=self.method, options=self.options)
                    res.append(res_seg if len(res) == 0 else res_seg[1:])
                    start_i = i+1
            res_seg = self.odeint(self.derivative_estimator, y0=y[:,:,start_i], t=t[start_i:], method=self.method, options=self.options)
            res.append(res_seg if len(res) == 0 else res_seg[1:])
            ret = torch.cat(res, dim=0)
        dims = [1,2,0] + list(range(y.dim()))[3:]
        return ret.permute(*dims)
