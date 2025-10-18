import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
import numpy as np
from time import time
from utils import pretty, batch_transform, batch_transform_inverse
from torch.autograd import grad

class EIModule(nn.Module):
    def __init__(self, trainer, args):
        super(EIModule, self).__init__()
        # Load Parameters
        self.trainer = trainer
        self.n_env = args.assumed_nenv
        self.real_n_env = int(len(trainer.params))
        self.batch_size = args.batch_size * self.real_n_env # Rollout Numbers
        self.test_size = args.test_size
        self.var_dim = args.var_dim
        self.test_ei_timestep = args.test_ei_timestep
        self.train_ei_epoch = args.train_ei_epoch
        self.device = args.device
        self.lr_ei = args.lr_ei
        self.random_assign = args.random_assign
        self.oracle = args.oracle
         # Initialize Loss
        self.hard_sum = self.var_dim
        # self.eii = torch.ones(self.n_env, self.batch_size).uniform_().to(self.device).requires_grad_()
        self.eii = torch.ones(self.n_env, self.batch_size).to(self.device).requires_grad_()
        self.optimizeri = optim.Adam([{'params': self.eii, 'lr': args.lr_ei}])

    # def train(self, batch):
    #     for i in range(self.train_ei_epoch):
    #         state, t = batch['state'].clone().to(self.device).requires_grad_(), batch['t'][0].to(self.device)
    #         eiis = self.eii[:, batch['index']]
    #         probs = torch.nn.functional.softmax(eiis, dim=0)
    #         batch_states = [state.clone() for i in range(self.n_env)]
    #         loss = -self.trainer.cal_regularization(batch_states, probs)
    #         loss.backward()
    #         self.optimizeri.step()
    #     # Return Hard Labels
    #     return self.eii[:, batch['index']].argmax(0)
    
    # def infer(self, state, t, ei_epoch=30):
    #     eii = torch.ones(self.n_env, self.test_size).to(self.device).requires_grad_()
    #     optimizeri = optim.Adam([{'params': eii, 'lr': self.lr_ei}])
    #     for i in range(ei_epoch):
    #         state = state.requires_grad_()
    #         probs = torch.nn.functional.softmax(eii, dim=0)
    #         batch_states = [state.clone() for i in range(self.n_env)]
    #         loss = -self.trainer.cal_regularization(batch_states, probs)
    #         loss.backward()
    #         optimizeri.step()
    #     # Return Hard Labels
    #     return eii.argmax(0)
    
    def train(self, batch, coda=False):
        if self.oracle:
            return batch['env']
        elif self.random_assign:
            return np.random.randint(0, self.n_env, batch['state'].size(0))
        elif not coda:
            state, t = batch['state'].clone().to(self.device).requires_grad_(), batch['t'][0].to(self.device)
            batch_states = [state for _ in range(self.n_env)]
            preds_train = [self.trainer.net(mbs, t, env=env, epsilon=self.trainer.epsilon) for env, mbs in enumerate(batch_states)]
            # loss_train = F.mse_loss(torch.stack(preds_train), state, reduce=False)
            loss_train = torch.stack([F.mse_loss(pred_, state, reduction='none') for pred_ in preds_train])
            return loss_train.mean(list(range(2, loss_train.dim()))).argmin(0)
        else:
            state, t = batch['state'].clone().to(self.device).requires_grad_(), batch['t'][0].to(self.device)
            batch_states = torch.concatenate([state for _ in range(self.n_env)])
            # batch_states = torch.repeat_interleave(state, self.n_env)
            input = batch_transform(batch_states, len(state))
            preds = batch_transform_inverse(self.trainer.net(input, t, epsilon=self.trainer.epsilon), self.n_env)
            loss = F.mse_loss(preds, batch_states, reduction='none')
            label = loss.mean(list(range(1, loss.dim()))).reshape(self.n_env, len(state))
            return label.argmin(0)
    
    def infer(self, state, t, true_env=None, coda=False):
        if self.oracle:
            return true_env
        elif self.random_assign:
            return np.random.randint(0, self.n_env, state.size(0))
        elif not coda:
            state = state[:,:,:self.test_ei_timestep]; t = t[:self.test_ei_timestep]
            batch_states = [state for _ in range(self.n_env)]
            preds = [self.trainer.net(mbs, t, env=env, epsilon=self.trainer.epsilon) for env, mbs in enumerate(batch_states)]
            loss = torch.stack([F.mse_loss(pred_, state, reduction='none') for pred_ in preds])
            return loss.mean(list(range(2, loss.dim()))).argmin(0)
        else:
            state = state[:,:,:self.test_ei_timestep]; t = t[:self.test_ei_timestep]
            batch_states = torch.concatenate([state for _ in range(self.n_env)])
            input = batch_transform(batch_states, len(state))
            preds = batch_transform_inverse(self.trainer.net(input, t), self.n_env)
            loss = F.mse_loss(preds, batch_states, reduction='none')
            label = loss.mean(list(range(1, loss.dim()))).reshape(self.n_env, len(state))
            return label.argmin(0)
            # label = loss.mean(list(range(1, loss.dim()))).reshape(len(state), self.n_env)
            # return label.argmin(1)
