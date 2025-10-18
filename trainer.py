import statistics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from forecasters import Forecaster
from forecasters_coda import Forecaster as Forecaster_coda
from datasets import init_dataloaders
from utils import set_requires_grad, init_weights, batch_transform, batch_transform_inverse
from utils import mean_absolute_percentage_error as mape
from norm import CalculateNorm
import os
from time import time
import numpy as np
from tqdm import tqdm
from torch import optim
from ei import EIModule
from datasets import init_dataloaders
from configure import set_configure, make_dir

class Trainer:
    def __init__(self, args, k=0.99, loss='mse', nupdate=10, nlog=50, plot=False, load_pretrained_model=None):
        # Initialize EI
        # args.device = torch.device("cpu")
        self.args = args
        self.device = args.device
        self.coda = args.coda
        self.decomp_type = 'leads_decomp' if 'leads' in args.exp_type else args.exp_type
        self.train_data, self.test_data, self.params = init_dataloaders(args)
        make_dir(args, plot=plot)
        self.test_loss = args.test_loss
        self.test_loss_func = F.mse_loss if args.test_loss == 'mse' else mape
        self.n_env = args.assumed_nenv
        self.real_n_env = int(len(self.params))
        self.ei = EIModule(self, args)
        # Initialize Backend
        if self.coda:
            self.regul = args.coda_norm
            self.net = Forecaster_coda(args, n_env=self.n_env).to(args.device)
            init_weights(self.net, init_type='default', init_gain=args.init_gain)
            self.load_model(load_pretrained_model)
            self.net.train(True)
            self.net.derivative.net_leaf.update_ghost()
            self.l_m = args.l_m; self.l_c = args.l_c; self.l_t = args.l_t
        else:
            self.net = Forecaster(args, n_env=self.n_env, decomp_type=self.decomp_type).to(args.device)
            init_weights(self.net, init_type='normal', init_gain=args.init_gain)
            self.load_model(load_pretrained_model)
            self.cal_norm = CalculateNorm(self.net.right_model)
            self.lambda_inv = 0 if args.exp_type == 'leads_no_min' else args.lambda_inv
            self.lipschitz_ratio = args.factor_lip
        self.net.train(True)
        # Initialize Norm and Optimizer
        self.loss = nn.MSELoss() if loss == 'mse' else nn.L1Loss()
        self.loss_ = nn.MSELoss(reduction='none')
        self.optimizer = optim.Adam(self.net.parameters(), lr=args.lr, betas=(0.9, 0.999))
        # Store Configs
        self.logger = args.logger
        self.batch_size = args.batch_size
        self.min_op = args.norm
        self.logger.info(f'Oracle mode: {args.oracle}')
        self.logger.info(f'Coda: {args.coda}')
        self.logger.info(f'Coda Norm: {args.coda_norm}')
        self.path = args.path
        self.epsilon = self.base_epsilon = k; self.decay_power = 1.
        self.nepoch = args.epochs; self.nlog = nlog; self.nupdate = nupdate
        self.loss_test_min = 1e5
        self.loss_test_min_metric = 1e5
    
    def run(self):
        for epoch in range(self.nepoch):
            for i, data in enumerate(self.train_data, 0):
                # Update EI
                ei = self.ei.train(data, self.coda)
                # Update Backend
                metrics = self.train(data, ei) if not self.coda else self.train_coda(data, ei)
                # Also only use f to predict y, Cal loss
                self.log(epoch, i, metrics)
                if (epoch*(len(self.train_data)) + (i+1)) % self.nupdate == 0:
                    self.decay_power += 1; self.epsilon = self.base_epsilon**self.decay_power
                    self.logger.info(f'espilon: {self.epsilon}')
                if (epoch*(len(self.train_data)) + (i+1)) % self.nlog == 0:
                    metric_test = self.test(epoch) if not self.coda else self.test_coda(epoch)
                    self.log(epoch, i, metric_test)
    
    def train(self, batch, ei):
        state, t = batch['state'].clone().to(self.device).requires_grad_(), batch['t'][0].to(self.device)
        # Reorder State Batch
        batch_states = [state[ei==i] for i in range(self.n_env)]
        preds_train, preds = [], []
        for env, mini_batch_state in enumerate(batch_states):
            if len(mini_batch_state) != 0:
                preds_train.append(self.net(mini_batch_state, t, env=env, epsilon=self.epsilon))
                preds.append(self.net(mini_batch_state, t, env=env))
            else:
                preds_train.append([]); preds.append([])
        # Caluate Main Loss
        all_preds_train = [i for i in preds_train if len(i)>0]; all_preds = [i for i in preds if len(i)>0]
        loss_train = self.loss(torch.cat(all_preds_train), torch.cat(batch_states))
        # Caluate/Add Regularization Loss
        loss_reg = self.cal_reg_leads(batch_states) if not self.coda else self.cal_reg_coda()
        loss_total = loss_train + loss_reg
        # Update and return Metrics
        self.optimizer.zero_grad(); loss_total.backward(); self.optimizer.step()
        metrics = {'loss': F.mse_loss(torch.cat(all_preds), torch.cat(batch_states)), 'loss_train': loss_train}
        for env in range(self.n_env):
            if len(preds[env])>0:
                metrics[f'loss_e{env}']= F.mse_loss(batch_states[env], preds[env]).detach()
        return metrics
    
    def train_coda(self, batch, ei):
        state, t = batch['state'].clone().to(self.device).requires_grad_(), batch['t'][0].to(self.device)
        # Reorder State Batch
        batch_states = torch.nn.utils.rnn.pad_sequence([state[ei==i] for i in range(self.n_env)]).transpose(0, 1)
        mask = (batch_states == 0)
        mask = ~torch.all(mask.view(mask.shape[0], mask.shape[1], -1), dim=-1)
        ei = torch.from_numpy(ei) if type(ei)==np.ndarray else ei
        inputs = batch_transform(batch_states.reshape([-1]+list(batch_states.shape[2:])),
                                 torch.bincount(ei).max())
        self.net.derivative.net_leaf.update_ghost()
        outputs = batch_transform_inverse(self.net(inputs, t, epsilon=self.epsilon), self.n_env)
        outputs = outputs.view([self.n_env, -1]+list(batch_states.shape[2:]))
        loss = F.mse_loss(outputs, batch_states, reduction='none')
        loss = loss.mean(list(range(2, loss.dim())))*mask.float()
        loss_ = loss.sum()/mask.sum()
        # Caluate/Add Regularization Loss
        loss_total = loss_ + self.cal_reg_coda()
        # Update and return Metrics
        self.optimizer.zero_grad(); loss_total.backward(); self.optimizer.step()
        metrics = {'loss_train': loss_.detach()}
        for env in range(self.n_env):
            if torch.any(mask[env, :]):
                mask_ = loss[env, :].sum()/mask[env, :].sum()
                metrics[f'loss_e{env}']= mask_.detach()
        return metrics
        
    def cal_reg_leads(self, batch_states, weight=None):
        if self.decomp_type == 'leads_decomp':
            if self.min_op == 'sum_spectral':
                batch_states = [i for i in batch_states if len(i)]
                derivs = [self.net.right_model[env](batch) for env, batch in enumerate(batch_states)]
                if weight is not None:
                    loss_ops = [((deriv_e.norm(p=2, dim=1) / (state_e.norm(p=2, dim=1) + 1e-5))**2).mean(1) for deriv_e, state_e in zip(derivs, batch_states)]
                    loss_op = torch.mul(torch.stack(loss_ops), weight).sum()
                    # print (loss_op)
                else:
                    loss_op_a = self.cal_norm.calculate_spectral_norm().sum()
                    loss_ops = [((deriv_e.norm(p=2, dim=1) / (state_e.norm(p=2, dim=1) + 1e-5))**2).mean() for deriv_e, state_e in zip(derivs, batch_states)]
                    loss_op = loss_op_a*self.lipschitz_ratio + torch.stack(loss_ops).sum()
            elif self.min_op == 'f_norm':
                loss_op = (self.cal_norm.calculate_frobenius_norm()**2).sum()
            return loss_op*max(self.lambda_inv, 0)
        return 0
    
    def cal_reg_coda(self, ):
        loss_reg_row = torch.zeros(1).to(self.device)
        loss_reg_col = torch.zeros(1).to(self.device)
        loss_reg_theta = torch.zeros(1).to(self.device)
        loss_reg_code = torch.zeros(1).to(self.device)
        regul = self.regul
        if "l2t" in regul:
            for i in range(self.n_env):
                loss_reg_theta += torch.norm(self.net.derivative.net_hyper(self.net.derivative.codes[i])) ** 2
        if "l2c" in regul:
            loss_reg_code += (torch.norm(self.net.derivative.codes, dim=0) ** 2).sum()
        if "l12m" in regul:
            loss_reg_row += (torch.norm(self.net.derivative.net_hyper.weight, dim=1)).sum()
        if "l2m" in regul:
            loss_reg_row += torch.norm(self.net.derivative.net_hyper.weight) ** 2
        if "l12col" in regul:
            loss_reg_col += torch.norm(self.net.derivative.net_hyper.weight, dim=0).sum()
        if "lcos" in regul:
            weight = self.net.derivative.net_hyper.weight # n x n_xi
            norm_weight = torch.norm(weight, dim=1, keepdim=True)
            weight_normalized = weight / norm_weight
            codes = self.net.derivative.codes  # n_env x n_xi
            norm_codes = torch.norm(codes, dim=1, keepdim=True)
            codes_normalized = codes / norm_codes
            cosines = F.linear(codes, weight_normalized)
            loss_reg_row += torch.norm(cosines, dim=0).sum()

        reg_loss = self.l_m*(loss_reg_row+loss_reg_col) + self.l_t*loss_reg_theta + self.l_c*loss_reg_code
        return reg_loss

    def test(self, epoch):
        loss_test, loss_test_dis = [], []
        for data_test in tqdm(self.test_data):
            state, t = data_test['state'].clone().to(self.device), data_test['t'][0].to(self.device)
            ei = self.ei.infer(state, t, data_test['env'])
            states = [state[ei==i] for i in range(self.n_env)]
            preds = torch.cat([self.net(states[i], t, env=i) for i in range(self.n_env) if len(states[i])])
            loss_test.append(F.mse_loss(preds, torch.cat(states)).item())
            loss_test_dis.append(self.test_loss_func(preds, torch.cat(states)).item())
        loss_mean = statistics.mean(loss_test)
        loss_mean_dis = statistics.mean(loss_test_dis)
        if self.loss_test_min > loss_mean:
            self.save_model(epoch, loss_mean); self.loss_test_min = loss_mean
        self.loss_test_min_metric = min(self.loss_test_min_metric, loss_mean_dis)
        metric_test = {'loss_test_mean': loss_mean, 'loss_test_std': statistics.stdev(loss_test), 'current_best': self.loss_test_min, 'current_best_dis': self.loss_test_min_metric,}
        return metric_test
    
    def test_coda(self, epoch):
        loss_test, loss_test_dis = [], []
        for data_test in tqdm(self.test_data):
            state, t = data_test['state'].clone().to(self.device), data_test['t'][0].to(self.device)
            ei = self.ei.infer(state, t, data_test['env'], self.coda)
            batch_states = torch.nn.utils.rnn.pad_sequence([state[ei==i] for i in range(self.n_env)]).transpose(0, 1)
            mask = (batch_states == 0)
            mask = ~torch.all(mask.view(mask.shape[0], mask.shape[1], -1), dim=-1)
            ei = torch.from_numpy(ei) if type(ei)==np.ndarray else ei
            inputs = batch_transform(batch_states.reshape([-1]+list(batch_states.shape[2:])),
                                    torch.bincount(ei).max())
            self.net.derivative.net_leaf.update_ghost()
            outputs = batch_transform_inverse(self.net(inputs, t), self.n_env)
            outputs = outputs.view([self.n_env, -1]+list(batch_states.shape[2:]))
            loss = F.mse_loss(outputs, batch_states, reduction='none')
            loss = loss.mean(list(range(2, loss.dim())))*mask.float()
            loss = loss.sum()/mask.sum()
            loss_test.append(loss.cpu().item())
            loss_dis = self.test_loss_func(outputs, batch_states, reduction=False)
            loss_dis = loss_dis.mean(list(range(2, loss_dis.dim())))*mask.float()
            loss_dis = loss_dis.sum()/mask.sum()
            loss_test_dis.append(loss_dis.cpu().item())
        loss_mean = statistics.mean(loss_test)
        loss_mean_dis = statistics.mean(loss_test_dis)
        if self.loss_test_min > loss_mean:
            self.save_model(epoch, loss_mean); self.loss_test_min = loss_mean
        self.loss_test_min_metric = min(self.loss_test_min_metric, loss_mean_dis)
        metric_test = {'loss_test_mean': loss_mean, 'loss_test_std': statistics.stdev(loss_test), 'current_best': self.loss_test_min, 'current_best_dis': self.loss_test_min_metric,}
        return metric_test

    def log(self, epoch, itera, metrics):
        message = '[{step}][{epoch}/{max_epoch}][{i}/{max_i}]'.format(
            step=epoch*len(self.train_data)+itera+1, epoch=epoch+1,
            max_epoch=self.nepoch, i=itera+1, max_i=len(self.train_data))
        for name, value in metrics.items():
            message += ' | {name}: {value:.3e}'.format(name=name, value=value)
        self.logger.info(message)

    def load_model(self, load_pretrained_model):
        if load_pretrained_model is not None:
            assert len(self.net.left_model) == 1
            print("Load pretrained model")
            pretrained_dict = torch.load(load_pretrained_model)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k.find('left_model') != -1}
            model_dict = self.net.state_dict()
            model_dict.update(pretrained_dict)
            self.net.load_state_dict(model_dict)
            set_requires_grad(self.net.left_model, False)

    def save_model(self, epoch, loss_test_min):
        torch.save({'epoch': epoch, 'model_state_dict': self.net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss_test_min}, self.path + f'/model_{loss_test_min:.3e}.pt')
    
if __name__ == '__main__':
    args = set_configure()
    trainer = Trainer(args, nlog=50, plot=False)
    trainer.run()