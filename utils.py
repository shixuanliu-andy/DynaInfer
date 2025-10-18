import os
import sys
import json
import csv
import itertools
from typing import Any, Callable, Dict, Optional, TextIO, Tuple, Type, TypeVar, Union, cast
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from torch import Tensor
from torch.nn import Parameter

class Logger(object):
    "Lumberjack class - duplicates sys.stdout to a log file and it's okay"
    def __init__(self, filename, mode="a"):
        self.stdout = sys.stdout
        self.file = open(filename, mode)
        sys.stdout = self

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self):
        if self.stdout != None:
            sys.stdout = self.stdout
            self.stdout = None
        if self.file != None:
            self.file.close()
            self.file = None

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def pretty_wrap(text, title=None, width=80):
    table = pt.PrettyTable(header=title is not None,)
    table.field_names = [title]
    for t in text.split('\n'):
        for i in range(0, len(t), width):
            table.add_row([t[i: i + width]])
    return table

def mean_absolute_percentage_error(y_pred, y_true, epsilon=1e-8, reduction=True):
    if reduction:
        mask = ~torch.isnan(y_true) & ~torch.isinf(y_true)  # Mask for valid values
    # if coda:
    #     mask = mask & (y_true != 0)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    ape = torch.abs((y_true - y_pred) / (y_true + epsilon))
    mape = torch.mean(ape) if reduction else ape
    return 100 * mape

def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'default':
                pass
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.weight is not None:
                init.normal_(m.weight.data, 1.0, init_gain)
            if m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    net.apply(init_func)

def pretty(vector):
    if type(vector) is list:
        vlist = vector
    elif type(vector) is np.ndarray:
        vlist = vector.reshape(-1).tolist()
    else:
        vlist = vector.view(-1).tolist()
    return "[" + ", ".join("{:+.4f}".format(vi) for vi in vlist) + "]"


def load_txt(input_dir, filter=False, merge_list=False):
    if filter:
        ans = [line for line in csv.reader(open(input_dir), delimiter='\t') if len(line)>1]
    else:
        ans = [line for line in csv.reader(open(input_dir), delimiter='\t')]
    if merge_list:
        ans = list(itertools.chain.from_iterable(ans))
    return ans

def write_txt(info_list, out_dir):
    with open(out_dir, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(info_list)

def load_json(input_dir, serial_key=False):
    ret_dict = json.load(open(input_dir))
    if serial_key:
        ret_dict = {tuple([int(i) for i in k.split('_')]):[tuple(l) for l in v] for k,v in ret_dict.items()}
    return ret_dict

def write_json(info_dict, out_dir, serial_key=False):
    if serial_key:
        info_dict = {'_'.join([str(i) for i in k]):v for k,v in info_dict.items()}
    with open(out_dir, "w") as f:
        json.dump(info_dict, f)

def batch_transform(batch, minibatch_size):
    # batch: b x c x t
    t = batch.shape[2:]
    new_batch = []
    for i in range(minibatch_size):
        sample = batch[i::minibatch_size]  # n_env x c x t
        sample = sample.reshape(-1, *t)
        new_batch.append(sample)
    return torch.stack(new_batch)  # minibatch_size x n_env * c x t

def batch_transform_loss(batch, minibatch_size):
    # batch: b x c x t
    t = batch.shape[2:]
    new_batch = []
    for i in range(minibatch_size):
        sample = batch[i::minibatch_size]  # n_env x c x t
        new_batch.append(sample)
    return torch.stack(new_batch)

def batch_transform_inverse(new_batch, n_env):
    # new_batch: minibatch_size x n_env * c x t
    c = new_batch.size(1) // n_env
    t = new_batch.shape[2:]
    new_batch = new_batch.reshape(-1, n_env, c, *t)
    batch = []
    for i in range(n_env):
        sample = new_batch[:, i]  # minibatch_size x c x t
        batch.append(sample)
    return torch.cat(batch)  # b x c x t

def count_parameters(model, mode='ind'):
    if mode == 'ind':
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    elif mode == 'layer':
        return sum(1 for p in model.parameters() if p.requires_grad)
    elif mode == 'row':
        n_mask = 0
        for p in model.parameters():
            if p.dim() == 1:
                n_mask += 1
            else:
                n_mask += p.size(0) 
        return n_mask

def get_n_param_layer(net, layers):
    n_param = 0
    for name, p in net.named_parameters():
        if any(f"net.{layer}" in name for layer in layers):
            n_param += p.numel()
    return n_param
