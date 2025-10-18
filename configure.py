import os
import sys
import argparse
import logging
import random
from datetime import datetime
import numpy as np
import torch

def get_configs():
    # General Config
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='lv', help='lv - Lotka-Volterra, gs - Gray-Scott, ns - Navier-Stokes')
    parser.add_argument("-p", "--path", type=str, default='./exp', help='Root path for the experiments.')
    parser.add_argument("-e", "--exp_type", type=str, default='leads', help='leads/leads_no_min/one_for_all/one_per_env')
    parser.add_argument('-d', '--device', type=int, default=0)
    parser.add_argument('-l', '--lr', type=float, default=1.e-3)
    parser.add_argument('--lr_ei', type=float, default=1.e-3)
    parser.add_argument('-n', '--norm', type=str, default='sum_spectral')
    parser.add_argument('--assumed_nenv', type=int, default=5, help='assumed num of envs, when -1 automatic refers to true value')
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--train_ei_epoch', type=int, default=1)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--random_assign', default=False, action='store_true')
    parser.add_argument('--oracle', default=False, action='store_true')
    parser.add_argument('--coda', default=True, action='store_true')
    parser.add_argument('--coda_norm', type=str, default='l12m-l2c')
    parser.add_argument('--dim_context', type=int, default=2)
    parser.add_argument('--da_pre_epoch', type=int, default=10000)
    parser.add_argument('--da_epoch', type=int, default=2000)
    parser.add_argument('--da_unfreeze_f', default=False, action='store_true')
    # parser.add_argument('--test_loss', type=str, default='mse')
    parser.add_argument('--test_loss', type=str, default='mape')
    args = parser.parse_args()
    args.hidden = 64
    args.codes_init = None
    # Specific Config
    if args.dataset == 'lv':
        args.batch_size = 4
        args.var_dim = 2
        args.test_ei_timestep = 2
        args.net_type ='mlp'
        args.factor = 1.
        args.method = 'rk4'
        args.init_gain = 0.05; args.lambda_inv = 1 / 5e3; args.factor_lip = 1.e-2
        args.l_m = 1e-6; args.l_c = 1e-4; args.l_t = 1e-6
    if args.dataset == 'gs':
        args.batch_size = 4
        args.var_dim = 2
        args.test_ei_timestep = 2
        args.net_type ='conv'
        args.factor = 1.e-3
        args.method = 'rk4'
        args.init_gain = 0.1; args.lambda_inv = 1 / 1e3; args.factor_lip = 1.e-2
        args.l_m = 1e-5; args.l_c = 1e-2; args.l_t = 1e-5
    if args.dataset == 'ns':
        args.batch_size = 4
        args.var_dim = 1
        args.test_ei_timestep = 3
        args.net_type ='fno'
        args.factor = 1.
        args.method = 'euler'
        args.init_gain = 0.1; args.lambda_inv = 1 / 1e5; args.factor_lip = 1.e-4
        args.l_m = 2e-3; args.l_c = 1e-3; args.l_t = 2e-3
    return args

def get_logger(log_dir):
    logger = logging.getLogger()
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt); logger.addHandler(console)
    logfile = logging.FileHandler(log_dir, 'w')
    logfile.setFormatter(fmt); logger.addHandler(logfile)
    return logger

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)  

def make_dir(args, da=False, plot=False):
    args.path = os.path.join(args.path, datetime.now().strftime("%Y-%m-%d %H-%M-%S.%f"))
    if not da: 
        args.path = args.path + f'_{args.dataset}_nenv_{args.assumed_nenv}' + f'_epochs_{args.epochs}'
    else:
        args.path = args.path + f'_{args.dataset}_nenv_{args.assumed_nenv}' + '_DA' + f'_seed_{args.seed}'
    args.path = args.path if not plot else args.path+f'_plot_{args.oracle}_{args.coda}_{args.coda_norm}'
    os.makedirs(args.path)
    args.logger = get_logger(os.path.join(args.path,'log.txt'))

def set_configure():
    args = get_configs()
    args.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    return args