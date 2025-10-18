import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.integrate import solve_ivp
from functools import partial
from statistics import mean

class LotkaVolterraDataset(Dataset):
    def __init__(self, num_traj_per_env, time_horizon, params, device, dt, batch_t=10, method='RK45', group='train'):
        super().__init__()
        self.num_traj_per_env = num_traj_per_env
        self.num_env = len(params)
        self.len = num_traj_per_env * self.num_env
        self.time_horizon = float(time_horizon)
        self.dt = dt
        self.batch_t = batch_t
        self.params = params
        self.device = device
        self.group = group
        self.max = np.iinfo(np.int32).max
        self.buffer = dict()
        self.method = method
        self.indices = [list(range(env*num_traj_per_env, (env+1)*num_traj_per_env)) for env in range(self.num_env)]

    def _f(self, t, x, env=0):
        d = np.zeros(2)
        d[0] = self.params[env]['alpha'] * x[0] - self.params[env]['beta'] * x[0]*x[1]
        d[1] = self.params[env]['delta'] * x[0]*x[1] - self.params[env]['gamma'] * x[1] 
        return d

    def __getitem__(self, index):
        env, env_index = index // self.num_traj_per_env, index % self.num_traj_per_env
        t = torch.arange(0, self.time_horizon, self.dt).float()
        t0 = torch.randint(t.size(0) - self.batch_t + 1, (1,)).item()
        if self.buffer.get(index) is None:
            np.random.seed(env_index if not self.group == 'test' else self.max-env_index)
            y0 = np.random.random(2) + 1.
            res = solve_ivp(partial(self._f, env=env), (0., self.time_horizon), y0=y0, method=self.method, t_eval=np.arange(0., self.time_horizon, self.dt))
            state = torch.from_numpy(res.y).float()
            self.buffer[index] = state.numpy()
        else:
            state = torch.from_numpy(self.buffer[index])
        return {'state' : state, 't' : t, 'env' : env, 'index': index}

    def __len__(self):
        return self.len