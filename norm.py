import torch
from torch import nn
from torch.nn import Parameter

class CalculateNorm:
    def __init__(self, module, power_iterations=5):
        self.module = module
        assert isinstance(module, nn.ModuleList)
        self.power_iterations = power_iterations
        self.u, self.v = dict(), dict()
        for i, module in enumerate(self.module):
            for name, w in module.named_parameters():
                if name.find('bias') == -1 and name.find('beta') == -1:
                    height = w.data.shape[0]
                    width = w.view(height, -1).data.shape[1]
                    u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
                    v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
                    u.data = l2normalize(u.data)
                    v.data = l2normalize(v.data)
                    self.u[f'{i},{name}'] = u
                    self.v[f'{i},{name}'] = v

    def calculate_spectral_norm(self):
        # Adapted to complex weights
        sigmas = [0. for i in range(len(self.module))]
        for i, module in enumerate(self.module):
            for name, w in module.named_parameters():
                if name.find('bias') == -1 and name.find('beta') == -1:
                    u, v = self.u[f'{i},{name}'], self.v[f'{i},{name}']
                    height = w.data.shape[0]
                    for _ in range(self.power_iterations):
                        v.data = l2normalize(torch.mv(w.view(height,-1).data.T, u.data))
                        u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))
                    sigma = torch.conj(u).dot(w.view(height, -1).mv(v))
                    sigmas[i] = sigmas[i] + sigma.real**2 if torch.is_complex(sigma) else sigmas[i] + sigma**2
        return torch.stack(sigmas)

    def calculate_frobenius_norm(self):
        # Only used for linear case
        sigmas = [0. for i in range(len(self.module))]
        for i, module in enumerate(self.module):
            for name, w in module.named_parameters():
                if name.find('bias') == -1 and name.find('beta') == -1:
                    sigmas[i] = sigmas[i] + torch.norm(w)
        return torch.stack(sigmas)

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)