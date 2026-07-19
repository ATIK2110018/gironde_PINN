import torch
import torch.nn as nn
import numpy as np

class FVM_PINN_Net(nn.Module):
    """
    Standard MLP for Continuous FVM-PINN.
    Input: (x, y, t)
    Output: (xi, hu, hv)
    Where xi is the water level perturbation, hu/hv are unit discharges.
    """
    def __init__(self, hidden_dim=128, num_layers=6):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(3, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
            
        layers.append(nn.Linear(hidden_dim, 3))
        self.net = nn.Sequential(*layers)
        
        # Scaling parameters to ensure inputs (x,y,t) are well-behaved [-1, 1]
        self.register_buffer("x_shift", torch.tensor(0.0))
        self.register_buffer("x_scale", torch.tensor(1.0))
        self.register_buffer("y_shift", torch.tensor(0.0))
        self.register_buffer("y_scale", torch.tensor(1.0))
        self.register_buffer("t_shift", torch.tensor(0.0))
        self.register_buffer("t_scale", torch.tensor(1.0))

    def set_scales(self, x_min, x_max, y_min, y_max, t_min, t_max):
        self.x_shift.fill_((x_max + x_min) / 2.0)
        self.x_scale.fill_(2.0 / (x_max - x_min + 1e-8))
        
        self.y_shift.fill_((y_max + y_min) / 2.0)
        self.y_scale.fill_(2.0 / (y_max - y_min + 1e-8))
        
        self.t_shift.fill_((t_max + t_min) / 2.0)
        self.t_scale.fill_(2.0 / (t_max - t_min + 1e-8))

    def forward(self, x, y, t):
        # Normalize inputs
        x_norm = (x - self.x_shift) * self.x_scale
        y_norm = (y - self.y_shift) * self.y_scale
        t_norm = (t - self.t_shift) * self.t_scale
        
        inputs = torch.cat([x_norm, y_norm, t_norm], dim=1)
        out = self.net(inputs)
        
        # outputs: xi, hu, hv
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]
