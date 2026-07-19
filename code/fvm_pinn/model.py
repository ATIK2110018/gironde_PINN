import torch
import torch.nn as nn
import numpy as np

class FVM_PINN_Net(nn.Module):
    """
    Standard MLP for Continuous FVM-PINN with Fourier Features.
    Input: (x, y, t)
    Output: (xi, hu, hv)
    """
    def __init__(self, hidden_dim=128, num_layers=6, fourier_features=32):
        super().__init__()
        
        self.fourier_features = fourier_features
        # 3 inputs (x,y,t) * fourier_features * 2 (sin, cos)
        input_dim = 3 * fourier_features * 2
        
        # Random Fourier Feature matrix (fixed during training)
        self.register_buffer("B", torch.randn(3, fourier_features) * 2.0 * np.pi)
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
            
        layers.append(nn.Linear(hidden_dim, 3))
        self.net = nn.Sequential(*layers)
        
        # Scaling parameters
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
        # Normalize inputs to [-1, 1]
        x_norm = (x - self.x_shift) * self.x_scale
        y_norm = (y - self.y_shift) * self.y_scale
        t_norm = (t - self.t_shift) * self.t_scale
        
        # We also concatenate for each of the 3 dimensions individually to ensure distinct frequencies
        proj_x = x_norm * self.B[0, :]
        proj_y = y_norm * self.B[1, :]
        proj_t = t_norm * self.B[2, :]
        
        encoded_x = torch.cat([torch.sin(proj_x), torch.cos(proj_x)], dim=1)
        encoded_y = torch.cat([torch.sin(proj_y), torch.cos(proj_y)], dim=1)
        encoded_t = torch.cat([torch.sin(proj_t), torch.cos(proj_t)], dim=1)
        
        encoded = torch.cat([encoded_x, encoded_y, encoded_t], dim=1)
        
        out = self.net(encoded)
        
        # outputs: xi, hu, hv
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]
