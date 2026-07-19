import torch
import torch.nn as nn

class SWE_PINN(nn.Module):
    """
    Classical Coordinate-Based Physics-Informed Neural Network (HydroNet style).
    Takes (x, y, t) as input and predicts (h, u, v).
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3) # Predicts water depth (h) and velocity (u, v)
        )
        
        # Normalization constants (set during training)
        self.x_mean = 0
        self.x_std = 1
        self.y_mean = 0
        self.y_std = 1
        self.t_mean = 0
        self.t_std = 1
        
    def forward(self, x, y, t):
        inputs = torch.cat([x, y, t], dim=1)
        out = self.net(inputs)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]
