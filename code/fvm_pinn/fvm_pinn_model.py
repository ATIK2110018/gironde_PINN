import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class HydroPINN(nn.Module):
    def __init__(self, layers=[3, 128, 128, 128, 128, 3]):
        super().__init__()
        # Input: (t, x, y) -> Output: (h, u, v)
        net = []
        for i in range(len(layers)-2):
            net.append(nn.Linear(layers[i], layers[i+1]))
            net.append(nn.Tanh())
        net.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*net)
        
    def forward(self, t, coords):
        # coords: (N, 2), t: (1, 1) or scalar
        if t.dim() == 0 or t.size(0) == 1:
            t = t.expand(coords.shape[0], 1)
            
        inputs = torch.cat([t, coords], dim=1)
        out = self.net(inputs)
        
        # We output (h, u, v)
        # We can add a softplus to h to guarantee positivity, but keeping it linear allows the network to explore freely
        h = out[:, 0:1]
        u = out[:, 1:2]
        v = out[:, 2:3]
        return h, u, v

class FVMPINNTrainer:
    def __init__(self, fvm_model, cell_coords_m, true_wl_matrix, times_seconds, boundary_mask):
        """
        fvm_model: Instance of GPUHydrodynamicModel containing exact mesh geometry
        cell_coords_m: cell center coordinates in meters (N, 2)
        true_wl_matrix: observed water levels (T, N)
        times_seconds: time array
        """
        self.device = fvm_model.device
        self.fvm = fvm_model
        
        # Normalize coordinates for Neural Network stability
        self.coords_mean = torch.mean(cell_coords_m, dim=0)
        self.coords_std = torch.std(cell_coords_m, dim=0)
        self.norm_coords = (cell_coords_m - self.coords_mean) / self.coords_std
        
        self.times = torch.tensor(times_seconds, dtype=torch.float32, device=self.device)
        self.time_mean = torch.mean(self.times)
        self.time_std = torch.std(self.times)
        
        # Initialize PINN
        self.pinn = HydroPINN().to(self.device)
        self.optimizer = optim.Adam(self.pinn.parameters(), lr=1e-3)
        
        self.true_wl = torch.tensor(true_wl_matrix, dtype=torch.float32, device=self.device)
        self.boundary_mask = boundary_mask
        
        # Define h_still reference state (bed elevation) for the FVM solver
        self.h_still = torch.clamp(-self.fvm.cell_z, min=0.01)

    def get_normalized_t(self, t_val):
        return (t_val - self.time_mean) / self.time_std

    def compute_physics_loss(self, t_val, dt=5.0):
        # 1. PINN Prediction at Current Time
        norm_t_curr = self.get_normalized_t(torch.tensor([t_val], device=self.device))
        h_curr, u_curr, v_curr = self.pinn(norm_t_curr, self.norm_coords)
        
        # Clamp h strictly for the FVM solver so it doesn't crash on unphysical NN predictions
        h_curr_safe = torch.clamp(h_curr, min=0.005)
        
        # 2. Physics Engine explicit step (The Exact Discrete FVM Residual)
        h_fvm_next, u_fvm_next, v_fvm_next, _ = self.fvm.simulate_one_step(
            h_curr_safe, u_curr, v_curr, self.h_still, dt
        )
        
        # 3. PINN Prediction at Future Time
        norm_t_next = self.get_normalized_t(torch.tensor([t_val + dt], device=self.device))
        h_next, u_next, v_next = self.pinn(norm_t_next, self.norm_coords)
        
        # 4. Residual (FVM Physics Loss)
        # The NN's future prediction MUST exactly match the rigid FVM calculation
        loss_h = nn.MSELoss()(h_next, h_fvm_next.detach())
        loss_u = nn.MSELoss()(u_next, u_fvm_next.detach())
        loss_v = nn.MSELoss()(v_next, v_fvm_next.detach())
        
        return loss_h + loss_u + loss_v
        
    def train_step(self, t_idx):
        t_val = self.times[t_idx]
        true_h = self.true_wl[t_idx].unsqueeze(1)
        
        # 1. Evaluate Data Loss
        norm_t_curr = self.get_normalized_t(t_val.unsqueeze(0))
        h_curr, u_curr, v_curr = self.pinn(norm_t_curr, self.norm_coords)
        
        # PINN predicts Depth (h). True data is Water Level (elevation).
        # Water Level = Depth + Bed Elevation (cell_z)
        wl_curr = h_curr + self.fvm.cell_z
        data_loss = nn.MSELoss()(wl_curr, true_h)
        
        # 2. Evaluate Exact FVM Physics Loss
        phys_loss = self.compute_physics_loss(t_val, dt=1.0)
        
        # Combine and Backpropagate
        total_loss = data_loss + 0.5 * phys_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return data_loss.item(), phys_loss.item()
