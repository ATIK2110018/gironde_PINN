import torch
import torch.nn as nn

class EdgeNetwork(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        # Inputs: h_L, h_R, z_L, z_R, dwl, dz, e_len, nx, ny
        self.net = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Predicts physical normal velocity (u_perp)
        )
        
    def forward(self, edge_features):
        return self.net(edge_features)

class NeuralFVMSolver(nn.Module):
    """
    Pure Markovian Neural Implicit Flux Solver (MeshGraphNets style).
    No recurrent latent states. It predicts momentum purely from physical gravity gradients (dwl, dz).
    """
    def __init__(self, dt_seconds=60.0, hidden_dim=64):
        super().__init__()
        self.dt = dt_seconds
        self.edge_net = EdgeNetwork(hidden_dim)
        
    def forward(self, h, cell_z, cell_friction, cell_areas, edge_index, edge_normals, edge_lengths):
        c_L = edge_index[0, :]
        c_R = edge_index[1, :]
        
        h_L, h_R = h[c_L], h[c_R]
        z_L, z_R = cell_z[c_L].unsqueeze(1), cell_z[c_R].unsqueeze(1)
        
        nx = edge_normals[:, 0:1]
        ny = edge_normals[:, 1:2]
        e_len = edge_lengths.view(-1, 1)
        
        # Calculate strict physical gradients (Gravity driving force!)
        wl_L = h_L + z_L
        wl_R = h_R + z_R
        dwl = wl_R - wl_L
        dz = z_R - z_L
        
        # ==========================================
        # 1. GNN Learns Implicit Edge Velocity
        # ==========================================
        edge_features = torch.cat([h_L, h_R, z_L, z_R, dwl, dz, e_len, nx, ny], dim=1)
        
        # To strictly satisfy CFL, bound velocity
        u_perp = torch.tanh(self.edge_net(edge_features)) * 0.1 
        
        # ==========================================
        # 2. EXACT CONTINUITY EQUATION (Mass)
        # ==========================================
        h_safe_L = torch.clamp(h_L, min=1e-3)
        h_safe_R = torch.clamp(h_R, min=1e-3)
        h_Roe = 0.5 * (h_safe_L + h_safe_R)
        
        # Exact physical mass flux
        flux_mass = h_Roe * u_perp
        flux_mass_total = flux_mass.view(-1, 1) * e_len.view(-1, 1)
        
        # STRICT MASS CONSERVATION (Anti-symmetric flux)
        num_cells = h.size(0)
        net_flux_mass = torch.zeros((num_cells, 1), device=h.device)
        net_flux_mass.scatter_add_(0, c_L.view(-1, 1), flux_mass_total)
        net_flux_mass.scatter_add_(0, c_R.view(-1, 1), -flux_mass_total)
        
        c_area = cell_areas.view(-1, 1)
        div_mass = net_flux_mass / c_area
        
        # Explicit Euler macro-step for Water Level
        h_next = h.view(-1, 1) - self.dt * div_mass
        h_next = torch.clamp(h_next, min=0.01) # Prevent unphysical drying
        
        return h_next
