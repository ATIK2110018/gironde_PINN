import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import netCDF4 as nc
import numpy as np
import math
from scipy.interpolate import griddata

# ==========================================
# 1. GRAPH & DATA EXTRACTION FROM DELFT3D
# ==========================================
def extract_graph_from_netcdf(nc_file_path):
    print(f"Reading mesh from {nc_file_path}...")
    dataset = nc.Dataset(nc_file_path, 'r')
    
    if 'mesh2d_node_x' in dataset.variables:
        node_x = dataset.variables['mesh2d_node_x'][:]
        node_y = dataset.variables['mesh2d_node_y'][:]
        edge_nodes = dataset.variables['mesh2d_edge_nodes'][:]
        if 'mesh2d_node_z' in dataset.variables:
            node_z = dataset.variables['mesh2d_node_z'][:]
        else:
            node_z = np.zeros_like(node_x)
    elif 'NetNode_x' in dataset.variables:
        node_x = dataset.variables['NetNode_x'][:]
        node_y = dataset.variables['NetNode_y'][:]
        edge_nodes = dataset.variables['NetLink'][:]
        if 'NetNode_z' in dataset.variables:
            node_z = dataset.variables['NetNode_z'][:]
        else:
            node_z = np.zeros_like(node_x)
    else:
        raise ValueError("Could not find standard node coordinate variables in the NetCDF file.")

    # Detect if coordinates are in Lat/Lon degrees
    is_spherical = (np.max(node_x) < 180.0) and (np.min(node_x) > -180.0)
    if is_spherical:
        mean_lat = np.mean(node_y)
        lat_to_m = 111139.0
        lon_to_m = 111139.0 * math.cos(math.radians(mean_lat))
    else:
        lat_to_m = 1.0
        lon_to_m = 1.0

    edge_index_list, edge_attr_list = [], []
    for i in range(edge_nodes.shape[1] if edge_nodes.shape[0] == 2 else edge_nodes.shape[0]):
        if edge_nodes.shape[0] == 2:
            n1 = int(edge_nodes[0, i]) - 1
            n2 = int(edge_nodes[1, i]) - 1
        else:
            n1 = int(edge_nodes[i, 0]) - 1
            n2 = int(edge_nodes[i, 1]) - 1
            
        if n1 >= 0 and n2 >= 0:
            edge_index_list.extend([[n1, n2], [n2, n1]])
            dx = node_x[n2] - node_x[n1]
            dy = node_y[n2] - node_y[n1]
            dx_m = dx * lon_to_m
            dy_m = dy * lat_to_m
            dist_m = math.sqrt(dx_m**2 + dy_m**2)
            edge_attr_list.extend([[dx_m, dy_m, dist_m], [-dx_m, -dy_m, dist_m]])
            
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
    node_coords = torch.tensor(np.column_stack((node_x, node_y)), dtype=torch.float32)
    node_z = torch.tensor(node_z, dtype=torch.float32)
    
    dataset.close()
    return node_coords, edge_index, edge_attr, node_z

def load_friction_xyz(filepath, node_coords):
    print(f"Loading friction from {filepath}...")
    data = np.loadtxt(filepath)
    xyz_x = data[:, 0]
    xyz_y = data[:, 1]
    xyz_val = data[:, 2]
    
    # Interpolate scatter .xyz to the unstructured mesh nodes
    val_interp = griddata((xyz_x, xyz_y), xyz_val, (node_coords[:, 0], node_coords[:, 1]), method='nearest')
    return torch.tensor(val_interp, dtype=torch.float32)

def load_boundary_pli(filepath, node_coords, threshold=0.002):
    """Finds all nodes along the line segments of a polyline in Lat/Lon."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    coords = np.array([[float(p[0]), float(p[1])] for line in lines[2:] if len(p := line.strip().split()) >= 2])
    node_coords_np = node_coords.numpy()
    boundary_nodes = []
    for i in range(len(coords)-1):
        p1, p2 = coords[i], coords[i+1]
        l2 = np.sum((p2 - p1)**2)
        if l2 == 0: continue
        t = np.clip(np.sum((node_coords_np - p1) * (p2 - p1), axis=1) / l2, 0, 1)
        projection = p1 + t[:, np.newaxis] * (p2 - p1)
        dist = np.sqrt(np.sum((node_coords_np - projection)**2, axis=1))
        boundary_nodes.extend(np.where(dist < threshold)[0])
    return list(set(boundary_nodes))

def load_boundary_bc(filepath):
    """Parses a .bc file for time series forcing data"""
    times = []
    values = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    t = float(parts[0])
                    v = float(parts[1])
                    times.append(t)
                    values.append(v)
                except ValueError:
                    continue
    return np.array(times), np.array(values)

# ==========================================
# 2. GRAPH NEURAL NETWORK ARCHITECTURE
# ==========================================
class HydroMPNNLayer(MessagePassing):
    def __init__(self, node_in_dim, edge_in_dim, out_dim):
        super(HydroMPNNLayer, self).__init__(aggr='mean')
        self.message_mlp = nn.Sequential(
            nn.Linear(node_in_dim * 2 + edge_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(node_in_dim + 64, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.message_mlp(tmp)

    def update(self, aggr_out, x):
        tmp = torch.cat([x, aggr_out], dim=1)
        return self.update_mlp(tmp)

class RiverPIGNN(nn.Module):
    def __init__(self, node_features_dim=5, hidden_dim=64, edge_dim=3):
        super(RiverPIGNN, self).__init__()
        self.encoder = nn.Linear(node_features_dim, hidden_dim)
        self.mpnn1 = HydroMPNNLayer(hidden_dim, edge_dim, hidden_dim)
        self.mpnn2 = HydroMPNNLayer(hidden_dim, edge_dim, hidden_dim)
        self.mpnn3 = HydroMPNNLayer(hidden_dim, edge_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3) 
        )

    def forward(self, x, edge_index, edge_attr):
        h = F.relu(self.encoder(x))
        h = F.relu(self.mpnn1(h, edge_index, edge_attr))
        h = self.mpnn2(h, edge_index, edge_attr)
        delta = self.decoder(h)
        
        delta_eta = torch.clamp(delta[:, 0], min=-0.5, max=0.5)
        delta_u = torch.clamp(delta[:, 1], min=-0.5, max=0.5)
        delta_v = torch.clamp(delta[:, 2], min=-0.5, max=0.5)
        
        out_eta = x[:, 0] + delta_eta
        out_u = x[:, 1] + delta_u
        out_v = x[:, 2] + delta_v
        
        zb = x[:, 3]
        out_eta = torch.clamp(out_eta, min=zb + 0.05, max=zb + 25.0)
        out_u = torch.clamp(out_u, min=-10.0, max=10.0)
        out_v = torch.clamp(out_v, min=-10.0, max=10.0)
        
        return torch.stack([out_eta, out_u, out_v], dim=1)

# ==========================================
# 3. PHYSICS-INFORMED LOSS 
# ==========================================
def physics_loss_swe(state_t, state_t_next, edge_index, edge_attr, dt=60.0, g=9.81):
    eta_t, u_t, v_t, zb, cf = state_t.T
    h_t = torch.clamp(eta_t - zb, min=0.05)
    eta_next, u_next, v_next = state_t_next.T
    deta_dt, du_dt, dv_dt = (eta_next - eta_t)/dt, (u_next - u_t)/dt, (v_next - v_t)/dt
    
    row, col = edge_index
    dx, dy, dist = edge_attr.T
    dist_clamped = torch.clamp(dist, min=1.0)
    
    delta_eta = eta_t[col] - eta_t[row]
    delta_u = u_t[col] - u_t[row]
    delta_v = v_t[col] - v_t[row]
    delta_hu = (h_t[col] * u_t[col]) - (h_t[row] * u_t[row])
    delta_hv = (h_t[col] * v_t[col]) - (h_t[row] * v_t[row])
    
    weight_x, weight_y = dx / (dist_clamped ** 2 + 1e-8), dy / (dist_clamped ** 2 + 1e-8)
    num_nodes = eta_t.size(0)
    
    def scatter_add_pt(src, index, dim_size):
        return torch.zeros(dim_size, dtype=src.dtype, device=src.device).scatter_add_(0, index, src)
    
    grad_eta_x, grad_eta_y = scatter_add_pt(delta_eta * weight_x, row, num_nodes), scatter_add_pt(delta_eta * weight_y, row, num_nodes)
    grad_u_x, grad_u_y = scatter_add_pt(delta_u * weight_x, row, num_nodes), scatter_add_pt(delta_u * weight_y, row, num_nodes)
    grad_v_x, grad_v_y = scatter_add_pt(delta_v * weight_x, row, num_nodes), scatter_add_pt(delta_v * weight_y, row, num_nodes)
    grad_hu_x, grad_hv_y = scatter_add_pt(delta_hu * weight_x, row, num_nodes), scatter_add_pt(delta_hv * weight_y, row, num_nodes)
    
    mass_residual = deta_dt + grad_hu_x + grad_hv_y
    velocity_magnitude = torch.sqrt(u_t**2 + v_t**2 + 1e-8)
    momentum_x_residual = du_dt + u_t * grad_u_x + v_t * grad_u_y + g * grad_eta_x + cf * u_t * velocity_magnitude / h_t
    momentum_y_residual = dv_dt + u_t * grad_v_x + v_t * grad_v_y + g * grad_eta_y + cf * v_t * velocity_magnitude / h_t
    
    return torch.mean(mass_residual**2) + torch.mean(momentum_x_residual**2) + torch.mean(momentum_y_residual**2)
