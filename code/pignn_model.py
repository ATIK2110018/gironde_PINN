import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add
import netCDF4 as nc
import numpy as np
import math

# ==========================================
# 1. GRAPH EXTRACTION FROM DELFT3D FM
# ==========================================
def extract_graph_from_netcdf(nc_file_path):
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

    edge_index_list = []
    edge_attr_list = []
    
    for i in range(edge_nodes.shape[1] if edge_nodes.shape[0] == 2 else edge_nodes.shape[0]):
        if edge_nodes.shape[0] == 2:
            n1 = int(edge_nodes[0, i]) - 1
            n2 = int(edge_nodes[1, i]) - 1
        else:
            n1 = int(edge_nodes[i, 0]) - 1
            n2 = int(edge_nodes[i, 1]) - 1
            
        if n1 >= 0 and n2 >= 0:
            edge_index_list.append([n1, n2])
            edge_index_list.append([n2, n1])
            
            dx = node_x[n2] - node_x[n1]
            dy = node_y[n2] - node_y[n1]
            dist = math.sqrt(dx**2 + dy**2)
            edge_attr_list.append([dx, dy, dist])
            edge_attr_list.append([-dx, -dy, dist])
            
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
    node_coords = torch.tensor(np.column_stack((node_x, node_y)), dtype=torch.float32)
    node_z = torch.tensor(node_z, dtype=torch.float32)
    
    dataset.close()
    return node_coords, edge_index, edge_attr, node_z

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
        h = F.relu(self.mpnn2(h, edge_index, edge_attr))
        h = self.mpnn3(h, edge_index, edge_attr)
        out = self.decoder(h)
        return out + x[:, :3]

# ==========================================
# 3. PHYSICS-INFORMED LOSS 
# ==========================================
def physics_loss_swe(state_t, state_t_next, edge_index, edge_attr, dt=1.0, g=9.81):
    eta_t, u_t, v_t, zb, cf = state_t.T
    h_t = eta_t - zb
    
    eta_next, u_next, v_next = state_t_next.T
    
    deta_dt = (eta_next - eta_t) / dt
    du_dt = (u_next - u_t) / dt
    dv_dt = (v_next - v_t) / dt
    
    row, col = edge_index
    dx, dy, dist = edge_attr.T
    
    delta_eta = eta_t[col] - eta_t[row]
    delta_u = u_t[col] - u_t[row]
    delta_v = v_t[col] - v_t[row]
    delta_hu = (h_t[col] * u_t[col]) - (h_t[row] * u_t[row])
    delta_hv = (h_t[col] * v_t[col]) - (h_t[row] * v_t[row])
    
    weight_x = dx / (dist ** 2 + 1e-8)
    weight_y = dy / (dist ** 2 + 1e-8)
    
    grad_eta_x = scatter_add(delta_eta * weight_x, row, dim=0, dim_size=eta_t.size(0))
    grad_eta_y = scatter_add(delta_eta * weight_y, row, dim=0, dim_size=eta_t.size(0))
    grad_u_x = scatter_add(delta_u * weight_x, row, dim=0, dim_size=eta_t.size(0))
    grad_u_y = scatter_add(delta_u * weight_y, row, dim=0, dim_size=eta_t.size(0))
    grad_v_x = scatter_add(delta_v * weight_x, row, dim=0, dim_size=eta_t.size(0))
    grad_v_y = scatter_add(delta_v * weight_y, row, dim=0, dim_size=eta_t.size(0))
    grad_hu_x = scatter_add(delta_hu * weight_x, row, dim=0, dim_size=eta_t.size(0))
    grad_hv_y = scatter_add(delta_hv * weight_y, row, dim=0, dim_size=eta_t.size(0))
    
    mass_residual = deta_dt + grad_hu_x + grad_hv_y
    
    velocity_magnitude = torch.sqrt(u_t**2 + v_t**2 + 1e-8)
    friction_x = cf * u_t * velocity_magnitude / (h_t + 1e-8)
    momentum_x_residual = du_dt + u_t * grad_u_x + v_t * grad_u_y + g * grad_eta_x + friction_x
    
    friction_y = cf * v_t * velocity_magnitude / (h_t + 1e-8)
    momentum_y_residual = dv_dt + u_t * grad_v_x + v_t * grad_v_y + g * grad_eta_y + friction_y
    
    physics_loss = torch.mean(mass_residual**2) + torch.mean(momentum_x_residual**2) + torch.mean(momentum_y_residual**2)
    return physics_loss
