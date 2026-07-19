import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import netCDF4 as nc
import numpy as np
import math
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

# Setup GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==========================================
# 1. DATA EXTRACTION FUNCTIONS
# ==========================================
def extract_graph_from_netcdf(nc_file_path):
    print(f"Reading mesh from {nc_file_path}...")
    dataset = nc.Dataset(nc_file_path, 'r')
    
    if 'mesh2d_node_x' in dataset.variables:
        node_x = dataset.variables['mesh2d_node_x'][:]
        node_y = dataset.variables['mesh2d_node_y'][:]
        edge_nodes = dataset.variables['mesh2d_edge_nodes'][:]
        node_z = dataset.variables['mesh2d_node_z'][:] if 'mesh2d_node_z' in dataset.variables else np.zeros_like(node_x)
    elif 'NetNode_x' in dataset.variables:
        node_x = dataset.variables['NetNode_x'][:]
        node_y = dataset.variables['NetNode_y'][:]
        edge_nodes = dataset.variables['NetLink'][:]
        node_z = dataset.variables['NetNode_z'][:] if 'NetNode_z' in dataset.variables else np.zeros_like(node_x)
    else:
        raise ValueError("Could not find standard node coordinate variables.")

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
    val_interp = griddata((data[:, 0], data[:, 1]), data[:, 2], (node_coords[:, 0], node_coords[:, 1]), method='nearest')
    return torch.tensor(val_interp, dtype=torch.float32)

def load_boundary_pli(filepath, node_coords, threshold=0.002):
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
    times, values = [], []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    times.append(float(parts[0]))
                    values.append(float(parts[1]))
                except ValueError:
                    continue
    return np.array(times), np.array(values)

def load_truth_data(nc_file_path, node_coords):
    print(f"Loading true state data from {nc_file_path} for data loss...")
    dataset = nc.Dataset(nc_file_path, 'r')
    times = dataset.variables['time'][:]
    
    eta_raw = dataset.variables['mesh2d_s1'][:]
    u_raw = dataset.variables['mesh2d_ucx'][:]
    v_raw = dataset.variables['mesh2d_ucy'][:]
    
    eta_raw = np.ma.filled(eta_raw, fill_value=0.0)
    u_raw = np.ma.filled(u_raw, fill_value=0.0)
    v_raw = np.ma.filled(v_raw, fill_value=0.0)
    
    if eta_raw.shape[1] == node_coords.shape[0]:
        eta_nodes, u_nodes, v_nodes = eta_raw, u_raw, v_raw
    else:
        face_x = dataset.variables['mesh2d_face_x'][:]
        face_y = dataset.variables['mesh2d_face_y'][:]
        face_coords = np.column_stack((face_x, face_y))
        
        tree = cKDTree(face_coords)
        node_coords_np = node_coords.cpu().numpy()
        _, indices = tree.query(node_coords_np)
        
        eta_nodes = eta_raw[:, indices]
        u_nodes = u_raw[:, indices]
        v_nodes = v_raw[:, indices]
        
    dataset.close()
    return times, torch.tensor(eta_nodes, dtype=torch.float32), torch.tensor(u_nodes, dtype=torch.float32), torch.tensor(v_nodes, dtype=torch.float32)

# ==========================================
# 2. GRAPH NEURAL NETWORK ARCHITECTURE
# ==========================================
class HydroMPNNLayer(MessagePassing):
    def __init__(self, node_in_dim, edge_in_dim, out_dim):
        super(HydroMPNNLayer, self).__init__(aggr='add') # Correctly using 'add'
        self.message_mlp = nn.Sequential(nn.Linear(node_in_dim * 2 + edge_in_dim, 128), nn.ReLU(), nn.Linear(128, 128))
        self.update_mlp = nn.Sequential(nn.Linear(node_in_dim + 128, 128), nn.ReLU(), nn.Linear(128, out_dim))
    
    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return x + out if x.size(-1) == out.size(-1) else out
        
    def message(self, x_i, x_j, edge_attr): return self.message_mlp(torch.cat([x_i, x_j, edge_attr], dim=1))
    def update(self, aggr_out, x): return self.update_mlp(torch.cat([x, aggr_out], dim=1))

class RiverPIGNN(nn.Module):
    def __init__(self, node_features_dim=6, hidden_dim=128, edge_dim=3):
        super(RiverPIGNN, self).__init__()
        self.encoder = nn.Linear(node_features_dim, hidden_dim)
        self.mpnn1 = HydroMPNNLayer(hidden_dim, edge_dim, hidden_dim)
        self.mpnn2 = HydroMPNNLayer(hidden_dim, edge_dim, hidden_dim)
        self.mpnn3 = HydroMPNNLayer(hidden_dim, edge_dim, hidden_dim) 
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x, edge_index, edge_attr):
        h = F.relu(self.encoder(x))
        h = F.relu(self.mpnn1(h, edge_index, edge_attr))
        h = F.relu(self.mpnn2(h, edge_index, edge_attr))
        h = F.relu(self.mpnn3(h, edge_index, edge_attr))
        delta = self.decoder(h)
        
        delta_eta = 0.5 * torch.tanh(delta[:, 0])
        delta_u = 0.5 * torch.tanh(delta[:, 1])
        delta_v = 0.5 * torch.tanh(delta[:, 2])
        
        out_eta = x[:, 0] + delta_eta
        out_u = x[:, 1] + delta_u
        out_v = x[:, 2] + delta_v
        
        zb = x[:, 3]
        
        min_eta = zb + 0.05
        max_eta = zb + 25.0
        
        out_eta = torch.where(out_eta < min_eta, min_eta + 0.1 * (out_eta - min_eta), out_eta)
        out_eta = torch.where(out_eta > max_eta, max_eta + 0.1 * (out_eta - max_eta), out_eta)
        
        out_u = torch.where(out_u < -10.0, -10.0 + 0.1 * (out_u + 10.0), out_u)
        out_u = torch.where(out_u > 10.0, 10.0 + 0.1 * (out_u - 10.0), out_u)
        
        out_v = torch.where(out_v < -10.0, -10.0 + 0.1 * (out_v + 10.0), out_v)
        out_v = torch.where(out_v > 10.0, 10.0 + 0.1 * (out_v - 10.0), out_v)
        
        return torch.stack([out_eta, out_u, out_v], dim=1)

# ==========================================
# 3. PHYSICS LOSS 
# ==========================================
def physics_loss_swe(state_t, state_t_next, edge_index, edge_attr, dt=60.0, g=9.81):
    eta_t, u_t, v_t, zb, cf, _ = state_t.T 
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


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    netcdf_path = "data/input/FlowFM_net.nc"
    node_coords, edge_index, edge_attr, node_z = extract_graph_from_netcdf(netcdf_path)
    num_nodes = node_coords.size(0)
    
    friction = load_friction_xyz("data/input/frictioncoefficient.xyz", node_coords)
    bnd_port = load_boundary_pli("data/input/port block.pli", node_coords, threshold=0.002)
    bnd_garonne = load_boundary_pli("data/input/garonne.pli", node_coords, threshold=0.002)
    bnd_dordogne = load_boundary_pli("data/input/dordogne.pli", node_coords, threshold=0.002)
    
    t_port, v_port = load_boundary_bc("data/input/WaterLevel.bc")
    t_garonne, v_garonne = load_boundary_bc("data/input/garonne.bc")
    t_dordogne, v_dordogne = load_boundary_bc("data/input/dordogne.bc")
    
    node_coords = node_coords.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    node_z = node_z.to(device)
    friction = friction.to(device)
    
    model = RiverPIGNN(node_features_dim=6, hidden_dim=128, edge_dim=3).to(device) 
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    # ------------------------------------------
    # FAST TRAINING PARAMETERS
    # ------------------------------------------
    dt = 600.0                # 10 minute time step
    max_time = 72 * 3600      # Train on first 72 hours
    num_epochs = 10           # Increased epochs
    unroll_steps = 12         # Unroll less frequently
    # ------------------------------------------

    def get_interp_val(t, times, values):
        return np.interp(t, times, values)

    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\n--- EPOCH {epoch+1}/{num_epochs} ---")
        current_time = 0.0
        epoch_total_loss = 0.0
        
        initial_water_level = torch.clamp(torch.tensor(0.0, device=device), min=node_z + 0.1)
        state_t = torch.zeros((num_nodes, 6), dtype=torch.float32).to(device)
        state_t[:, 0] = initial_water_level
        state_t[:, 3] = node_z
        state_t[:, 4] = friction
        
        map_files = []
        for search_dir in ['/kaggle', '.']:
            if os.path.exists(search_dir):
                for root, dirs, files in os.walk(search_dir):
                    for f in files:
                        if 'map.nc' in f.lower() or 'flowfm_map.nc' in f.lower():
                            map_files.append(os.path.join(root, f))
        
        has_truth_data = False
        if map_files:
            map_file_path = map_files[0]
            true_times, true_eta, true_u, true_v = load_truth_data(map_file_path, node_coords.cpu())
            has_truth_data = True
        
        optimizer.zero_grad()
        accumulated_loss = 0.0
        step_count = 0
        
        alpha = max(0.0, 1.0 - (epoch / 15.0))
        if not has_truth_data: alpha = 0.0
        
        while current_time < max_time:
            model.train()
            
            if has_truth_data:
                t_target = current_time
                if t_target <= true_times[0]:
                    curr_true_eta = true_eta[0].to(device)
                elif t_target >= true_times[-1]:
                    curr_true_eta = true_eta[-1].to(device)
                else:
                    idx2 = np.searchsorted(true_times, t_target)
                    idx1 = idx2 - 1
                    t1, t2 = true_times[idx1], true_times[idx2]
                    w = (t_target - t1) / (t2 - t1 + 1e-8)
                    curr_true_eta = ((1 - w) * true_eta[idx1] + w * true_eta[idx2]).to(device)
            
            target_port_wl = get_interp_val(current_time, t_port, v_port)
            target_gar_q = get_interp_val(current_time, t_garonne, v_garonne) * 0.001
            target_dor_q = get_interp_val(current_time, t_dordogne, v_dordogne) * 0.001

            state_t[:, 5] = 0.0
            state_t[bnd_port, 5] = torch.tensor(target_port_wl, dtype=torch.float32, device=device).expand(len(bnd_port))
            state_t[bnd_garonne, 5] = torch.tensor(target_gar_q, dtype=torch.float32, device=device).expand(len(bnd_garonne))
            state_t[bnd_dordogne, 5] = torch.tensor(target_dor_q, dtype=torch.float32, device=device).expand(len(bnd_dordogne))
            
            if has_truth_data:
                state_t[bnd_port, 0] = curr_true_eta[bnd_port]
            else:
                state_t[bnd_port, 0] = torch.tensor(target_port_wl, dtype=torch.float32, device=device).expand(len(bnd_port))

            predicted_state_next = model(state_t, edge_index, edge_attr)
            
            raw_loss_physics = physics_loss_swe(state_t, predicted_state_next, edge_index, edge_attr, dt=dt)
            
            if has_truth_data:
                t_target = current_time + dt
                if t_target <= true_times[0]:
                    target_eta = true_eta[0].to(device)
                    target_u = true_u[0].to(device)
                    target_v = true_v[0].to(device)
                elif t_target >= true_times[-1]:
                    target_eta = true_eta[-1].to(device)
                    target_u = true_u[-1].to(device)
                    target_v = true_v[-1].to(device)
                else:
                    idx2 = np.searchsorted(true_times, t_target)
                    idx1 = idx2 - 1
                    t1, t2 = true_times[idx1], true_times[idx2]
                    w = (t_target - t1) / (t2 - t1 + 1e-8)
                    
                    target_eta = ((1 - w) * true_eta[idx1] + w * true_eta[idx2]).to(device)
                    target_u = ((1 - w) * true_u[idx1] + w * true_u[idx2]).to(device)
                    target_v = ((1 - w) * true_v[idx1] + w * true_v[idx2]).to(device)
                
                loss_data = F.mse_loss(predicted_state_next[:, 0], target_eta) + \
                            F.mse_loss(predicted_state_next[:, 1], target_u) + \
                            F.mse_loss(predicted_state_next[:, 2], target_v)
            else:
                loss_data = torch.tensor(0.0, device=device)
            
            scaled_physics_loss = raw_loss_physics * 10000.0
            
            step_loss = scaled_physics_loss + loss_data
            accumulated_loss = accumulated_loss + step_loss
            epoch_total_loss += step_loss.item()
            step_count += 1
            
            if current_time % 3600 == 0: 
                print(f"Time: {current_time/3600:.1f}h | Total: {step_loss.item():.2f}")
            
            if step_count % unroll_steps == 0:
                accumulated_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                accumulated_loss = 0.0
                
                state_t = torch.zeros((num_nodes, 6), dtype=torch.float32, device=device)
                if has_truth_data:
                    state_t[:, 0] = (alpha * target_eta) + ((1 - alpha) * predicted_state_next[:, 0].detach())
                    state_t[:, 1] = (alpha * target_u) + ((1 - alpha) * predicted_state_next[:, 1].detach())
                    state_t[:, 2] = (alpha * target_v) + ((1 - alpha) * predicted_state_next[:, 2].detach())
                else:
                    state_t[:, :3] = predicted_state_next.detach().clone()
                    
                state_t[:, 3] = node_z
                state_t[:, 4] = friction
            else:
                state_t = torch.zeros((num_nodes, 6), dtype=torch.float32, device=device)
                state_t[:, :3] = predicted_state_next.clone()
                state_t[:, 3] = node_z
                state_t[:, 4] = friction
                
            current_time += dt

        if isinstance(accumulated_loss, torch.Tensor) and accumulated_loss.requires_grad:
            accumulated_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch_total_loss < best_loss:
            best_loss = epoch_total_loss
            torch.save(model.state_dict(), "pignn_weights_best.pth")
            print(f"--> Saved new best model! (Epoch Loss: {best_loss:.2f})")

    print("Training Complete!")
