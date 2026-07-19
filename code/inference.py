import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.collections as mcoll
from train_kaggle import * # Import all classes and data functions from our main script

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Load Data (Fast)
print("Loading data for inference...")
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

# Load truth data for comparison
has_truth_data = False
map_files = [os.path.join(r, f) for r, d, files in os.walk('.') for f in files if 'map.nc' in f.lower() or 'flowfm_map.nc' in f.lower()]
if map_files:
    true_times, true_eta, true_u, true_v = load_truth_data(map_files[0], node_coords)
    has_truth_data = True

node_coords = node_coords.to(device)
edge_index = edge_index.to(device)
edge_attr = edge_attr.to(device)
node_z = node_z.to(device)
friction = friction.to(device)

# 2. Init Model and Load Weights
print("Loading saved best weights...")
model = RiverPIGNN(node_features_dim=6, hidden_dim=128, edge_dim=3).to(device)
model.load_state_dict(torch.load("pignn_weights_best.pth", map_location=device, weights_only=True))
model.eval()

# 3. Inference Loop
dt = 600.0
max_time = 72 * 3600
current_time = 0.0

def get_interp_val(t, times, values):
    return np.interp(t, times, values)

node_idx = 5000
print(f"Extracting timeseries for Node {node_idx}...")

state_t = torch.zeros((num_nodes, 6), dtype=torch.float32).to(device)
state_t[:, 0] = torch.clamp(torch.tensor(0.0, device=device), min=node_z + 0.1)
state_t[:, 3] = node_z
state_t[:, 4] = friction

pred_times, pred_wl, true_wl_list = [], [], []

with torch.no_grad():
    while current_time < max_time:
        target_port_wl = get_interp_val(current_time + dt, t_port, v_port)
        target_gar_q = get_interp_val(current_time + dt, t_garonne, v_garonne) * 0.001
        target_dor_q = get_interp_val(current_time + dt, t_dordogne, v_dordogne) * 0.001
        
        state_t[:, 5] = 0.0
        state_t[bnd_port, 5] = torch.tensor(target_port_wl, dtype=torch.float32, device=device).expand(len(bnd_port))
        state_t[bnd_garonne, 5] = torch.tensor(target_gar_q, dtype=torch.float32, device=device).expand(len(bnd_garonne))
        state_t[bnd_dordogne, 5] = torch.tensor(target_dor_q, dtype=torch.float32, device=device).expand(len(bnd_dordogne))
        state_t[bnd_port, 0] = torch.tensor(target_port_wl, dtype=torch.float32, device=device).expand(len(bnd_port))
        
        state_t_next = model(state_t, edge_index, edge_attr)
        
        pred_times.append(current_time / 3600.0)
        pred_wl.append(state_t_next[node_idx, 0].item())
        
        if has_truth_data:
            t_target = current_time + dt
            if t_target <= true_times[0]: target_eta = true_eta[0][node_idx].item()
            elif t_target >= true_times[-1]: target_eta = true_eta[-1][node_idx].item()
            else:
                idx2 = np.searchsorted(true_times, t_target)
                idx1 = idx2 - 1
                w = (t_target - true_times[idx1]) / (true_times[idx2] - true_times[idx1] + 1e-8)
                target_eta = ((1 - w) * true_eta[idx1][node_idx] + w * true_eta[idx2][node_idx]).item()
            true_wl_list.append(target_eta)
        
        state_t[:, :3] = state_t_next.clone()
        current_time += dt

# 4. Plot
print("Plotting!")
plt.figure(figsize=(14, 6))
if has_truth_data:
    plt.plot(pred_times, true_wl_list, 'b-', label='True Water Level (Delft3D)', linewidth=2)
plt.plot(pred_times, pred_wl, 'r--', label='Predicted Water Level (Best Epoch)', linewidth=2)
plt.title(f"Best Epoch Water Level Memorization Comparison (Node {node_idx})")
plt.xlabel("Time (hours)")
plt.ylabel("Water Level (m)")
plt.legend()
plt.grid(True)
plt.savefig("timeseries_validation.png")
print("Saved timeseries_validation.png!")
plt.show()
