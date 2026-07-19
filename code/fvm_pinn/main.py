import os
try:
    import netCDF4
except ImportError:
    print("Installing netCDF4...")
    os.system('pip install -q netCDF4')

import torch
import numpy as np
import netCDF4 as nc
from data_extractor import extract_fvm_geometry, load_friction_xyz
from neural_fvm import NeuralFVMSolver
from train_gnn import train_neural_fvm

def load_delft3d_teacher_data(nc_file_path):
    print(f"Loading Teacher Data from {nc_file_path}...")
    dataset = nc.Dataset(nc_file_path, 'r')
    times_sec = dataset.variables['time'][:]
    wl_matrix = dataset.variables['mesh2d_s1'][:] 
    wl_matrix = np.ma.filled(wl_matrix, 0.0)
    dataset.close()
    times_hr = times_sec / 3600.0
    return times_hr, wl_matrix

def find_file(filename_keyword, search_paths=['/kaggle/input', '../data']):
    for path in search_paths:
        if os.path.exists(path):
            for root, dirs, files in os.walk(path):
                for f in files:
                    if filename_keyword.lower() in f.lower():
                        return os.path.join(root, f)
    return None

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")
    
    nc_mesh_path = find_file('flowfm_net.nc')
    nc_data_path = find_file('map.nc')
    fric_path = find_file('frictioncoefficient.xyz')
    
    if not nc_mesh_path or not nc_data_path:
        print("Error: Could not find 'FlowFM_net.nc' or 'map.nc'")
    else:
        # 1. Extract EXACT Geometry for the solver
        cell_coords, cell_z, cell_areas, edge_index, edge_normals, edge_lengths = extract_fvm_geometry(nc_data_path, device)
        
        if fric_path:
            cell_friction = load_friction_xyz(fric_path, cell_coords, device)
        else:
            cell_friction = torch.full((cell_coords.size(0),), 0.02, dtype=torch.float32, device=device)
        
        # 2. Load Teacher Data
        times_hr, true_wl_matrix = load_delft3d_teacher_data(nc_data_path)
        
        mask = (times_hr >= 30) & (times_hr <= 72)
        times_hr = times_hr[mask]
        true_wl_matrix = true_wl_matrix[mask]
        
        # 3. Build Neural FVM Solver
        model = NeuralFVMSolver()
        
        # 4. Train the Differentiable Solver
        train_neural_fvm(model, cell_coords, cell_z, cell_areas, edge_index, edge_normals, edge_lengths, cell_friction, times_hr, true_wl_matrix, epochs=2000, lr=1e-3, device=device)
        
        # 5. Full Simulation Rollout (Testing it like Delft3D)
        print("\nRunning Full Simulation Forward...")
        model.eval()
        
        # Get boundary indices to force boundary condition during rollout
        x_min, x_max = cell_coords[:,0].min(), cell_coords[:,0].max()
        y_min, y_max = cell_coords[:,1].min(), cell_coords[:,1].max()
        boundary_mask = (cell_coords[:,0] < x_min + 0.05*(x_max-x_min)) | \
                        (cell_coords[:,0] > x_max - 0.05*(x_max-x_min)) | \
                        (cell_coords[:,1] < y_min + 0.05*(y_max-y_min)) | \
                        (cell_coords[:,1] > y_max - 0.05*(y_max-y_min))
        bc_indices = torch.where(boundary_mask)[0]
        
        h_current = torch.tensor(true_wl_matrix[0], dtype=torch.float32, device=device).unsqueeze(1) - cell_z.unsqueeze(1)
        h_current = torch.clamp(h_current, min=0.01)
        u_current = torch.zeros_like(h_current)
        v_current = torch.zeros_like(h_current)
        
        pred_wl_matrix = []
        
        with torch.no_grad():
            for t_idx in range(len(times_hr)):
                # Convert depth (h) back to water level (eta) for saving
                eta = h_current.squeeze(1) + cell_z
                pred_wl_matrix.append(eta.cpu().numpy())
                
                # Hard Boundary Condition for the NEXT step
                if t_idx < len(times_hr) - 1:
                    true_h_next = torch.tensor(true_wl_matrix[t_idx + 1], dtype=torch.float32, device=device).unsqueeze(1) - cell_z.unsqueeze(1)
                    true_h_next = torch.clamp(true_h_next, min=0.01)
                    h_current[bc_indices] = true_h_next[bc_indices]
                
                # Mathematical Step Forward (t -> t+1)
                h_current, u_current, v_current = model(h_current, u_current, v_current, cell_z, cell_friction, cell_areas, edge_index, edge_normals, edge_lengths)
                
        pred_wl_matrix = np.array(pred_wl_matrix)
        
        import matplotlib.pyplot as plt
        node_idx = 5000
        plt.figure(figsize=(12, 5))
        plt.plot(times_hr, true_wl_matrix[:, node_idx], 'k-', label='True Water Level (Delft3D)')
        plt.plot(times_hr, pred_wl_matrix[:, node_idx], 'r--', label='Neural FVM Solver')
        plt.title(f'Neural FVM Autoregressive Rollout (Node {node_idx})')
        plt.xlabel('Time (hours)')
        plt.ylabel('Water Level (m)')
        plt.legend()
        os.makedirs("/kaggle/working/outputs", exist_ok=True)
        plt.savefig(f"/kaggle/working/outputs/neural_fvm_timeseries.png")
        print("Simulation complete! Plot saved to /kaggle/working/outputs/")
