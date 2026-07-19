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
from model import FVM_PINN_Net
from trainer import train_fvm_pinn
from visualization import generate_water_level_gif, plot_timeseries_comparison

def load_delft3d_teacher_data(nc_file_path):
    print(f"Loading Teacher Data from {nc_file_path}...")
    dataset = nc.Dataset(nc_file_path, 'r')
    times_sec = dataset.variables['time'][:]
    wl_matrix = dataset.variables['mesh2d_s1'][:] 
    
    # Fill masked values with 0
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
        print("Error: Could not find 'FlowFM_net.nc' or 'map.nc' in /kaggle/input/ or ../data/")
        print("Please ensure your dataset is attached to the Kaggle notebook!")
    else:
        print(f"Found Mesh: {nc_mesh_path}")
        print(f"Found Teacher Data: {nc_data_path}")
        if fric_path: print(f"Found Friction: {fric_path}")
        else: print("Warning: No friction file found, using default n=0.02")
        
        # 1. Extract Geometry directly from the output map file to guarantee shape match
        cell_coords, cell_z, cell_areas, edge_index, edge_normals, edge_lengths = extract_fvm_geometry(nc_data_path, device)
        
        # Load friction
        if fric_path:
            cell_friction = load_friction_xyz(fric_path, cell_coords, device)
        else:
            cell_friction = torch.full((cell_coords.size(0),), 0.02, dtype=torch.float32, device=device)
        
        # 2. Load Teacher Data
        times_hr, true_wl_matrix = load_delft3d_teacher_data(nc_data_path)
        
        # Filter for 30h to 72h
        mask = (times_hr >= 30) & (times_hr <= 72)
        times_hr = times_hr[mask]
        true_wl_matrix = true_wl_matrix[mask]
        
        # 3. Build Model
        model = FVM_PINN_Net(hidden_dim=128, num_layers=6)
        
        # 4. Train Model
        train_fvm_pinn(model, cell_coords, cell_z, cell_areas, edge_index, edge_normals, edge_lengths, cell_friction, times_hr, true_wl_matrix, epochs=5000, lr=1e-3, device=device)
        
        # 5. Generate Visualizations
        print("\nGenerating Output Visualizations...")
        os.makedirs("/kaggle/working/outputs", exist_ok=True)
        
        # Generate Animation
        generate_water_level_gif(cell_coords, times_hr, true_wl_matrix, model, 30*3600, 42*3600, 600, "/kaggle/working/outputs/wave_animation.gif", device)
        
        # Generate Timeseries for a specific node (e.g. Node 5000)
        node_idx = 5000
        pred_wl = []
        model.eval()
        with torch.no_grad():
            for t in times_hr:
                x = cell_coords[node_idx, 0:1].unsqueeze(0).to(device)
                y = cell_coords[node_idx, 1:2].unsqueeze(0).to(device)
                t_tensor = torch.tensor([[t * 3600.0]], dtype=torch.float32, device=device)
                xi, _, _ = model(x, y, t_tensor)
                pred_wl.append(xi.item())
        
        plot_timeseries_comparison(times_hr, pred_wl, true_wl_matrix[:, node_idx], node_idx, f"/kaggle/working/outputs/timeseries_node_{node_idx}.png")
        print("All done! Check /kaggle/working/outputs/ for your visualizations.")
