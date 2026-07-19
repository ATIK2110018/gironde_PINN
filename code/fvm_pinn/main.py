import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}")
    
    from data_extractor import extract_fvm_geometry
    print("Extracting FVM Geometry from /kaggle/input/datasets/atikurr/gironde-hydro-out/FlowFM_map.nc...")
    cell_coords, cell_z, cell_areas, edge_index, edge_normals, edge_lengths = extract_fvm_geometry("/kaggle/input/datasets/atikurr/gironde-hydro-out/FlowFM_map.nc", device=device)
    
    import netCDF4 as nc
    ds = nc.Dataset("/kaggle/input/datasets/atikurr/gironde-hydro-out/FlowFM_map.nc")
    true_wl_matrix = ds.variables['mesh2d_s1'][:]
    times_seconds = ds.variables['time'][:]
    ds.close()
    
    # Define Boundary Nodes (Ocean is at the West end of the Gironde Estuary, i.e., minimum Longitude X)
    x_coords = cell_coords[:, 0]
    x_min = torch.min(x_coords)
    # Select cells within a small margin of the western-most edge as the tidal boundary (0.005 degrees ~ 400m thick)
    boundary_mask = (x_coords < x_min + 0.005)
    
    print(f"Identified {torch.sum(boundary_mask).item()} boundary cells at the ocean mouth.")
    
    # We use the true data ONLY to extract the initial condition and the tidal forcing at the boundary!
    initial_wl = true_wl_matrix[0, :]
    
    # Extract boundary tidal forcing across time
    # Average the tidal forcing at the boundary mask to create a single tidal wave time-series
    boundary_wl_matrix = np.mean(true_wl_matrix[:, boundary_mask.cpu().numpy()], axis=1)
    
    # ==========================================
    # GPU Explicit FVM Simulation (No Neural Networks!)
    # ==========================================
    from numerical_model import GPUHydrodynamicModel
    
    model = GPUHydrodynamicModel(
        cell_coords=cell_coords.cpu().numpy(),
        cell_areas=cell_areas.squeeze(1).cpu().numpy(),
        cell_z=cell_z.cpu().numpy(),
        edge_index=edge_index.cpu().numpy(),
        edge_normals=edge_normals.cpu().numpy(),
        edge_lengths=edge_lengths.squeeze(1).cpu().numpy(),
        boundary_mask=boundary_mask,
        device=device
    )
    
    # Run the pure mathematical solver forward in time!
    pred_wl_matrix = model.simulate(
        initial_wl=initial_wl,
        boundary_wl_matrix=boundary_wl_matrix,
        times_seconds=times_seconds
    )
    
    # ==========================================
    # Plotting
    # ==========================================
    os.makedirs('/kaggle/working/outputs', exist_ok=True)
    
    node_id = 5000
    times_hr = times_seconds / 3600.0
    
    plt.figure(figsize=(10, 5))
    plt.plot(times_hr, true_wl_matrix[:, node_id], 'k--', label='True Water Level (SRH-2D)', linewidth=2)
    plt.plot(times_hr, pred_wl_matrix[:, node_id], 'r-', label='Pure FVM Numerical Model', alpha=0.7, linewidth=2)
    plt.xlabel('Time (Hours)')
    plt.ylabel('Water Level (m)')
    plt.title(f'GPU Explicit FVM Solver - Water Level at Interior Node {node_id}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'/kaggle/working/outputs/node_{node_id}_fvm_numerical_comparison.png')
    
    print("Simulation complete! Plot saved to /kaggle/working/outputs")

if __name__ == "__main__":
    main()
