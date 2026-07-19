import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def get_cells_near_line_dynamic(cell_coords_m, cell_areas, p1_deg, p2_deg):
    """
    Dynamically identifies cells intersecting the boundary line without ANY assumed distance thresholds.
    It calculates the exact distance in meters and compares it to the cell's specific local radius 
    derived from its exact area, perfectly adapting to non-uniform unstructured meshes.
    """
    # Convert degrees to meters to match cell_coords_m projection
    p1_m = p1_deg * np.array([78700.0, 111000.0])
    p2_m = p2_deg * np.array([78700.0, 111000.0])
    
    l2 = np.sum((p2_m - p1_m)**2)
    if l2 == 0: return np.zeros(cell_coords_m.shape[0], dtype=bool)
    
    t = np.sum((cell_coords_m - p1_m) * (p2_m - p1_m), axis=1) / l2
    t = np.clip(t, 0.0, 1.0)
    
    projection = p1_m + t[:, np.newaxis] * (p2_m - p1_m)
    dist_m = np.sqrt(np.sum((cell_coords_m - projection)**2, axis=1))
    
    # The cell's local width is approx sqrt(Area). 
    # If the distance is less than half the width (plus a tiny 10% safety margin), it strictly intersects.
    local_threshold_m = np.sqrt(cell_areas.flatten()) * 0.6
    
    return dist_m < local_threshold_m

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}")
    
    from data_extractor import extract_fvm_geometry
    print("Extracting FVM Geometry from /kaggle/input/datasets/atikurr/gironde-hydro-out/FlowFM_map.nc...")
    cell_coords_t, cell_z, cell_areas, edge_index, edge_normals, edge_lengths, topo_boundary_mask = extract_fvm_geometry("/kaggle/input/datasets/atikurr/gironde-hydro-out/FlowFM_map.nc", device=device)
    cell_coords = cell_coords_t.cpu().numpy()
    
    import netCDF4 as nc
    ds = nc.Dataset("/kaggle/input/datasets/atikurr/gironde-hydro-out/FlowFM_map.nc")
    
    # Strictly extract as dense numpy array and kill any masks
    raw_wl = ds.variables['mesh2d_s1'][:]
    if hasattr(raw_wl, 'filled'):
        raw_wl = raw_wl.filled(np.nan)
    true_wl_matrix = np.array(raw_wl, dtype=np.float32)
    
    # Replace NaNs (dry cells at low tide) with bed elevation so depth becomes 0
    cell_z_np = cell_z.cpu().numpy().flatten()
    invalid_mask = np.isnan(true_wl_matrix) | (true_wl_matrix < -900)
    true_wl_matrix[invalid_mask] = np.broadcast_to(cell_z_np, true_wl_matrix.shape)[invalid_mask]
    
    times_seconds = ds.variables['time'][:]
    ds.close()
    
    # Convert cell_coords to meters FIRST for the dynamic spatial extraction
    x_coords_m = cell_coords[:, 0] * 78700.0  # Approx longitude scaling at 45 deg N
    y_coords_m = cell_coords[:, 1] * 111000.0
    cell_coords_m = np.column_stack((x_coords_m, y_coords_m))
    cell_areas_np = cell_areas.squeeze(1).cpu().numpy()
    
    # EXACT BOUNDARIES from the .pli files intersected with true topological boundary cells!
    # No more cell size assumptions! The model dynamically adapts to non-uniform mesh sizes!
    # 1. Ocean Boundary (Port Block)
    p1_port = np.array([-1.055107109535667E+000, 4.558144911918696E+001])
    p2_port = np.array([-1.043691864509240E+000, 4.559334500610923E+001])
    port_mask = get_cells_near_line_dynamic(cell_coords_m, cell_areas_np, p1_port, p2_port) & topo_boundary_mask
    
    # 2. Garonne River Inflow
    p1_gar = np.array([-5.308167329151710E-001, 4.480884916128741E+001])
    p2_gar = np.array([-5.262550852925010E-001, 4.481051805675912E+001])
    gar_mask = get_cells_near_line_dynamic(cell_coords_m, cell_areas_np, p1_gar, p2_gar) & topo_boundary_mask
    
    # 3. Dordogne River Inflow
    p1_dor = np.array([-2.586704969143130E-001, 4.491934439849670E+001])
    p2_dor = np.array([-2.586418807368147E-001, 4.491740422166230E+001])
    dor_mask = get_cells_near_line_dynamic(cell_coords_m, cell_areas_np, p1_dor, p2_dor) & topo_boundary_mask
    
    exact_boundary_mask = port_mask | gar_mask | dor_mask
    boundary_mask_t = torch.tensor(exact_boundary_mask, device=device)
    
    print(f"Identified EXACT boundary cells: {np.sum(port_mask)} Ocean, {np.sum(gar_mask)} Garonne, {np.sum(dor_mask)} Dordogne.")
    
    initial_wl = true_wl_matrix[0, :]
    
    # Extract exact boundary forcing for EACH boundary cell individually! (Shape: 265, K)
    boundary_wl_matrix = true_wl_matrix[:, exact_boundary_mask]
    
    from numerical_model import GPUHydrodynamicModel
    
    model = GPUHydrodynamicModel(
        cell_coords=cell_coords_m,
        cell_areas=cell_areas_np,
        cell_z=cell_z.cpu().numpy(),
        edge_index=edge_index.cpu().numpy(),
        edge_normals=edge_normals.cpu().numpy(),
        edge_lengths=edge_lengths.squeeze(1).cpu().numpy(),
        boundary_mask=boundary_mask_t,
        device=device
    )
    
    pred_wl_matrix = model.simulate(
        initial_wl=initial_wl,
        boundary_wl_matrix=boundary_wl_matrix,
        times_seconds=times_seconds
    )
    
    # ==========================================
    # Plotting
    # ==========================================
    os.makedirs('/kaggle/working/outputs', exist_ok=True)
    
    nodes_to_plot = [100, 1000, 5000, 15000, 25000, 35000]
    times_hr = times_seconds / 3600.0
    
    plt.figure(figsize=(16, 12))
    for idx, node_id in enumerate(nodes_to_plot):
        plt.subplot(3, 2, idx + 1)
        plt.plot(times_hr, true_wl_matrix[:, node_id], 'k--', label='True Water Level (SRH-2D)', linewidth=2)
        plt.plot(times_hr, pred_wl_matrix[:, node_id], 'r-', label='Pure FVM Numerical Model', alpha=0.7, linewidth=2)
        plt.xlabel('Time (Hours)')
        plt.ylabel('Water Level (m)')
        plt.title(f'Water Level at Interior Node {node_id}')
        plt.legend()
        plt.grid(True)
        
    plt.tight_layout()
    plt.savefig('/kaggle/working/outputs/multi_node_fvm_comparison.png')
    
    print("Simulation complete! Plot saved to /kaggle/working/outputs")

if __name__ == "__main__":
    main()
