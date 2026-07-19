import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from data_extractor import extract_fvm_geometry
from numerical_model import GPUHydrodynamicModel
from fvm_pinn_model import FVMPINNTrainer

def get_cells_near_line_dynamic(cell_coords_m, cell_areas, p1_deg, p2_deg):
    p1_m = p1_deg * np.array([78700.0, 111000.0])
    p2_m = p2_deg * np.array([78700.0, 111000.0])
    
    l2 = np.sum((p2_m - p1_m)**2)
    if l2 == 0: return np.zeros(cell_coords_m.shape[0], dtype=bool)
    
    t = np.sum((cell_coords_m - p1_m) * (p2_m - p1_m), axis=1) / l2
    t = np.clip(t, 0.0, 1.0)
    
    projection = p1_m + t[:, np.newaxis] * (p2_m - p1_m)
    dist_m = np.sqrt(np.sum((cell_coords_m - projection)**2, axis=1))
    
    local_threshold_m = np.sqrt(cell_areas.flatten()) * 2.0
    return dist_m < local_threshold_m

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Starting FVM-PINN Training on {device}")
    
    print("Extracting FVM Geometry...")
    cell_coords_t, cell_z, cell_areas, edge_index, edge_normals, edge_lengths, topo_boundary_mask = extract_fvm_geometry("/kaggle/input/datasets/atikurr/gironde-hydro-out/FlowFM_map.nc", device=device)
    cell_coords = cell_coords_t.cpu().numpy()
    
    import netCDF4 as nc
    ds = nc.Dataset("/kaggle/input/datasets/atikurr/gironde-hydro-out/FlowFM_map.nc")
    
    raw_wl = ds.variables['mesh2d_s1'][:]
    if hasattr(raw_wl, 'filled'):
        raw_wl = raw_wl.filled(np.nan)
    true_wl_matrix = np.array(raw_wl, dtype=np.float32)
    
    cell_z_np = cell_z.cpu().numpy().flatten()
    invalid_mask = np.isnan(true_wl_matrix) | (true_wl_matrix < -900)
    true_wl_matrix[invalid_mask] = np.broadcast_to(cell_z_np, true_wl_matrix.shape)[invalid_mask]
    
    times_seconds = ds.variables['time'][:]
    ds.close()
    
    x_coords_m = cell_coords[:, 0] * 78700.0
    y_coords_m = cell_coords[:, 1] * 111000.0
    cell_coords_m = np.column_stack((x_coords_m, y_coords_m))
    cell_areas_np = cell_areas.squeeze(1).cpu().numpy()
    
    p1_port = np.array([-1.055107109535667E+000, 4.558144911918696E+001])
    p2_port = np.array([-1.043691864509240E+000, 4.559334500610923E+001])
    port_mask = get_cells_near_line_dynamic(cell_coords_m, cell_areas_np, p1_port, p2_port) & topo_boundary_mask
    
    p1_gar = np.array([-5.308167329151710E-001, 4.480884916128741E+001])
    p2_gar = np.array([-5.262550852925010E-001, 4.481051805675912E+001])
    gar_mask = get_cells_near_line_dynamic(cell_coords_m, cell_areas_np, p1_gar, p2_gar) & topo_boundary_mask
    
    p1_dor = np.array([-2.586704969143130E-001, 4.491934439849670E+001])
    p2_dor = np.array([-2.586418807368147E-001, 4.491740422166230E+001])
    dor_mask = get_cells_near_line_dynamic(cell_coords_m, cell_areas_np, p1_dor, p2_dor) & topo_boundary_mask
    
    exact_boundary_mask = port_mask | gar_mask | dor_mask
    boundary_mask_t = torch.tensor(exact_boundary_mask, device=device)
    
    # 1. Instantiate the Differentiable FVM Physics Engine
    fvm_model = GPUHydrodynamicModel(
        cell_coords=cell_coords_m,
        cell_areas=cell_areas_np,
        cell_z=cell_z.cpu().numpy(),
        edge_index=edge_index.cpu().numpy(),
        edge_normals=edge_normals.cpu().numpy(),
        edge_lengths=edge_lengths.squeeze(1).cpu().numpy(),
        boundary_mask=boundary_mask_t,
        device=device
    )
    
    # 2. Instantiate the FVM-PINN Trainer
    trainer = FVMPINNTrainer(
        fvm_model=fvm_model,
        cell_coords_m=torch.tensor(cell_coords_m, dtype=torch.float32, device=device),
        true_wl_matrix=true_wl_matrix,
        times_seconds=times_seconds,
        boundary_mask=boundary_mask_t
    )
    
    epochs = 100
    loss_history_data = []
    loss_history_phys = []
    
    print(f"Starting Training for {epochs} Epochs...")
    
    for epoch in range(epochs):
        epoch_data_loss = 0.0
        epoch_phys_loss = 0.0
        
        # Randomly sample 10 time steps per epoch for stochastic gradient descent
        sampled_t_indices = np.random.choice(range(len(times_seconds)-1), size=10, replace=False)
        
        for t_idx in sampled_t_indices:
            d_loss, p_loss = trainer.train_step(t_idx)
            epoch_data_loss += d_loss
            epoch_phys_loss += p_loss
            
        epoch_data_loss /= 10
        epoch_phys_loss /= 10
        
        loss_history_data.append(epoch_data_loss)
        loss_history_phys.append(epoch_phys_loss)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{epochs} | Data Loss: {epoch_data_loss:.4f} | FVM Physics Loss: {epoch_phys_loss:.4f}")
            
    # Save the trained model
    os.makedirs('/kaggle/working/outputs', exist_ok=True)
    torch.save(trainer.pinn.state_dict(), '/kaggle/working/outputs/fvm_pinn_model.pth')
    
    # Plot Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history_data, label='Data Loss')
    plt.plot(loss_history_phys, label='FVM Physics Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('FVM-PINN Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('/kaggle/working/outputs/fvm_pinn_loss.png')
    
    print("Training Complete! Model and Loss plot saved to /kaggle/working/outputs")

if __name__ == "__main__":
    main()
