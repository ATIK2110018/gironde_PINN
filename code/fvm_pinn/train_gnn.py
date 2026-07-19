import torch
import torch.nn.functional as F
import numpy as np

def train_neural_fvm(model, cell_coords, cell_z, cell_areas, edge_index, edge_normals, edge_lengths, cell_friction, times_hr, true_wl_matrix, epochs=10000, lr=1e-3, device='cpu'):
    """
    Classical Coordinate PINN Trainer (HydroNet Style).
    We randomly sample (x, y, t) points from the dataset, enforce the data loss on Water Level,
    and use Autograd to enforce the Mass Continuity Equation to infer (u, v) velocities.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    
    print(f"Starting Classical PINN Training (Coordinate-based).")
    
    num_t = len(times_hr)
    num_cells = cell_coords.shape[0]
    
    # 1. Normalize coordinates to prevent exploding gradients in the MLP
    x_mean, x_std = cell_coords[:,0].mean(), cell_coords[:,0].std()
    y_mean, y_std = cell_coords[:,1].mean(), cell_coords[:,1].std()
    t_mean, t_std = times_hr.mean(), times_hr.std()
    
    model.x_mean, model.x_std = x_mean, x_std
    model.y_mean, model.y_std = y_mean, y_std
    model.t_mean, model.t_std = t_mean, t_std
    
    # 2. Pre-load all coordinates into tensors
    cell_x_all = torch.tensor(cell_coords[:,0], dtype=torch.float32, device=device).unsqueeze(1)
    cell_y_all = torch.tensor(cell_coords[:,1], dtype=torch.float32, device=device).unsqueeze(1)
    times_all = torch.tensor(times_hr, dtype=torch.float32, device=device).unsqueeze(1)
    cell_z_all = cell_z.to(device).unsqueeze(1)
    
    batch_size = 15000 # Massive batch size for dense point cloud training
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # 3. Randomly sample a batch of points in space-time
        batch_c = np.random.randint(0, num_cells, batch_size)
        batch_t = np.random.randint(0, num_t, batch_size)
        
        # Extract true water depth for this batch
        true_h_batch = torch.tensor(true_wl_matrix[batch_t, batch_c], dtype=torch.float32, device=device).unsqueeze(1)
        z_batch = cell_z_all[batch_c]
        true_depth_batch = torch.clamp(true_h_batch - z_batch, min=0.01)
        
        # Extract normalized coordinates
        x_norm = (cell_x_all[batch_c] - x_mean) / x_std
        y_norm = (cell_y_all[batch_c] - y_mean) / y_std
        t_norm = (times_all[batch_t] - t_mean) / t_std
        
        # We MUST set requires_grad=True to compute PDE residuals via Autograd
        x_norm.requires_grad_(True)
        y_norm.requires_grad_(True)
        t_norm.requires_grad_(True)
        
        # 4. Forward pass
        pred_depth, pred_u, pred_v = model(x_norm, y_norm, t_norm)
        
        # 5. DATA LOSS (Force model to match the exact water levels from the Teacher)
        loss_data = F.mse_loss(pred_depth, true_depth_batch)
        
        # 6. PHYSICS LOSS (Mass Continuity Equation)
        # We use PyTorch Autograd to compute exactly how the predicted fields change over space and time.
        # This infers the hidden (u, v) velocities!
        dh_dt_norm = torch.autograd.grad(pred_depth, t_norm, grad_outputs=torch.ones_like(pred_depth), create_graph=True)[0]
        dh_dx_norm = torch.autograd.grad(pred_depth, x_norm, grad_outputs=torch.ones_like(pred_depth), create_graph=True)[0]
        dh_dy_norm = torch.autograd.grad(pred_depth, y_norm, grad_outputs=torch.ones_like(pred_depth), create_graph=True)[0]
        
        du_dx_norm = torch.autograd.grad(pred_u, x_norm, grad_outputs=torch.ones_like(pred_u), create_graph=True)[0]
        dv_dy_norm = torch.autograd.grad(pred_v, y_norm, grad_outputs=torch.ones_like(pred_v), create_graph=True)[0]
        
        # Un-normalize gradients back to physical units (meters/second)
        dh_dt = dh_dt_norm / t_std / 3600.0 # Convert hours to seconds
        dh_dx = dh_dx_norm / x_std
        dh_dy = dh_dy_norm / y_std
        du_dx = du_dx_norm / x_std
        dv_dy = dv_dy_norm / y_std
        
        # Continuity Residual: dh/dt + d(hu)/dx + d(hv)/dy = 0
        # Expanded using product rule: dh/dt + h*du/dx + u*dh/dx + h*dv/dy + v*dh/dy = 0
        physics_residual = dh_dt + pred_depth*du_dx + pred_u*dh_dx + pred_depth*dv_dy + pred_v*dh_dy
        loss_physics = torch.mean(physics_residual**2)
        
        # Total Loss (Weighted to balance magnitudes)
        loss = loss_data + 1000.0 * loss_physics
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:5d} | Total Loss: {loss.item():.4f} | Data Loss: {loss_data.item():.4f} | Physics Residual: {loss_physics.item():.8f}")
            
    print("Solver Training Complete! Saving weights to 'neural_fvm_best.pth'")
    torch.save(model.state_dict(), "neural_fvm_best.pth")
