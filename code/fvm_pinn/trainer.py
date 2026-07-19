import torch
from loss import compute_fvm_physics_loss, compute_data_loss
import numpy as np

def train_fvm_pinn(model, cell_coords, cell_z, cell_areas, edge_index, edge_normals, edge_lengths, cell_friction, times_hr, true_wl_matrix, epochs=5000, lr=1e-3, device='cpu'):
    """
    The training loop for exact FVM_PINN.
    Uses the "Teacher Strategy" recommended by HydroNet for complex rivers.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.5)
    
    # Scale network inputs
    x_np = cell_coords[:, 0].cpu().numpy()
    y_np = cell_coords[:, 1].cpu().numpy()
    model.set_scales(x_np.min(), x_np.max(), y_np.min(), y_np.max(), times_hr[0]*3600, times_hr[-1]*3600)
    
    print("Starting exact FVM-PINN Training (Teacher Strategy)...")
    
    # Physics is heavily regularized, but guided by Teacher data to avoid flat water collapse
    lambda_fvm = 1.0
    lambda_data = 10.0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # 1. Randomly sample a time step to train on this batch
        t_idx = np.random.randint(0, len(times_hr))
        t_scalar = times_hr[t_idx] * 3600.0
        
        # 2. Compute Teacher Data Loss (MSE against Delft3D)
        true_wl_at_t = torch.tensor(true_wl_matrix[t_idx], dtype=torch.float32, device=device)
        loss_data = compute_data_loss(model, cell_coords, true_wl_at_t, None, None, t_scalar)
        
        # 3. Compute Exact Physics Loss
        t_batch = torch.full((cell_coords.size(0), 1), t_scalar, device=device, requires_grad=True)
        loss_fvm = compute_fvm_physics_loss(model, cell_coords, cell_z, cell_areas, edge_index, edge_normals, edge_lengths, cell_friction, t_batch)
        
        # 4. Total Loss
        loss = lambda_data * loss_data + lambda_fvm * loss_fvm
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        scheduler.step(loss)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Total: {loss.item():.4f} | Data: {loss_data.item():.4f} | FVM: {loss_fvm.item():.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            
    print("Training Complete. Saving model...")
    torch.save(model.state_dict(), "fvm_pinn_best.pth")
