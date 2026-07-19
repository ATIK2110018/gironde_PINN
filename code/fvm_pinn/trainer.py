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
    # We use a very low lambda_fvm because raw SWE residuals are massive compared to MSE loss
    lambda_fvm = 0.001
    lambda_data = 100.0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Gradient Accumulation: sample 4 random time steps to average out the chaos
        batch_size = 4
        t_idxs = np.random.choice(len(times_hr), size=batch_size, replace=False)
        
        total_loss = 0
        total_data = 0
        total_fvm = 0
        
        for t_idx in t_idxs:
            t_scalar = times_hr[t_idx] * 3600.0
            true_wl_at_t = torch.tensor(true_wl_matrix[t_idx], dtype=torch.float32, device=device)
            
            loss_data = compute_data_loss(model, cell_coords, true_wl_at_t, None, None, t_scalar)
            
            t_batch = torch.full((cell_coords.size(0), 1), t_scalar, device=device, requires_grad=True)
            loss_fvm = compute_fvm_physics_loss(model, cell_coords, cell_z, cell_areas, edge_index, edge_normals, edge_lengths, cell_friction, t_batch)
            
            loss = (lambda_data * loss_data + lambda_fvm * loss_fvm) / batch_size
            loss.backward()
            
            total_loss += loss.item() * batch_size
            total_data += loss_data.item()
            total_fvm += loss_fvm.item()
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Update scheduler based on averaged loss
        scheduler.step(total_loss)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Total: {total_loss:.4f} | Data: {total_data/batch_size:.4f} | FVM: {total_fvm/batch_size:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            
    print("Training Complete. Saving model...")
    torch.save(model.state_dict(), "fvm_pinn_best.pth")
