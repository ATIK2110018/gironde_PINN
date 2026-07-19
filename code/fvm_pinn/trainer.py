import torch
from loss import compute_fvm_physics_loss, compute_data_loss
import numpy as np

def train_fvm_pinn(model, cell_coords, cell_z, cell_areas, edge_index, edge_normals, edge_lengths, cell_friction, train_times, train_wl, val_times, val_wl, epochs=5000, lr=1e-3, device='cpu'):
    """
    The training loop for exact FVM_PINN.
    Uses the "Teacher Strategy" recommended by HydroNet for complex rivers.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.5)
    
    # Scale network inputs based on the ENTIRE time domain (train + val)
    x_np = cell_coords[:, 0].cpu().numpy()
    y_np = cell_coords[:, 1].cpu().numpy()
    
    # We must scale t using the maximum validation time so the network knows the boundary
    all_times = np.concatenate((train_times, val_times))
    model.set_scales(x_np.min(), x_np.max(), y_np.min(), y_np.max(), all_times[0]*3600, all_times[-1]*3600)
    
    print("Starting exact FVM-PINN Training with Validation...")
    
    # Physics is heavily regularized, but guided by Teacher data to avoid flat water collapse
    # We use a very low lambda_fvm because raw SWE residuals are massive compared to MSE loss
    lambda_fvm = 0.001
    lambda_data = 100.0
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Gradient Accumulation: sample 4 random time steps to average out the chaos
        batch_size = 4
        t_idxs = np.random.choice(len(train_times), size=batch_size, replace=False)
        
        total_loss = 0
        total_data = 0
        total_fvm = 0
        
        for t_idx in t_idxs:
            t_scalar = train_times[t_idx] * 3600.0
            true_wl_at_t = torch.tensor(train_wl[t_idx], dtype=torch.float32, device=device)
            
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
        
        # Update scheduler based on averaged training loss
        scheduler.step(total_loss)
        
        if epoch % 100 == 0:
            # --- VALIDATION STEP ---
            model.eval()
            with torch.no_grad():
                val_idx = np.random.randint(0, len(val_times))
                val_t_scalar = val_times[val_idx] * 3600.0
                true_val_wl = torch.tensor(val_wl[val_idx], dtype=torch.float32, device=device)
                val_data_loss = compute_data_loss(model, cell_coords, true_val_wl, None, None, val_t_scalar)
            
            # Save the best model based on validation loss!
            if val_data_loss.item() < best_val_loss:
                best_val_loss = val_data_loss.item()
                torch.save(model.state_dict(), "fvm_pinn_best.pth")
                
            print(f"Epoch {epoch:4d} | Train Tot: {total_loss:.4f} | Train Data: {total_data/batch_size:.4f} | Train FVM: {total_fvm/batch_size:.6f} | Val Data: {val_data_loss.item():.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            
    print("Training Complete. Best model saved as 'fvm_pinn_best.pth'")
