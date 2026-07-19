import torch
from loss import compute_fvm_physics_loss, compute_data_loss
import numpy as np

def train_fvm_pinn(model, cell_coords, cell_z, cell_areas, edge_index, edge_normals, edge_lengths, cell_friction, train_times, train_wl, val_times, val_wl, epochs=5000, lr=1e-3, device='cpu'):
    """
    RIGOROUS PHYSICS-DRIVEN SOLVER (Hard PINN).
    The network is forced to solve the interior PDE purely using Physics (SWE).
    Data is ONLY used for Initial Conditions (t=0) and Boundary Conditions (edges).
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.5)
    
    x_np = cell_coords[:, 0].cpu().numpy()
    y_np = cell_coords[:, 1].cpu().numpy()
    
    all_times = np.concatenate((train_times, val_times))
    model.set_scales(x_np.min(), x_np.max(), y_np.min(), y_np.max(), all_times[0]*3600, all_times[-1]*3600)
    
    # --- IDENTIFY BOUNDARY CELLS FOR BOUNDARY CONDITIONS (BC) ---
    # We define the open boundaries as the cells at the geographical extremes of the river
    x_min, x_max = x_np.min(), x_np.max()
    y_min, y_max = y_np.min(), y_np.max()
    
    boundary_mask = (x_np < x_min + 0.05*(x_max-x_min)) | \
                    (x_np > x_max - 0.05*(x_max-x_min)) | \
                    (y_np < y_min + 0.05*(y_max-y_min)) | \
                    (y_np > y_max - 0.05*(y_max-y_min))
    
    bc_indices = torch.tensor(np.where(boundary_mask)[0], dtype=torch.long, device=device)
    bc_coords = cell_coords[bc_indices]
    
    print(f"Starting PURE PHYSICS Solver... (Interior solved entirely by SWE Physics)")
    print(f"Identified {len(bc_indices)} Boundary Condition cells.")
    
    # Physics is now the dominant driver!
    lambda_fvm = 1.0 
    lambda_ic = 10.0 # Initial condition weight
    lambda_bc = 10.0 # Boundary condition weight
    
    best_val_loss = float('inf')
    
    # Prepare Initial Condition (IC) Data
    ic_t_scalar = train_times[0] * 3600.0
    ic_wl = torch.tensor(train_wl[0], dtype=torch.float32, device=device)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # 1. INITIAL CONDITION LOSS (All cells, but ONLY at t = 0)
        loss_ic = compute_data_loss(model, cell_coords, ic_wl, None, None, ic_t_scalar)
        
        # 2. BOUNDARY CONDITION & PHYSICS LOSS (Over time)
        batch_size = 4
        t_idxs = np.random.choice(len(train_times), size=batch_size, replace=False)
        
        total_loss = 0
        total_bc = 0
        total_fvm = 0
        
        for t_idx in t_idxs:
            t_scalar = train_times[t_idx] * 3600.0
            
            # Boundary Condition Loss (ONLY on BC cells)
            true_bc_wl = torch.tensor(train_wl[t_idx][bc_indices.cpu().numpy()], dtype=torch.float32, device=device)
            loss_bc = compute_data_loss(model, bc_coords, true_bc_wl, None, None, t_scalar)
            
            # PDE Physics Loss (Governs the entire interior river domain!)
            t_batch = torch.full((cell_coords.size(0), 1), t_scalar, device=device, requires_grad=True)
            loss_fvm = compute_fvm_physics_loss(model, cell_coords, cell_z, cell_areas, edge_index, edge_normals, edge_lengths, cell_friction, t_batch)
            
            loss = (lambda_ic * loss_ic + lambda_bc * loss_bc + lambda_fvm * loss_fvm) / batch_size
            loss.backward(retain_graph=True)
            
            total_loss += loss.item() * batch_size
            total_bc += loss_bc.item()
            total_fvm += loss_fvm.item()
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        scheduler.step(total_loss)
        
        if epoch % 100 == 0:
            # Validation (Evaluating if the Physics accurately predicted the hidden data)
            model.eval()
            with torch.no_grad():
                val_idx = np.random.randint(0, len(val_times))
                val_t_scalar = val_times[val_idx] * 3600.0
                true_val_wl = torch.tensor(val_wl[val_idx], dtype=torch.float32, device=device)
                
                # Check global error against hidden data
                val_data_loss = compute_data_loss(model, cell_coords, true_val_wl, None, None, val_t_scalar)
            
            if val_data_loss.item() < best_val_loss:
                best_val_loss = val_data_loss.item()
                torch.save(model.state_dict(), "fvm_pinn_best.pth")
                
            print(f"Epoch {epoch:4d} | Tot: {total_loss:.4f} | IC: {loss_ic.item():.4f} | BC: {total_bc/batch_size:.4f} | FVM: {total_fvm/batch_size:.4f} | Val: {val_data_loss.item():.4f}")
            
    print("Training Complete. Best model saved as 'fvm_pinn_best.pth'")
