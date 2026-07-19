import torch
import torch.nn.functional as F
import numpy as np

def train_neural_fvm(model, cell_coords, cell_z, cell_areas, edge_index, edge_normals, edge_lengths, cell_friction, times_hr, true_wl_matrix, epochs=2000, lr=1e-3, device='cpu'):
    """
    Trains the Differentiable Simulator explicitly over time (Autoregressive).
    This acts EXACTLY like Delft3D.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5)
    
    # Calculate exact delta T from the dataset
    dt_seconds = (times_hr[1] - times_hr[0]) * 3600.0
    model.dt = float(dt_seconds)
    
    print(f"Starting Autoregressive Neural FVM Solver (dt = {model.dt} seconds)")
    
    # Identify Boundary Nodes to enforce Hard Boundary Conditions during the rollout!
    x_min, x_max = cell_coords[:,0].min(), cell_coords[:,0].max()
    y_min, y_max = cell_coords[:,1].min(), cell_coords[:,1].max()
    
    boundary_mask = (cell_coords[:,0] < x_min + 0.05*(x_max-x_min)) | \
                    (cell_coords[:,0] > x_max - 0.05*(x_max-x_min)) | \
                    (cell_coords[:,1] < y_min + 0.05*(y_max-y_min)) | \
                    (cell_coords[:,1] > y_max - 0.05*(y_max-y_min))
                    
    bc_indices = torch.tensor(np.where(boundary_mask)[0], dtype=torch.long, device=device)
    interior_indices = torch.tensor(np.where(~boundary_mask)[0], dtype=torch.long, device=device)
    
    print(f"Identified {len(bc_indices)} Boundary nodes and {len(interior_indices)} Interior nodes.")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # We unroll the simulation for a "window" of time to train the dynamics.
        # Training on 4 consecutive steps (e.g. 4 hours)
        rollout_steps = min(4, len(times_hr) - 1)
        start_idx = np.random.randint(0, len(times_hr) - rollout_steps)
        
        # 1. INITIAL CONDITION (Set perfectly from data)
        h_current = torch.tensor(true_wl_matrix[start_idx], dtype=torch.float32, device=device).unsqueeze(1) - cell_z.unsqueeze(1)
        h_current = torch.clamp(h_current, min=0.01)
        
        # We don't have true velocity data, so we let the network initialize it to 0. 
        # (The GNN will naturally spin up the velocity fields within the first step).
        u_current = torch.zeros_like(h_current)
        v_current = torch.zeros_like(h_current)
        
        total_loss = 0
        
        for step in range(rollout_steps):
            # 2. ENFORCE HARD BOUNDARY CONDITIONS (Exactly like a numerical solver)
            true_h_next = torch.tensor(true_wl_matrix[start_idx + step + 1], dtype=torch.float32, device=device).unsqueeze(1) - cell_z.unsqueeze(1)
            true_h_next = torch.clamp(true_h_next, min=0.01)
            
            # Physically overwrite the boundary nodes with the incoming tidal wave!
            h_current[bc_indices] = true_h_next[bc_indices]
            
            # 3. MATHEMATICALLY STEP FORWARD IN TIME (t -> t+1)
            h_next, u_next, v_next = model(h_current, u_current, v_current, cell_z, cell_friction, cell_areas, edge_index, edge_normals, edge_lengths)
            
            # 4. MEASURE SOLVER ACCURACY ON THE INTERIOR
            loss = F.mse_loss(h_next[interior_indices], true_h_next[interior_indices])
            total_loss += loss
            
            # Update state for next step
            h_current = h_next
            u_current = u_next
            v_current = v_next
            
        total_loss = total_loss / rollout_steps
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(total_loss)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:4d} | Rollout Loss: {total_loss.item():.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            
    print("Solver Training Complete! Saving weights to 'neural_fvm_best.pth'")
    torch.save(model.state_dict(), "neural_fvm_best.pth")
