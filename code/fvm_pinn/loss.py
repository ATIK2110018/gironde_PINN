import torch
import torch.nn.functional as F
from .riemann_solver import roe_flux_2d

def compute_fvm_physics_loss(model, cell_coords, cell_z, edge_index, edge_normals, t_batch, g=9.81):
    """
    Computes the Finite Volume physics loss over the mesh.
    Because FVM_PINN evaluates continuous time and discrete space, we use autograd 
    for temporal derivatives and the Roe solver for spatial fluxes.
    """
    # 1. Enable autograd tracking for the temporal derivative
    t_batch.requires_grad_(True)
    
    # 2. Predict the state across all cells at this time
    x = cell_coords[:, 0:1]
    y = cell_coords[:, 1:2]
    
    # Model outputs xi, hu, hv
    xi, hu, hv = model(x, y, t_batch)
    
    h = xi - cell_z.unsqueeze(1) # xi is wse (eta), h is total depth
    h = torch.clamp(h, min=1e-3)
    u = hu / h
    v = hv / h
    
    # 3. Compute continuous temporal derivatives dQ/dt using PyTorch Autograd
    dxi_dt = torch.autograd.grad(xi, t_batch, grad_outputs=torch.ones_like(xi), create_graph=True)[0]
    dhu_dt = torch.autograd.grad(hu, t_batch, grad_outputs=torch.ones_like(hu), create_graph=True)[0]
    dhv_dt = torch.autograd.grad(hv, t_batch, grad_outputs=torch.ones_like(hv), create_graph=True)[0]
    
    # 4. Compute the discrete spatial fluxes using the Roe Riemann solver
    c_L = edge_index[0, :]
    c_R = edge_index[1, :]
    nx = edge_normals[:, 0:1]
    ny = edge_normals[:, 1:2]
    
    h_L, h_R = h[c_L], h[c_R]
    u_L, u_R = u[c_L], u[c_R]
    v_L, v_R = v[c_L], v[c_R]
    zb_L, zb_R = cell_z[c_L].unsqueeze(1), cell_z[c_R].unsqueeze(1)
    
    h_still_L = -zb_L # Assuming 0 sea level for still water reference
    h_still_R = -zb_R
    
    flux_mass, flux_mom_x, flux_mom_y = roe_flux_2d(
        h_L, h_R, u_L, u_R, v_L, v_R, 
        h_still_L, h_still_R, nx, ny, g=g
    )
    
    # 5. Sum fluxes into cells
    num_cells = cell_coords.size(0)
    net_flux_mass = torch.zeros((num_cells, 1), device=xi.device).scatter_add_(0, c_L.unsqueeze(1), flux_mass)
    net_flux_mom_x = torch.zeros((num_cells, 1), device=xi.device).scatter_add_(0, c_L.unsqueeze(1), flux_mom_x)
    net_flux_mom_y = torch.zeros((num_cells, 1), device=xi.device).scatter_add_(0, c_L.unsqueeze(1), flux_mom_y)
    
    # Note: In a true 2D FVM, fluxes are multiplied by edge_length and divided by cell_area.
    # To keep the neural network loss stable and simple in Kaggle without true geometric areas, 
    # we treat it as an unscaled residual. The network absorbs the area scale.
    
    # 6. SWE Residuals
    res_mass = dxi_dt + net_flux_mass
    res_mom_x = dhu_dt + net_flux_mom_x
    res_mom_y = dhv_dt + net_flux_mom_y
    
    loss_phys = torch.mean(res_mass**2) + torch.mean(res_mom_x**2) + torch.mean(res_mom_y**2)
    return loss_phys

def compute_data_loss(model, cell_coords, true_wl, true_u, true_v, t_scalar):
    """
    Teacher forcing loss / Boundary Condition loss.
    """
    x = cell_coords[:, 0:1]
    y = cell_coords[:, 1:2]
    t_tensor = torch.full_like(x, t_scalar)
    
    xi, hu, hv = model(x, y, t_tensor)
    
    loss_data = F.mse_loss(xi, true_wl.unsqueeze(1))
    
    # We don't strictly penalize velocity to let physics handle it if true velocities are noisy
    return loss_data
