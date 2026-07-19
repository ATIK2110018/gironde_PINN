import torch
import torch.nn.functional as F
from riemann_solver import roe_flux_2d

def compute_fvm_physics_loss(model, cell_coords, cell_z, cell_areas, edge_index, edge_normals, edge_lengths, cell_friction, t_batch, g=9.81):
    """
    Computes the exact Finite Volume physics loss over the mesh.
    Includes exact geometric flux scaling and Manning's bottom friction.
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
    
    # Multiply fluxes by edge lengths! (flux = flux_density * length)
    flux_mass = flux_mass * edge_lengths
    flux_mom_x = flux_mom_x * edge_lengths
    flux_mom_y = flux_mom_y * edge_lengths
    
    # 5. Sum fluxes into cells
    num_cells = cell_coords.size(0)
    net_flux_mass = torch.zeros((num_cells, 1), device=xi.device).scatter_add_(0, c_L.unsqueeze(1), flux_mass)
    net_flux_mom_x = torch.zeros((num_cells, 1), device=xi.device).scatter_add_(0, c_L.unsqueeze(1), flux_mom_x)
    net_flux_mom_y = torch.zeros((num_cells, 1), device=xi.device).scatter_add_(0, c_L.unsqueeze(1), flux_mom_y)
    
    # Divide by cell area to get divergence (div(F) = SUM(F * L) / A)
    net_flux_mass = net_flux_mass / cell_areas
    net_flux_mom_x = net_flux_mom_x / cell_areas
    net_flux_mom_y = net_flux_mom_y / cell_areas
    
    # 6. Bottom Friction (Manning's n formula: S_f = g * n^2 * U * |U| / h^(1/3))
    vel_mag = torch.sqrt(u**2 + v**2 + 1e-8)
    n_roughness = cell_friction.unsqueeze(1)
    # Clip n to avoid crazy spikes if friction map is messy
    n_roughness = torch.clamp(n_roughness, 0.01, 0.1) 
    
    fric_x = g * (n_roughness**2) * u * vel_mag / (h**(1/3) + 1e-8)
    fric_y = g * (n_roughness**2) * v * vel_mag / (h**(1/3) + 1e-8)
    
    # 7. Exact SWE Residuals (dQ/dt + div(F) + Source = 0)
    res_mass = dxi_dt + net_flux_mass
    res_mom_x = dhu_dt + net_flux_mom_x + fric_x
    res_mom_y = dhv_dt + net_flux_mom_y + fric_y
    
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
