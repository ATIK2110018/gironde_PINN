import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import griddata

def plot_water_level_map(cell_coords, water_levels, time_hr, output_path):
    """
    Creates a 2D scatter map of the water level over the entire Gironde mesh.
    """
    plt.figure(figsize=(10, 8))
    x = cell_coords[:, 0].cpu().numpy()
    y = cell_coords[:, 1].cpu().numpy()
    wl = water_levels.cpu().numpy()
    
    sc = plt.scatter(x, y, c=wl, cmap='jet', s=5, vmin=-3.0, vmax=3.0)
    plt.colorbar(sc, label='Water Level (m)')
    plt.title(f'Gironde Water Level Map at t={time_hr:.2f} hours')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_timeseries_comparison(times_hr, pred_wl, true_wl, node_idx, output_path):
    """
    Plots the predicted vs true water level over time for a specific cell.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(times_hr, true_wl, 'k-', label='True Water Level (Delft3D)', linewidth=2)
    plt.plot(times_hr, pred_wl, 'r--', label='Predicted Water Level (FVM-PINN)', linewidth=2)
    
    plt.title(f'Water Level Timeseries (Cell {node_idx})')
    plt.xlabel('Time (hours)')
    plt.ylabel('Water Level (m)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

def generate_water_level_gif(cell_coords, true_times, true_eta, model, t_start, t_end, dt, output_path, device='cpu'):
    """
    Creates an animated GIF of the water wave propagating through the estuary.
    Must be called after training to see the results.
    """
    print(f"Generating GIF animation from {t_start/3600:.1f}h to {t_end/3600:.1f}h...")
    
    times = np.arange(t_start, t_end + dt, dt)
    x = cell_coords[:, 0:1].to(device)
    y = cell_coords[:, 1:2].to(device)
    
    # Pre-compute frames
    frames_data = []
    model.eval()
    with torch.no_grad():
        for t in times:
            t_tensor = torch.full_like(x, t, device=device)
            xi, _, _ = model(x, y, t_tensor)
            frames_data.append(xi.cpu().numpy().flatten())
            
    fig, ax = plt.subplots(figsize=(10, 8))
    x_np = x.cpu().numpy().flatten()
    y_np = y.cpu().numpy().flatten()
    
    sc = ax.scatter(x_np, y_np, c=frames_data[0], cmap='jet', s=5, vmin=-3.0, vmax=3.0)
    cbar = plt.colorbar(sc, ax=ax, label='Water Level (m)')
    title = ax.set_title(f'Time: {times[0]/3600:.2f} hours')
    
    def update(frame_idx):
        sc.set_array(frames_data[frame_idx])
        title.set_text(f'Time: {times[frame_idx]/3600:.2f} hours')
        return sc, title
        
    ani = animation.FuncAnimation(fig, update, frames=len(times), blit=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ani.save(output_path, writer='pillow', fps=10)
    plt.close()
    print(f"Saved animation to {output_path}")
