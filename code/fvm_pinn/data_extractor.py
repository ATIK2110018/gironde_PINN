import os
import netCDF4 as nc
import numpy as np
import torch
from scipy.spatial import cKDTree

def extract_fvm_geometry(nc_file_path, device='cpu'):
    print(f"Extracting FVM Geometry from {nc_file_path}...")
    dataset = nc.Dataset(nc_file_path, 'r')
    
    # In D-Flow FM, water levels are defined at "faces" (circumcenters)
    # So "faces" in D-Flow FM = FVM "cells"
    if 'mesh2d_face_x' in dataset.variables:
        cell_x = dataset.variables['mesh2d_face_x'][:]
        cell_y = dataset.variables['mesh2d_face_y'][:]
        cell_z = dataset.variables['mesh2d_face_z'][:] if 'mesh2d_face_z' in dataset.variables else np.zeros_like(cell_x)
        if 'mesh2d_edge_faces' in dataset.variables:
            edge_cells = dataset.variables['mesh2d_edge_faces'][:]
            edge_cells = np.ma.filled(edge_cells, -1)
        else:
            edge_cells = dataset.variables['mesh2d_edge_nodes'][:]
            edge_cells = np.ma.filled(edge_cells, -1)
    elif 'mesh2d_node_x' in dataset.variables:
        cell_x = dataset.variables['mesh2d_node_x'][:]
        cell_y = dataset.variables['mesh2d_node_y'][:]
        cell_z = dataset.variables['mesh2d_node_z'][:] if 'mesh2d_node_z' in dataset.variables else np.zeros_like(cell_x)
        edge_cells = dataset.variables['mesh2d_edge_nodes'][:]
        edge_cells = np.ma.filled(edge_cells, -1)
    else:
        cell_x = dataset.variables['NetNode_x'][:]
        cell_y = dataset.variables['NetNode_y'][:]
        cell_z = dataset.variables['NetNode_z'][:] if 'NetNode_z' in dataset.variables else np.zeros_like(cell_x)
        edge_cells = dataset.variables['NetLink'][:]
        edge_cells = np.ma.filled(edge_cells, -1)
        
    # Extract Area
    if 'mesh2d_face_area' in dataset.variables:
        cell_areas = dataset.variables['mesh2d_face_area'][:]
    else:
        print("Warning: mesh2d_face_area not found. Calculating exact polygon areas using Shoelace formula...")
        if 'mesh2d_face_nodes' in dataset.variables and 'mesh2d_node_x' in dataset.variables:
            face_nodes = dataset.variables['mesh2d_face_nodes'][:]
            node_x = dataset.variables['mesh2d_node_x'][:]
            node_y = dataset.variables['mesh2d_node_y'][:]
            cell_areas = np.zeros(len(cell_x))
            for c in range(len(cell_x)):
                nodes = face_nodes[c, :] if len(face_nodes.shape) > 1 else [face_nodes[c]]
                # Filter out masked values (-1, etc)
                valid_nodes = [int(n)-1 for n in nodes if not np.ma.is_masked(n) and int(n) > 0]
                if len(valid_nodes) >= 3:
                    px = node_x[valid_nodes]
                    py = node_y[valid_nodes]
                    cell_areas[c] = 0.5 * np.abs(np.dot(px, np.roll(py, -1)) - np.dot(py, np.roll(px, -1)))
                else:
                    cell_areas[c] = 1.0 # Fallback
        else:
            cell_areas = np.ones_like(cell_x)
        
    num_cells = len(cell_x)
    
    edges_list = []
    normals_list = []
    lengths_list = []
    
    for i in range(edge_cells.shape[0] if len(edge_cells.shape) == 2 else edge_cells.shape[1]):
        c1 = int(edge_cells[i, 0]) - 1 if len(edge_cells.shape) == 2 else int(edge_cells[0, i]) - 1
        c2 = int(edge_cells[i, 1]) - 1 if len(edge_cells.shape) == 2 else int(edge_cells[1, i]) - 1
        
        if c1 >= 0 and c2 >= 0:
            dx = cell_x[c2] - cell_x[c1]
            dy = cell_y[c2] - cell_y[c1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 0:
                nx = dx / dist
                ny = dy / dist
                
                edge_len = dist
                if 'mesh2d_edge_length' in dataset.variables:
                    edge_len = dataset.variables['mesh2d_edge_length'][i]
                
                edges_list.append([c1, c2])
                normals_list.append([nx, ny])
                lengths_list.append(edge_len)
                
                edges_list.append([c2, c1])
                normals_list.append([-nx, -ny])
                lengths_list.append(edge_len)
                
    dataset.close()
    
    # --- DIMENSIONAL SCALING ---
    # If the mesh is in Lat/Lon (degrees), we MUST scale areas and lengths to meters!
    # 1 degree is roughly 111,139 meters.
    if np.max(np.abs(cell_x)) <= 360.0:
        print("Detected Lat/Lon coordinate system. Scaling geometric Area and Length to METERS to balance SWE physics.")
        lat_rad = np.radians(np.mean(cell_y))
        deg_to_m_y = 111139.0
        deg_to_m_x = 111139.0 * np.cos(lat_rad)
        
        # Scale areas (Area = dx * dy)
        cell_areas = cell_areas * (deg_to_m_x * deg_to_m_y)
        
        # Scale lengths (rough approximation for lengths list)
        for idx in range(len(lengths_list)):
            lengths_list[idx] *= np.sqrt(deg_to_m_x * deg_to_m_y)  # Geometric mean scale
            
    # CRITICAL: Prevent Division by Zero from microscopic boundary sliver cells
    cell_areas = np.clip(cell_areas, a_min=10.0, a_max=None)
    
    cell_coords = torch.tensor(np.column_stack((cell_x, cell_y)), dtype=torch.float32, device=device)
    cell_z_t = torch.tensor(cell_z, dtype=torch.float32, device=device)
    cell_areas_t = torch.tensor(cell_areas, dtype=torch.float32, device=device).unsqueeze(1)
    
    edge_index = torch.tensor(edges_list, dtype=torch.long, device=device).t().contiguous()
    edge_normals = torch.tensor(normals_list, dtype=torch.float32, device=device)
    edge_lengths_t = torch.tensor(lengths_list, dtype=torch.float32, device=device).unsqueeze(1)
    
    print(f"Extracted {num_cells} FVM cells and {edge_index.size(1)//2} internal faces.")
    return cell_coords, cell_z_t, cell_areas_t, edge_index, edge_normals, edge_lengths_t

def load_friction_xyz(filepath, cell_coords, device='cpu'):
    # Loads Manning's n from frictioncoefficient.xyz
    from scipy.interpolate import griddata
    data = np.loadtxt(filepath)
    fric_np = griddata((data[:, 0], data[:, 1]), data[:, 2], (cell_coords[:, 0].cpu().numpy(), cell_coords[:, 1].cpu().numpy()), method='nearest')
    return torch.tensor(fric_np, dtype=torch.float32, device=device)

def get_boundary_cells(cell_coords, boundary_pli_path, threshold=0.01):
    # Extracts cell indices near the boundary pli lines
    with open(boundary_pli_path, 'r') as f:
        lines = f.readlines()
    coords = []
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) >= 2:
            try: coords.append([float(parts[0]), float(parts[1])])
            except ValueError: pass
    
    coords = np.array(coords)
    cell_coords_np = cell_coords.cpu().numpy()
    boundary_cells = []
    
    for i in range(len(coords)-1):
        p1, p2 = coords[i], coords[i+1]
        l2 = np.sum((p2 - p1)**2)
        if l2 == 0: continue
        t = np.clip(np.sum((cell_coords_np - p1) * (p2 - p1), axis=1) / l2, 0, 1)
        projection = p1 + t[:, np.newaxis] * (p2 - p1)
        dist = np.sqrt(np.sum((cell_coords_np - projection)**2, axis=1))
        boundary_cells.extend(np.where(dist < threshold)[0])
        
    return list(set(boundary_cells))
