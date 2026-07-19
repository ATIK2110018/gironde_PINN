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
    cell_x = dataset.variables['mesh2d_face_x'][:]
    cell_y = dataset.variables['mesh2d_face_y'][:]
    cell_z = dataset.variables['mesh2d_face_z'][:] if 'mesh2d_face_z' in dataset.variables else np.zeros_like(cell_x)
    
    # We need the links between cells to compute fluxes
    # "edge_faces" tells us which two cells share an edge
    if 'mesh2d_edge_faces' in dataset.variables:
        edge_cells = dataset.variables['mesh2d_edge_faces'][:]
    else:
        # Fallback if standard UGRID isn't perfectly followed
        print("Warning: mesh2d_edge_faces not found, falling back to node-based approximation.")
        cell_x = dataset.variables['mesh2d_node_x'][:]
        cell_y = dataset.variables['mesh2d_node_y'][:]
        cell_z = np.zeros_like(cell_x)
        edge_cells = dataset.variables['mesh2d_edge_nodes'][:]
        
    num_cells = len(cell_x)
    
    edges_list = []
    normals_list = []
    
    for i in range(edge_cells.shape[0] if len(edge_cells.shape) == 2 else edge_cells.shape[1]):
        c1 = int(edge_cells[i, 0]) - 1 if len(edge_cells.shape) == 2 else int(edge_cells[0, i]) - 1
        c2 = int(edge_cells[i, 1]) - 1 if len(edge_cells.shape) == 2 else int(edge_cells[1, i]) - 1
        
        if c1 >= 0 and c2 >= 0:
            dx = cell_x[c2] - cell_x[c1]
            dy = cell_y[c2] - cell_y[c1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 0:
                # The normal vector points from c1 to c2
                nx = dx / dist
                ny = dy / dist
                edges_list.append([c1, c2])
                normals_list.append([nx, ny])
                
                # Opposite direction
                edges_list.append([c2, c1])
                normals_list.append([-nx, -ny])
                
    dataset.close()
    
    cell_coords = torch.tensor(np.column_stack((cell_x, cell_y)), dtype=torch.float32, device=device)
    cell_z_t = torch.tensor(cell_z, dtype=torch.float32, device=device)
    edge_index = torch.tensor(edges_list, dtype=torch.long, device=device).t().contiguous()
    edge_normals = torch.tensor(normals_list, dtype=torch.float32, device=device)
    
    print(f"Extracted {num_cells} FVM cells and {edge_index.size(1)//2} internal faces.")
    return cell_coords, cell_z_t, edge_index, edge_normals

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
