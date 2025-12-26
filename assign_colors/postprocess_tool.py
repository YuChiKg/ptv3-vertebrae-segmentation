import trimesh
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import pandas as pd
import json
import open3d as o3d
from pysdf import SDF
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans




def get_sj_folders(base_path):
    """Get all SJ0000xxx folders from the base directory"""
    return [f for f in sorted(os.listdir(base_path)) 
            if f.startswith('SJ0000') and os.path.isdir(os.path.join(base_path, f))]

def get_dates_in_folder(sj_folder_path):
    """Get all date folders within an SJ folder"""
    return [d for d in sorted(os.listdir(sj_folder_path)) 
            if os.path.isdir(os.path.join(sj_folder_path, d)) and 
            len(d) == 10 and d[4] == '-' and d[7] == '-']  # YYYY-MM-DD format check

def get_JSON_files(base_path):
    """
    Get all .json files organized by SJ folder and date
    Returns a dictionary with structure: {SJ_folder: {date: json_file_path}}
    """
    sj_folders = get_sj_folders(base_path)
    JSON_files = {}
    
    for sj_folder in sj_folders:
        sj_path = os.path.join(base_path, sj_folder)
        dates = get_dates_in_folder(sj_path)
        
        if not dates:
            continue
            
        JSON_files[sj_folder] = {}
        
        for date in sorted(dates):
            lis_JSON_file = os.path.join(sj_path, date, f"LIS_{sj_folder}.json")
            JSON_file = os.path.join(sj_path, date, f"{sj_folder}.json")
            
            if os.path.exists(lis_JSON_file):
                JSON_files[sj_folder][date] = lis_JSON_file
            elif os.path.exists(JSON_file):
                JSON_files[sj_folder][date] = JSON_file
            else:
                print(f"Warning: .json file not found at {JSON_file}")
    
    return JSON_files

def get_ply_files(base_path):
    """
    Get all .ply files organized by SJ folder and date
    Returns a dictionary with structure: {SJ_folder: {date: json_file_path}}
    each ply file is named as {sj}_{data}.ply (located: /home/travail/Antonin_Dataset/Point_Cloud_ply_symlink/{sj}/{date}/{sj}_{date}.ply)
    """
    sj_folders = get_sj_folders(base_path)
    ply_files = {}
    
    for sj_folder in sj_folders:
        sj_path = os.path.join(base_path, sj_folder)
        dates = get_dates_in_folder(sj_path)
        
        if not dates:
            continue
            
        ply_files[sj_folder] = {}
        
        for date in dates:
            ply_file = os.path.join(sj_path, date, f"{sj_folder}.ply")
            
            if os.path.exists(ply_file):
                ply_files[sj_folder][date] = ply_file
            else:
                print(f"Warning: .ply file not found at {ply_file}")
    
    return ply_files

### Function for processing o3 files to JSON files
# def preview_o3_file(file_path, lines_to_show=29):
#     """Preview the first few lines of an o3 file"""
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             content = f.readlines()
#         print(f"\nPreview of {file_path}:")
#         for line in content[:lines_to_show]: 
#             print(line.strip())
#     except Exception as e:
#         print(f"Error previewing file {file_path}: {str(e)}")

# def parse_o3_file(file_path):
#     """
#     Function to process o3 file and return structured data
#     """
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"File not found: {file_path}")
    
#     with open(file_path, 'r', encoding='utf-8') as f:
#         data = f.read()
    
#     # Extract objects section
#     objects_section = data.split("DATA 3D")[1].strip()
#     objects_dict = {}

#     # Process each object
#     object_blocks = objects_section.split("Objet:")
#     for block in object_blocks[1:]:  
#         lines = block.strip().splitlines()
#         if not lines:
#             continue
            
#         name = lines[0].strip() 
#         points = []
        
#         for line in lines[2:]:
#             parts = line.split()
#             if len(parts) == 5:
#                 try:
#                     tag, x, y, z, err = parts
#                     points.append({
#                         "tag": tag,
#                         "x": float(x),
#                         "y": float(y),
#                         "z": float(z),
#                         "err": float(err)
#                     })
#                 except ValueError:
#                     continue
        
#         if name and points:  # Only add if we have valid data
#             objects_dict[name] = {"Points": points}

#     structured_data = {
#         "File": os.path.basename(file_path),
#         "Path": file_path,
#         "Objets": objects_dict
#     }
    
#     return structured_data

# def save_json(data, output_path):
#     """Save structured data as JSON file"""
#     try:
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump(data, f, indent=4, ensure_ascii=False)
#         print(f"JSON file saved at: {output_path}")
#         return True
#     except Exception as e:
#         print(f"Error saving JSON file {output_path}: {str(e)}")
#         return False

# def process_all_files(base_path, preview_lines=0):
#     """
#     Process all .o3 files in all date folders of all SJ folders
#     :param base_path: Root directory containing SJ folders
#     :param preview_lines: Number of lines to preview (0 to disable)
#     """
#     sj_folders = get_sj_folders(base_path)
    
#     if not sj_folders:
#         print("No SJ0000xxx folders found in the directory.")
#         return
    
#     for sj_folder in sorted(sj_folders):
#         sj_path = os.path.join(base_path, sj_folder)
#         dates = get_dates_in_folder(sj_path)
        
#         if not dates:
#             print(f"No date folders found in {sj_folder}")
#             continue
            
#         for date in sorted(dates):
#             o3_file = os.path.join(sj_path, date, f"{sj_folder}.o3")
            
#             if not os.path.exists(o3_file):
#                 print(f"File not found: {o3_file}")
#                 continue
                
#             print(f"\nProcessing: {o3_file}")
            
#             # Preview file if requested
#             if preview_lines > 0:
#                 preview_o3_file(o3_file, preview_lines)
            
#             try:
#                 # Parse the o3 file
#                 structured_data = parse_o3_file(o3_file)
                
#                 # Save as JSON
#                 output_path = o3_file.replace(".o3", ".json")
#                 if save_json(structured_data, output_path):
#                     print(f"Successfully processed {o3_file}")
#                 else:
#                     print(f"Failed to save JSON for {o3_file}")
                    
#             except Exception as e:
#                 print(f"Error processing {o3_file}: {str(e)}")

# base_path = '/home/travail/Antonin_Dataset/o3_symlink'
# process_all_files(base_path, preview_lines=0)  
### The above code are the function for processing o3 files to JSON files and saved in same directory .JSON


# function to convert STL -> mesh (vertices, faces)
def mesh_from_stl(stl_file):
    # Load the STL file 
    mesh = trimesh.load_mesh(stl_file)
    # Extract vertices and faces
    vertices = mesh.vertices
    faces = mesh.faces
    
    return vertices, faces


# Compute the surface's plane equation
# def compute_plane_equation(A, B, C):
#     """
#     Compute the plane equation coefficients (a, b, c, d) given three points.
#     the points will extract from certain face to following the order of vertices
#     """
#     N = np.cross(B - A, C - A)  # Normal vector
#     N_normalized = N / np.linalg.norm(N)  # Normalize for consistent comparison
#     if np.dot(N_normalized, A) < 0:  # Ensure normal points outward
#         N_normalized = -N_normalized
#     d = -np.dot(N_normalized, A)  # Compute d
#     return np.append(N_normalized, d)



def compute_plane_equation_consistent(A, B, C):
    N = np.cross(B - A, C - A)
    N = N / np.linalg.norm(N)
    ## Ensure that the normal vetor points in -x axis directions
    REFERENCE_DIRECTION = np.array([-1, 0, 0])  
    # Flip if pointing opposite to reference
    if np.dot(N, REFERENCE_DIRECTION) < 0:
        N = -N
    
    d = -np.dot(N, A)
    return np.append(N, d)

def is_above_planes(points, vertices, faces, epsilon=1):
    """Check if points are above all faces.
       add epsilon avoiding 'Strict Above' condition 
       -> the results is not perfect, but at least more realistic 
    """
    above_mask = np.ones(points.shape[0], dtype=bool)  # Start with all points valid

    for face in faces:
        A, B, C = vertices[face]  # Extract three vertices of the face
        a, b, c, d = compute_plane_equation_consistent(A, B, C)

        # Compute f(P) for all points
        f_values = a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d

        # Keep points that are above **all** planes (Not too strict)
        above_mask &= f_values > -epsilon  

    return points[above_mask]  # Return points above all planes


# bc: num_points in each mesh is way less than num_point_cloud
def upsample_points(points, target_num):
    """
    Upsample points in mesh to match the removed annotated points in point cloud
    """
    num_points = points.shape[0]
    if num_points == 0:
        return np.array([])  # No points to upsample
    
    if num_points >= target_num:
        return points[:target_num]  # Trim if we have excess points
    
    # Compute how many additional points are needed
    num_extra = target_num - num_points
    
    # Duplicate existing points with slight variations
    extra_indices = np.random.choice(num_points, num_extra, replace=True)
    noise = np.random.normal(scale=0.001, size=(num_extra, 3))  # Small random perturbation
    extra_points = points[extra_indices] + noise
    
    return np.vstack((points, extra_points))


def add_gaussian_curve_noise(points, scale_x=1.0, scale_y=0.2, scale_z=0.1):
    """
    Adds Gaussian noise following a Gaussian curve along the x-axis while aligning the vertebra centroid with y-axis.

    Parameters:
    - points: (N, 4) array where first 3 columns are x, y, z and last column is the label.
    - scale_x: Strength of noise in x direction (larger at Gaussian curve edges).
    - scale_y: Strength of noise in y direction (subtle).
    - scale_z: Strength of noise in z direction (small).

    Returns:
    - Noisy version of input points with shape (N, 4).
    """
    # Compute vertebra centroid
    centroid = np.mean(points[:, :3], axis=0)

    # Define Gaussian curve in XY plane (centered at vertebra centroid)
    y_relative = points[:, 1] - centroid[1]  # Shift y-values to align with vertebra center
    
    # Gaussian function for shaping noise (curve peak towards -x)
    gaussian_x_factor = np.exp(-0.5 * (y_relative / 2.0)**2).reshape(-1, 1)  # Reshape for broadcasting
    
    # Reverse x-direction (move peak to -x instead of x)
    noise_x = np.random.normal(loc=-scale_x * gaussian_x_factor, scale=scale_x * gaussian_x_factor, size=(points.shape[0], 1)) 
    noise_y = np.random.normal(loc=0, scale=scale_y, size=(points.shape[0], 1))  # Subtle in y
    noise_z = np.random.normal(loc=0, scale=scale_z, size=(points.shape[0], 1))  # Small in z

    # Apply noise to points
    noisy_xyz = points[:, :3] + np.hstack((noise_x, noise_y, noise_z))
    
    return np.hstack((noisy_xyz, points[:, 3:]))  # Concatenate with labels


def blend_colors(color1, color2, alpha):
    """Blend two colors based on alpha (0 to 1)."""
    return (1 - alpha) * color1 + alpha * color2

def assign_realistic_colors(points, labels, anatomy_colors):
    np.random.seed(42)  # For reproducibility
    colors = np.zeros((points.shape[0], 3))

    vertebra_colors = np.array(anatomy_colors['vertebra'])  # Shape: (4, 3)
    
    for i, point in enumerate(points):
        if labels[i] == 0:  # Non-vertebra points
            dist, nearest_vertebra = centroid_tree.query(point)

            # Assign colors based on distance from vertebrae
            if dist < 15:  
                base_color = anatomy_colors['dark']
                mix_color = anatomy_colors['clot']
            elif dist < 20:  
                base_color = anatomy_colors['blood']
                mix_color = anatomy_colors['bleeding']
            elif dist < 28: 
                base_color = anatomy_colors['blood']
                mix_color = anatomy_colors['muscle']
            elif dist <= int(max_dist)-5:
                base_color = anatomy_colors['dark']
                mix_color = anatomy_colors['clot']
            
            # Random interpolation between base and mix color
            alpha = np.random.uniform(0, 1)
            blended_color = blend_colors(base_color, mix_color, alpha)

            # Add small noise for realism
            noise = np.random.normal(0, 15, 3)  # Small RGB noise
            colors[i] = np.clip(blended_color + noise, 0, 255)

        else:  # Vertebra points
            # Generate 4 random weights that sum to 1
            weights = np.random.dirichlet(np.ones(4), size=1).flatten()

            # Blend all four vertebra colors using weighted sum
            blended_color = np.sum(vertebra_colors * weights[:, None], axis=0)

            # Add slight variations for realism
            noise = np.random.normal(0, 10, 3)  # Small RGB variation
            colors[i] = np.clip(blended_color + noise, 0, 255)

    return colors

