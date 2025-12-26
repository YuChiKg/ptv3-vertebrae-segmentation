import os
import numpy as np
import json
import open3d as o3d
from pysdf import SDF
import argparse
from tqdm import tqdm
from postprocess_tool import get_JSON_files, mesh_from_stl, is_above_planes, upsample_points, add_gaussian_curve_noise
from scipy.spatial import cKDTree
from typing import Dict, List, Tuple, Generator
'''
Code for iterate through ply files {sj}/{date}/{sj}.ply
Assigning point cloud colors with surgical exposure with CHU statistics results
'''
# Constants
VERTEBRA_NAMES = [f"Vertebre_T{i}" for i in range(1, 13)] + ["Vertebre_L1"]
TAGS = ["Apo_Epin_Post", "Ped_Inf_D", "Ped_Inf_G", "Ped_Sup_D", "Ped_Sup_G", "Apo_Trans_D", "Apo_Trans_G"]
VERTEBRA_MEAN = np.array([100, 69, 50])
VERTEBRA_STD = np.array([21, 25, 25])
VERTEBRA_MIN = np.array([50, 24, 10])
VERTEBRA_MAX = np.array([255, 255, 254])
VERTEBRA_MODE = np.array([85, 50, 31])


NON_VERTEBRA_MEAN = np.array([48, 27, 19])
NON_VERTEBRA_STD = np.array([15, 9, 7])
NON_VERTEBRA_MIN = np.array([9, 8, 5])
NON_VERTEBRA_MAX = np.array([74, 71, 73])
NON_VERTEBRA_MODE = np.array([46, 23, 14])

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process and label vertebral point clouds.")
    
    # Directory arguments
    parser.add_argument('--o3_dir', type=str, default='/home/travail/Antonin_Dataset/o3_symlink',
                        help='Base directory for JSON files')
    parser.add_argument('--ply_dir', type=str, default='/home/travail/Antonin_Dataset/pc_symlink',
                        help='Base directory for PLY files')
    parser.add_argument('--stl_dir', type=str, default='/home/travail/Antonin_Dataset/Vertebre_symlink',
                        help='Base directory for STL files')
    
    # Processing parameters
    # parser.add_argument('--num_division', type = int, default=10, help='Number of division for Apo_Trans_x to Ped_Mid_x')
    parser.add_argument('--upsample', action='store_true', help='Enable upsampling of points above surfaces')
    parser.add_argument('--noise_scale', type=float, nargs=3, default=[6.0, 5.0, 5.0], help='Scale for Gaussian noise (x, y, z)')
    parser.add_argument('--alpha', type=float, default=0.4, help='Alpha blending value between mean and mode colors (range: 0.0 to 1.0)')
    
    return parser.parse_args()

def file_generator(json_dict: Dict) -> Generator[Tuple[str, str, str], None, None]:
    """Yield tuples of (subject_folder, date, file_path) from JSON dictionary."""
    for sj_folder, date_dict in json_dict.items():
        for date, file_path in date_dict.items():
            yield sj_folder, date, file_path

def create_vertebral_surfaces(data: Dict, tags: List[str]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create surfaces between adjacent vertebrae based on JSON data."""
    surfaces = []
    
    for idx in range(len(VERTEBRA_NAMES) - 1):
        v1_name, v2_name = VERTEBRA_NAMES[idx], VERTEBRA_NAMES[idx + 1]
        v1 = data['Objets'].get(v1_name, {})
        v2 = data['Objets'].get(v2_name, {})
        
        if not v1 or not v2:
            print(f'{v1_name} or {v2_name} does not exist in the JSON file')
            continue
        
        # Extract points for both vertebrae
        point_dict_1 = {p['tag']: (p['x'], p['y'], p['z']) for p in v1.get('Points', []) if p['tag'] in tags}
        point_dict_2 = {p['tag']: (p['x'], p['y'], p['z']) for p in v2.get('Points', []) if p['tag'] in tags}
        
        # Calculate intermediate points
        pd_1, pd_2 = point_dict_1["Apo_Trans_D"], point_dict_2["Apo_Trans_D"]
        pg_1, pg_2 = point_dict_1["Apo_Trans_G"], point_dict_2["Apo_Trans_G"]
        
        # Create surface vertices and faces
        vertices_surface = np.array([
            pd_1, pg_1, 
            pg_2, pd_2
        ])
        
        faces_surface = np.array([
            [0, 1, 2], [2, 3, 0]  
        ], dtype=np.uint32)
        
        surfaces.append((vertices_surface, faces_surface))
    
    return surfaces

def blend_colors(color1: np.ndarray, color2: np.ndarray, alpha: float) -> np.ndarray:
    """Blend two colors based on alpha (0 to 1)."""
    return (1 - alpha) * color1 + alpha * color2

def assign_realistic_colors(points: np.ndarray, labels: np.ndarray, centroids: List[np.ndarray], alpha: float) -> np.ndarray:
    np.random.seed(42)
    colors = np.zeros((points.shape[0], 3))
    
    centroid_tree = cKDTree(centroids)
    # For distance-based influence
    distances, _ = centroid_tree.query(points)
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    print(f"Max Distance: {max_dist}")
    for i, (point, label, dist) in enumerate(zip(points, labels, distances)):
        if label == 0:
            # ---- Non-Vertebra Point ----
            # Blend non-vertebra MEAN and MODE
            base_color = blend_colors(NON_VERTEBRA_MEAN, NON_VERTEBRA_MODE, alpha=alpha)
            
            # Generate noisy color based on base and STD
            color = np.random.normal(loc=base_color, scale=NON_VERTEBRA_STD)
            
            # Distance-aware blend: shift toward vertebra tone if closer
            if dist < max_dist:
                # Normalize dist to [0,1], invert it (closer = higher influence)
                dist_weight = 1 - dist / 30.0
                vertebra_tone = blend_colors(VERTEBRA_MEAN, VERTEBRA_MODE, alpha=alpha)
                color = blend_colors(color, vertebra_tone, alpha=dist_weight * 0.5)

            # Clip to observed bounds
            color = np.clip(color, NON_VERTEBRA_MIN, NON_VERTEBRA_MAX)
        
        else:
            # ---- Vertebra Point ----
            # Blend vertebra MEAN and MODE
            base_color = blend_colors(VERTEBRA_MEAN, VERTEBRA_MODE, alpha=alpha)
            
            # Generate noisy color based on base and STD
            color = np.random.normal(loc=base_color, scale=VERTEBRA_STD)
            
            # Clip to observed bounds
            color = np.clip(color, VERTEBRA_MIN, VERTEBRA_MAX)

        # Final safety clip
        colors[i] = np.clip(color, 0, 255)

    return colors

def process_vertebra(vertebra_num: int, current_points: np.ndarray, current_labels: np.ndarray, 
                    surfaces: List[Tuple[np.ndarray, np.ndarray]], vertebra_data: List[Tuple[np.ndarray, np.ndarray]],
                    upsample: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Process a single vertebra, updating points and labels."""
    vertebra_name = f"Vertebre_T{vertebra_num}" if vertebra_num <= 12 else "Vertebre_L1"
    print(f"\nProcessing {vertebra_name}...")

    # Determine which surface to use
    if vertebra_num < 13:  # For T1-T11, use their corresponding surface
        surface_idx = vertebra_num - 1
    elif vertebra_num == 13:  # T12 uses the T11-T12 surface
        surface_idx = 11  # Index of T11-T12 surface
        
    vertices_surface, faces_surface = surfaces[surface_idx]
    vertices_vertebra = vertebra_data[vertebra_num - 1][0]

    vertices_above_surface = is_above_planes(vertices_vertebra, vertices_surface, faces_surface)
    print(f"Found {vertices_above_surface.shape[0]} points above {vertebra_name} surface")

    # Remove points currently labeled with this vertebra number
    keep_mask = current_labels != vertebra_num
    filtered_points = current_points[keep_mask]
    filtered_labels = current_labels[keep_mask]
    num_removed = (current_labels == vertebra_num).sum()
    print(f"Removed {num_removed} points labeled as {vertebra_num}")

    # Upsample if needed
    if upsample and num_removed > vertices_above_surface.shape[0]:
        vertices_above_surface = upsample_points(vertices_above_surface, num_removed)
        print(f"Upsampled points from {vertices_above_surface.shape[0]} → {num_removed}")

    # Add new points with this vertebra's label
    new_labels = np.full((vertices_above_surface.shape[0],), vertebra_num)
    updated_points = np.concatenate((filtered_points, vertices_above_surface), axis=0)
    updated_labels = np.concatenate((filtered_labels, new_labels), axis=0)
    
    print(f"New total points: {updated_points.shape[0]}")
    return updated_points, updated_labels

def main():
    args = parse_arguments()
    
    # Load file structure
    json_dict = get_JSON_files(args.o3_dir)
    
    files_to_process = list(file_generator(json_dict))
    with tqdm(files_to_process, desc="Processing subjects", unit="subject") as pbar:
    
        for sj, date, json_path in file_generator(json_dict):
            print(f"\nProcessing {sj} {date} files: ")
            
            # Check if output exists and skip if requested
            ply_file_path = os.path.join(args.ply_dir, sj, date, f'{sj}.ply')
            npy_file_path = ply_file_path.replace('.ply', '.npy')
            
            # Load point cloud
            point_cloud = o3d.io.read_point_cloud(ply_file_path)
            points = np.asarray(point_cloud.points)
            labels = np.zeros(points.shape[0], dtype=int)
            
            # Step 1: Label the point cloud with vertebrae
            vertebra_data = []
            for idx, vertebra_name in enumerate(VERTEBRA_NAMES, start=1):
                stl_dir = f"/home/travail/Antonin_Dataset/Vertebre_symlink/{sj}/{date}"
                # Try both possible file name patterns
                stl_file = os.path.join(stl_dir, f"{sj}_{vertebra_name}.stl")
                lis_stl_file = os.path.join(stl_dir, f"LIS_{sj}_{vertebra_name}.stl")

                # Pick the one that exists
                chosen_stl_file = None
                if os.path.exists(stl_file):
                    chosen_stl_file = stl_file
                elif os.path.exists(lis_stl_file):
                    chosen_stl_file = lis_stl_file
                    
                vertices, faces = mesh_from_stl(chosen_stl_file)
                vertebra_data.append((vertices, faces))
                
                sdf = SDF(vertices, faces)
                labels[sdf.contains(points)] = idx
            
            print("Labeling complete!")
            
            # Step 2: Create vertebral surfaces from JSON data
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            surfaces = create_vertebral_surfaces(data, TAGS)
            print(f'Created {len(surfaces)} vertebra surfaces')
            
            # Step 3: Process each vertebra
            current_points, current_labels = points.copy(), labels.copy()
            for vertebra_num in range(1, 14):  # T1 to L1
                current_points, current_labels = process_vertebra(
                    vertebra_num, current_points, current_labels, surfaces, vertebra_data, args.upsample
                )
            
            # Final labeled point cloud
            final_labeled_points = np.concatenate((current_points, current_labels[:, None]), axis=1)
            print("\nFinal labeled point cloud shape:", final_labeled_points.shape)
            
            labels = final_labeled_points[:, -1]
            # Add noise to non-vertebra points
            mask = labels == 0
            noisy_non_vertebrae = add_gaussian_curve_noise(
                final_labeled_points[mask], 
                scale_x=args.noise_scale[0], 
                scale_y=args.noise_scale[1], 
                scale_z=args.noise_scale[2]
            )
            vertebre_mask = labels > 0
            labeled_vertebre = final_labeled_points[vertebre_mask]
            added_noise_pc = np.vstack((noisy_non_vertebrae, labeled_vertebre))
            
            # Assign colors
            centroids = [np.mean(added_noise_pc[labels == i], axis=0) for i in range(1, 13)]
            colors = assign_realistic_colors(added_noise_pc, labels, centroids, args.alpha)
            print("Color assignment complete!")
            
            # Save final point cloud with colors
            bg_pc = np.concatenate((added_noise_pc[:, 0:3], colors, current_labels[:, None]), axis=1)
            np.save(npy_file_path, bg_pc)
            print(f"Saved colored point cloud to {npy_file_path}")

if __name__ == "__main__":
    main()