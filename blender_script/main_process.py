import bpy
import json
import os


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
            print(f"No date folders found in {sj_folder}")
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

def JSON_file_generator(JSON_dict):
    """Yield tuples of (sj_folder, date, file_path)"""
    for sj_folder, date_dict in JSON_dict.items():
        for date, file_path in date_dict.items():
            yield sj_folder, date, file_path
        
def clear_scene():
    """Clear all objects and collections except the master collection"""
    # Unlink all objects from all collections
    for collection in bpy.data.collections:
        for obj in collection.objects:
            collection.objects.unlink(obj)
    
    # Remove all objects
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj)
    
    # Remove all collections except the master collection
    for collection in bpy.data.collections:
        if collection.name != "Collection":
            bpy.data.collections.remove(collection)
    
    # Clear orphan data
    for block in [bpy.data.meshes, bpy.data.materials, bpy.data.curves]:
        for item in block:
            if not item.users:
                block.remove(item)

def get_or_create_collection(name, parent_name=None):
    """Safely get or create collection with parenting"""
    collection = bpy.data.collections.get(name)
    if not collection:
        collection = bpy.data.collections.new(name)
        parent = bpy.data.collections.get(parent_name) if parent_name else bpy.context.scene.collection
        parent.children.link(collection)
    return collection

def create_vertebra_surface(vert1, vert2, landmark_tags, collection):
    """Create a single surface between two vertebrae with error handling"""
    try:
        col1 = bpy.data.collections.get(f"Vertebre_{vert1}")
        col2 = bpy.data.collections.get(f"Vertebre_{vert2}")
        
        if not col1 or not col2:
            print(f"Skipping {vert1}-{vert2}: Missing vertebra collections")
            return None
        
        # Collect landmarks safely
        landmarks = []
        for tag in landmark_tags:
            obj1 = col1.objects.get(f"{tag}_{vert1}")
            obj2 = col2.objects.get(f"{tag}_{vert2}")
            if obj1 and obj2:
                landmarks.extend([obj1, obj2])
            else:
                print(f"Skipping {vert1}-{vert2}: Missing {tag} landmarks")
                return None
        
        # Create mesh
        surface_name = f"surface_{vert1}{vert2}"
        mesh = bpy.data.meshes.new(surface_name)
        obj = bpy.data.objects.new(surface_name, mesh)
        
        # Get vertices safely
        vertices = []
        for landmark in landmarks:
            if landmark and landmark.data and landmark.data.vertices:
                vertices.append(landmark.data.vertices[0].co)
            else:
                print(f"Invalid landmark data in {vert1}-{vert2}")
                return None
        
        # Face pattern
        faces = [[0, 2, 3, 1]]
        
        mesh.from_pydata(vertices, [], faces)
        collection.objects.link(obj)
        print(f"Created {surface_name}")
        return obj
        
    except Exception as e:
        print(f"Error creating {vert1}-{vert2} surface: {str(e)}")
        return None

def create_joined_surface(surface_objects, target_collection):
    """Safely create joined surface with proper object handling"""
    if not surface_objects:
        return
    
    # Make copies and join
    bpy.ops.object.select_all(action='DESELECT')
    copies = []
    
    try:
        for obj in surface_objects:
            if obj and obj.name in bpy.data.objects:
                copy = obj.copy()
                copy.data = obj.data.copy()
                target_collection.objects.link(copy)
                copies.append(copy)
                copy.select_set(True)
        
        if copies:
            bpy.context.view_layer.objects.active = copies[0]
            bpy.ops.object.join()
            joined = bpy.context.active_object
            joined.name = "surface_join_all"
            print(f"Created joined surface with {len(surface_objects)} segments")
            return joined
        
    except Exception as e:
        print(f"Error joining surfaces: {str(e)}")
    
    return None

def setup_geometry_nodes(obj):
    """Set up geometry nodes for point distribution on the surface"""
    if not obj:
        print("No object provided for geometry nodes setup")
        return False
    
    # Add a new Geometry Nodes modifier
    if "GeometryNodes" not in obj.modifiers:
        geo_mod = obj.modifiers.new(name="GeometryNodes", type='NODES')
    else:
        geo_mod = obj.modifiers["GeometryNodes"]

    # Create or get an existing node group
    if geo_mod.node_group is None:
        node_group = bpy.data.node_groups.new(name="surface_points", type='GeometryNodeTree')
        geo_mod.node_group = node_group
    else:
        node_group = geo_mod.node_group

    # Clear existing nodes
    node_group.nodes.clear()

    # Create Group Input and Output nodes
    group_input = node_group.nodes.new(type='NodeGroupInput')
    group_output = node_group.nodes.new(type='NodeGroupOutput')

    # Move nodes for better organization
    group_input.location = (-600, 0)
    group_output.location = (600, 0)

    # Ensure the node group has correct input/output sockets
    node_group.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    node_group.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')

    # Add Distribute Points on Faces
    distribute_points = node_group.nodes.new(type='GeometryNodeDistributePointsOnFaces')
    distribute_points.distribute_method = 'POISSON'
    distribute_points.inputs["Distance Min"].default_value = 0.5
    distribute_points.inputs["Density Max"].default_value = 18.0
    distribute_points.inputs["Density Factor"].default_value = 1.0
    distribute_points.inputs["Seed"].default_value = 2
    distribute_points.location = (-350, 0)

    # Add Instance on Points
    instance_on_points = node_group.nodes.new(type='GeometryNodeInstanceOnPoints')
    instance_on_points.location = (-100, 0)

    # Add Instances to Points
    instances_to_points = node_group.nodes.new(type='GeometryNodeInstancesToPoints')
    instances_to_points.inputs["Radius"].default_value = 0.05
    instances_to_points.location = (100, 0)

    # Add Points to Vertices
    points_to_vertices = node_group.nodes.new(type='GeometryNodePointsToVertices')
    points_to_vertices.location = (300, 0)

    # Connect the nodes
    node_group.links.new(group_input.outputs[0], distribute_points.inputs["Mesh"])
    node_group.links.new(distribute_points.outputs["Points"], instance_on_points.inputs["Points"])
    node_group.links.new(instance_on_points.outputs["Instances"], instances_to_points.inputs["Instances"])
    node_group.links.new(instances_to_points.outputs["Points"], points_to_vertices.inputs["Points"])
    node_group.links.new(points_to_vertices.outputs["Mesh"], group_output.inputs[0])

    print("Geometry Nodes setup applied successfully!")
    return True

def export_points_as_ply(obj, export_path):
    """Export the points from an object as a PLY file"""
    if not obj:
        print("No object provided for export")
        return False
    
    # Evaluate the geometry nodes
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    
    # Get the mesh data from the evaluated object
    mesh = eval_obj.data
    
    # Extract vertices
    vertices = [vert.co for vert in mesh.vertices]
    
    if not vertices:
        print("No vertices found in the evaluated geometry")
        return False
    
    # Create temporary mesh and object
    temp_mesh = bpy.data.meshes.new("temp_export_mesh")
    temp_mesh.from_pydata(vertices, [], [])
    temp_obj = bpy.data.objects.new("temp_export_obj", temp_mesh)
    
    # Link to scene temporarily
    bpy.context.collection.objects.link(temp_obj)
    
    # Select and export
    bpy.ops.object.select_all(action='DESELECT')
    temp_obj.select_set(True)
    bpy.context.view_layer.objects.active = temp_obj
    
    # Export as PLY
    bpy.ops.wm.ply_export(filepath=export_path, ascii_format=True, export_selected_objects=True)
    
    # Clean up
    bpy.data.objects.remove(temp_obj)
    bpy.data.meshes.remove(temp_mesh)
    
    print(f"Successfully exported points to: {export_path}")
    return True

# Main processing loop
ply_dir = '/Volumes/Seagate/Antonin/Dataset/Ply'
JSON_files = get_JSON_files("/Volumes/Seagate/Antonin/Dataset/o3")

for sj, date, json_path in JSON_file_generator(JSON_files):
    save_dir = os.path.join(ply_dir, sj, date)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nProcessing {sj} {date}")
    
    # Clear the scene before processing each patient/date
    clear_scene()
    
    stl_dir = f"/Volumes/Seagate/Antonin/Dataset/Vertebre/{sj}/{date}"

    # --- Part 1: Import STL files ---
    stl_collection = get_or_create_collection("Vertebre_STL")

    # Vertebrae names (T1-T12 and L1)
    vertebra_names = [f"Vertebre_T{i}" for i in range(1, 13)] + ["Vertebre_L1"]

    for vertebra_name in vertebra_names:
        # Try both possible file name patterns
        stl_file = os.path.join(stl_dir, f"{sj}_{vertebra_name}.stl")
        lis_stl_file = os.path.join(stl_dir, f"LIS_{sj}_{vertebra_name}.stl")
        
        # Pick the one that exists
        chosen_stl_file = None
        if os.path.exists(stl_file):
            chosen_stl_file = stl_file
        elif os.path.exists(lis_stl_file):
            chosen_stl_file = lis_stl_file

        if chosen_stl_file:
            # Import STL
            bpy.ops.wm.stl_import(filepath=chosen_stl_file)
            imported_obj = bpy.context.selected_objects[0]
            imported_obj.name = vertebra_name  # Rename object
            
            # Link to collection and unlink from default collection
            stl_collection.objects.link(imported_obj)
            bpy.context.scene.collection.objects.unlink(imported_obj)
            print(f"Imported: {chosen_stl_file}")
        else:
            print(f"STL file not found for {vertebra_name} in {stl_dir}")

    # --- Part 2: Process JSON and create point markers ---
    # Load JSON data
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def create_point_marker(tag, coords, collection):
        """Create a visible point marker at specified coordinates"""
        mesh = bpy.data.meshes.new(tag)
        mesh.from_pydata([coords], [], [])
        mesh.update()

        obj = bpy.data.objects.new(tag, mesh)
        collection.objects.link(obj)
        return obj

    # Process Thoracic (T1-T12) and Lumbar (L1)
    for vertebra_name in vertebra_names:
        vertebra_short = vertebra_name.replace("Vertebre_", "")  # e.g., "T1", "L1"
        vertebra = data["Objets"].get(vertebra_name, {})
        if not vertebra:
            print(f"{vertebra_name} not found in JSON")
            continue

        # Retrieve points
        points = [(p["tag"], (p["x"], p["y"], p["z"])) for p in vertebra.get("Points", [])]

        # Create or get the collection
        collection = get_or_create_collection(vertebra_name)

        # Create point markers
        created_count = 0
        for tag, coords in points:
            unique_tag = f"{tag}_{vertebra_short}"  # e.g., "Apo_Trans_D_T1"
            if unique_tag not in collection.objects:
                create_point_marker(unique_tag, coords, collection)
                created_count += 1

        print(f"Added {created_count} points to '{vertebra_name}' collection")

    # --- Part 3: Create surfaces between vertebrae ---
    # Setup collections for surfaces
    spinal_surfaces_col = get_or_create_collection("Spinal_Surfaces")
    separate_surfaces_col = get_or_create_collection("Separate_Surfaces", "Spinal_Surfaces")
    joined_surfaces_col = get_or_create_collection("Joined_Surfaces", "Spinal_Surfaces")
    
    # Tags for surface creation
    landmark_tags = ['Apo_Trans_D', 'Apo_Trans_G']
    surface_objects = []
    
    # Create all surfaces
    surface_pairs = (
        [(f"T{i}", f"T{i+1}") for i in range(1, 12)] +  # Thoracic
        [("T12", "L1")]                               # Transition
    )
    
    for vert1, vert2 in surface_pairs:
        obj = create_vertebra_surface(vert1, vert2, landmark_tags, separate_surfaces_col)
        if obj:
            surface_objects.append(obj)
    
    # Create joined version
    joined_surface = create_joined_surface(surface_objects, joined_surfaces_col)
    
    if not joined_surface:
        print("Failed to create joined surface, skipping export")
        continue
    
    # --- Part 4: Setup Geometry Nodes for point distribution ---
    if not setup_geometry_nodes(joined_surface):
        print("Failed to setup geometry nodes, skipping export")
        continue
    
    # --- Part 5: Export points as PLY ---
    export_path = os.path.join(save_dir, f"{sj}.ply")
    if not export_points_as_ply(joined_surface, export_path):
        print(f"Failed to export PLY for {sj} {date}")

    # Clear the scene after processing each patient/date
    clear_scene()

print("=== All processing complete ===")