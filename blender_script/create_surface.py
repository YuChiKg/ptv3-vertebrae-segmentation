import bpy
"""
Automatically select the tag vertices in Vertebre_T{i} and Vertevre_T{i+1} and create the surface
Add join surface to join all the sub surfaces
"""

def create_surface_between_vertebrae():
    """Create surfaces between adjacent vertebrae and make a joined version"""
    
    # Create or get the 'Surfaces' collection
    surfaces_collection = bpy.data.collections.get("Surfaces")
    if not surfaces_collection:
        surfaces_collection = bpy.data.collections.new("Surfaces")
        bpy.context.scene.collection.children.link(surfaces_collection)
        print("Created 'Surfaces' collection")
    
    # Tags we'll use to create surfaces
    landmark_tags = ['Apo_Trans_D', 'Apo_Trans_G', 'Ped_Mid_Third_D', 'Ped_Mid_Third_G']
    
    # List to store all created surface objects for joining
    surface_objects = []
    
    # Create 11 surfaces (T1-T2 to T11-T12)
    for surface_num in range(1, 12):
        vert1_num = surface_num
        vert2_num = surface_num + 1
        collection1_name = f"Vertebre_T{vert1_num}"
        collection2_name = f"Vertebre_T{vert2_num}"
        
        # Get both vertebra collections
        col1 = bpy.data.collections.get(collection1_name)
        col2 = bpy.data.collections.get(collection2_name)
        
        if not col1 or not col2:
            print(f"Missing collection {collection1_name} or {collection2_name}")
            continue
        
        # Get all landmark objects from both vertebrae
        landmark_objects = []
        suffix1 = f".{vert1_num-1:03}" if vert1_num > 1 else ""
        suffix2 = f".{vert2_num-1:03}" if vert2_num > 1 else ""
        
        for tag in landmark_tags:
            obj1 = col1.objects.get(f"{tag}{suffix1}")
            obj2 = col2.objects.get(f"{tag}{suffix2}")
            
            if obj1 and obj2:
                landmark_objects.extend([obj1, obj2])
            else:
                print(f"Missing {tag} in {collection1_name} or {collection2_name}")
        
        if len(landmark_objects) != 8:  # 4 tags × 2 vertebrae
            print(f"Skipping surface {vert1_num}{vert2_num} due to missing landmarks")
            continue
        
        # Create the surface mesh
        surface_name = f"surface_{vert1_num}{vert2_num}"
        mesh = bpy.data.meshes.new(surface_name)
        obj = bpy.data.objects.new(surface_name, mesh)
        
        # Get vertex coordinates
        vertices = [obj.data.vertices[0].co for obj in landmark_objects]
        
        # Define faces
        faces = [
            [0, 4, 5, 1], 
            [4, 6, 7, 5],
            [2, 3, 7, 6]

        ]
        
        # Build the mesh
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        
        # Add to collection and tracking list
        surfaces_collection.objects.link(obj)
        surface_objects.append(obj)
        print(f"Created {surface_name}")

    # Create joined version if we have surfaces
    if surface_objects:
        # Make a copy of all surfaces for joining
        bpy.ops.object.select_all(action='DESELECT')
        copies = []
        
        for obj in surface_objects:
            copy = obj.copy()
            copy.data = obj.data.copy()
            surfaces_collection.objects.link(copy)
            copies.append(copy)
            copy.select_set(True)
        
        # Join the copies
        bpy.context.view_layer.objects.active = copies[0]
        bpy.ops.object.join()
        
        # Rename and position the joined surface
        joined_surface = bpy.context.active_object
        joined_surface.name = "surface_join"
        
        print(f"Created joined surface 'surface_join' from {len(copies)} surfaces")
    else:
        print("No surfaces were created - cannot create joined version")

# Execute the function
create_surface_between_vertebrae()
print("=== Surface creation complete ===")