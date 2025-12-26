"""
From tag points, add extra points to create a better surface to cover desired area
"""
import bpy


# define create point function
def create_point(tag, coords, collection):
    """ Create a point object at given coordinates if it doesn't already exist. """
    if tag not in collection.objects:
        mesh = bpy.data.meshes.new(tag)
        obj = bpy.data.objects.new(tag, mesh)
        mesh.from_pydata([coords], [], [])
        mesh.update()
        collection.objects.link(obj)
        print(f"Created '{tag}' at {coords}")
    else:
        print(f"'{tag}' already exists, skipping.")


# Loop through all 12 vertebrae collections
for i in range(1, 13):
    collection_name = f"Vertebre_T{i}"
    suffix = f".{i-1:03}" if i > 1 else ""
    
    if collection_name not in bpy.data.collections:
        print(f"Collection '{collection_name}' not found.")
        continue
    
    collection = bpy.data.collections[collection_name]
    tags = [
        f"Ped_Inf_D{suffix}", f"Ped_Inf_G{suffix}",
        f"Ped_Sup_D{suffix}", f"Ped_Sup_G{suffix}",
        f"Apo_Trans_D{suffix}", f"Apo_Trans_G{suffix}"
    ]
    
    objects = [collection.objects.get(tag) for tag in tags]
    if not all(obj and obj.type == 'MESH' for obj in objects):
        print(f"Missing objects in collection '{collection_name}'")
        continue
    
    p1, p2, p3, p4, pd, pg = [obj.data.vertices[0].co for obj in objects]

    # Calculate midpoints
    Ped_Inf_Mid = [(p1[j] + p2[j]) / 2 for j in range(3)]
    Ped_Sup_Mid = [(p3[j] + p4[j]) / 2 for j in range(3)]

    # Calculate 4*1/3 points
    Ped_Inf_Third_D = [(2/3) * pd[j] + (1/3) * Ped_Inf_Mid[j] for j in range(3)]
    Ped_Sup_Third_D = [(2/3) * pd[j] + (1/3) * Ped_Sup_Mid[j] for j in range(3)]
    # Midpoint of Ped_Inf_Third_D and Ped_Sup_Third_D
    Ped_Mid_Third_D = [(Ped_Inf_Third_D[j] + Ped_Sup_Third_D[j]) / 2 for j in range(3)] # to reduce the number of vertices
    
    Ped_Inf_Third_G = [(2/3) * pg[j] + (1/3) * Ped_Inf_Mid[j] for j in range(3)]
    Ped_Sup_Third_G = [(2/3) * pg[j] + (1/3) * Ped_Sup_Mid[j] for j in range(3)]
    # Midpoint of Ped_Inf_Third_G and Ped_Sup_Third_G
    Ped_Mid_Third_G = [(Ped_Inf_Third_G[j] + Ped_Sup_Third_G[j]) / 2 for j in range(3)] # to reduce the number of vertices

    # Create points in Blender
    # create_point(f"Ped_Inf_Third_D{suffix}", Ped_Inf_Third_D, collection)
    # create_point(f"Ped_Sup_Third_D{suffix}", Ped_Sup_Third_D, collection)
    create_point(f"Ped_Mid_Third_D{suffix}", Ped_Mid_Third_D, collection)
    
    # create_point(f"Ped_Inf_Third_G{suffix}", Ped_Inf_Third_G, collection)
    # create_point(f"Ped_Sup_Third_G{suffix}", Ped_Sup_Third_G, collection)
    create_point(f"Ped_Mid_Third_G{suffix}", Ped_Mid_Third_G, collection)
    
    print(f"Processed {collection_name}")

print("Processing complete!")