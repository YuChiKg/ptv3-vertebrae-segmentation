"""
This script is used to create tag points as vertices in blender
"""
import bpy
import json

# Load the json file
json_path = '/Users/yuchi/Documents/Antonin_Dataset/o3/SJ0000270/2006-09-05/SJ0000270.json'
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    
# iterate through each vertebre (Vertebre_T1 ~ Vertebre_T12)
for i in range(1, 13):
    vertebra_name = f'Vertebre_T{i}'
    
    # check if the vertebra exists in the json file
    vertebra = data['Objets'].get(vertebra_name, {})
    if not vertebra:
        print(f'{vertebra_name} does not exist in the json file')
        continue
    # retrieve the points for current vertebra
    points =[]
    for p in vertebra.get('Points', []):
        points.append((p['tag'], p['x'], p['y'], p['z']))
        
    # Create or get the collection for this vertebra
    collection_name = vertebra_name
    if collection_name not in bpy.data.collections:
        collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(collection)
    else:
        collection = bpy.data.collections[collection_name]

    # Create each point as an individual object
    for point in points:
        tag = point[0]  # The tag (e.g., 'Apo_Epin_Post', 'Ped_Sup_G', etc.)
        x, y, z = point[1], point[2], point[3]

        # Check if the point object already exists in the collection
        if tag not in bpy.data.objects:
            # Create the mesh for the point
            mesh = bpy.data.meshes.new(tag)
            obj = bpy.data.objects.new(tag, mesh)

            # Set the location of the object
            obj.location = (x, y, z)

            # Link the object to the collection
            collection.objects.link(obj)

    print(f"Added {len(points)} points to the '{collection_name}' collection, each named after its tag.")

print("Processing complete!")
