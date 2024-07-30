"""
translate all obj in dataset/obj/mesh to the origin point
"""
import trimesh
import os

parent_folder = "dataset/obj/mesh"
subfolders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]
for subfolder in subfolders:
    subsubfolders = [f.path for f in os.scandir(subfolder) if f.is_dir()]
    for subsubfolder in subsubfolders:
        print(subsubfolder)
        for root, dirs, files in os.walk(subsubfolder):
            for file in files:
                if file.endswith('.obj'):
                    obj_path = os.path.join(subsubfolder, file)
                    print(obj_path)
                    mesh = trimesh.load(obj_path)
                    centroid = mesh.bounding_box.centroid

                    translation = -centroid

                    mesh.apply_translation(translation)

                    output_file = obj_path
                    mesh.export(output_file)

                    print(f"Mesh centroid moved to origin and saved to {output_file}")

'''
mesh = trimesh.load('dataset/obj/mesh/desk/desk_0241_brown/mesh.obj')

centroid = mesh.bounding_box.centroid

translation = -centroid

mesh.apply_translation(translation)

output_file = 'exps/test_desk_oringin.obj'
mesh.export(output_file)

print(f"Mesh centroid moved to origin and saved to {output_file}")
'''