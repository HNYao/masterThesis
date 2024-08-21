
import objaverse
import trimesh
import os

print(objaverse.__version__)
uids = objaverse.load_uids()
print(len(uids), type(uids))
annotations = objaverse.load_annotations(uids[:10])
#print(annotations[uids[0]])
annotations = objaverse.load_annotations()
cc_by_uids = [uid for uid, annotation in annotations.items() if "keyboard" in annotation["name"]]
print(cc_by_uids[:10])
objects = objaverse.load_objects(uids=cc_by_uids[:20])
for i in range(20):
    mesh = trimesh.load(list(objects.values())[i])
    mesh.show()
    path = f'dataset/obj/objaverse/keyboard/keyboard_{i}'
    os.makedirs(path, exist_ok=True)
    mesh.export(f'{path}/keyboard_{i}.obj')
    
