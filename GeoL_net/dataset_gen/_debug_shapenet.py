
import os
from glob import glob

catId2name = {2:'Bowl', 6: 'Mug', 1: 'Bottle', 102:'Cap'}
catId2dir  = {2:'02880940', 6: '03797390', 1:'02876657', 102: '02954340'}


# Acquire all samples
shapenet_dir= '/home/stud/zhoy/storage/group/srl/students/zhong/ShapeNet_NOCS_ONet_Deformation_demo/obj_models/train'
all_samples = {}
for class_id, class_dir in catId2dir.items():
    all_dirs = glob(os.path.join(shapenet_dir, class_dir, '*'))
    all_samples[class_id] = [d.split('/')[-1] for d in all_dirs]
print("===========>", all_samples)