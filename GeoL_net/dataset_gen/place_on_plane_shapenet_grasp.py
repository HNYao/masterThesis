import blenderproc as bproc
from blenderproc.python.types.MeshObjectUtility import MeshObject, get_all_mesh_objects
from blenderproc.python.types.EntityUtility import delete_multiple 
import argparse
import numpy as np
import os
from blenderproc.python.writer.MyWriterUtility import write_my
from blenderproc.python.writer.CocoWriterUtility import write_coco_annotations
import time
from glob import glob


# read some parameters
parser = argparse.ArgumentParser()
parser.add_argument('config_path', nargs='?', help="Path to the configuration files")
min_dist = {'train': 0.05, 'val': 0.20, 'test': 0.20}
max_dist = {'train': 0.20, 'val': 0.25, 'test': 0.25}
num_objects = {'train': 2, 'val': 2, 'test':2}
num_objects_distract = {'train': 0, 'val': 0, 'test':0}

catId2name = {2:'Bowl', 6: 'Mug', 1: 'Bottle'}
catId2dir  = {2:'02880940', 6: '03797390', 1:'02876657'}
all_samples = {}
num_views = 50

# slurm
mode = 'train'
cfg = dict(H=480, W=640, K=[591.0125, 0, 322.525, 0, 590.16775, 244.11084, 0, 0, 1], 
        out_dir='/home/stud/zhoy/storage/group/srl/students/zhong/ShapeNet_NOCS_ONet_Deformation_demo', 
        # out_dir='ShapeNet_NOCS_ONet_Deformation', 
        cc_textures_path='resources',
        shapenet_dir= '/home/stud/zhoy/storage/group/srl/students/zhong/ShapeNet_NOCS_ONet_Deformation_demo/obj_models/{}'.format(mode))

K = np.array(cfg['K']).reshape(3, 3)
out_dir = cfg['out_dir']
shapenet_dir = cfg['shapenet_dir']
num_target_objects_perclass = num_objects[mode]
num_distractor_objects_perclass = num_objects_distract[mode]

# Acquire all samples
for class_id, class_dir in catId2dir.items():
    all_dirs = glob(os.path.join(shapenet_dir, class_dir, '*'))
    all_samples[class_id] = [d.split('/')[-1] for d in all_dirs]
print("===========>", all_samples)

# initialize the scene
bproc.init()
bproc.camera.set_intrinsics_from_K_matrix(K,  cfg['W'],  cfg['H'])

# sample target objects pose and spawn them
target_objects = []
for i, (category_id, category_name) in enumerate(catId2name.items()):
    if category_id in [2, 6, 1]:
        num_objs = np.random.choice(range(num_target_objects_perclass-1,num_target_objects_perclass+1))
        source_ids = np.random.choice(all_samples[category_id], num_objs, replace=False)  # No repeated instances from the same
        for source_id in source_ids:
            curr_obj = bproc.loader.load_shapenet(shapenet_dir, used_synset_id=catId2dir[category_id], 
                            used_source_id=source_id, move_object_origin=False)
            curr_obj.set_cp("category_id", category_id) 
            curr_obj.set_cp("scene_id", category_id)
            target_objects.append(curr_obj)

# sample distractor objects pose and spawn them
distractor_objects = []
for category_id, category_name in catId2name.items():
    if category_id not in [2, 6, 1]:
        for j in range(num_distractor_objects_perclass):
            source_id = np.random.choice(all_samples[category_id])
            curr_obj = bproc.loader.load_shapenet(shapenet_dir, used_synset_id=catId2dir[category_id], used_source_id=source_id)
            curr_obj.set_cp("category_id", category_id) 
            distractor_objects.append(curr_obj)

# set shading and physics properties and randomize PBR materials
for j, obj in enumerate(target_objects+distractor_objects):
    mass, fiction_coeff = (0.4, 0.5)
    obj.enable_rigidbody(True, mass=mass, friction=mass * fiction_coeff, 
    linear_damping = 1.99, angular_damping = 0, collision_margin=0.0001)
    obj.set_shading_mode('auto')
        
    mat = obj.get_materials()[0]
    mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
    mat.set_principled_shader_value("Specular", np.random.uniform(0, 1.0))

# create room
room = [bproc.object.create_primitive('PLANE', scale=[2.5, 2.5, 1]),
        bproc.object.create_primitive('PLANE', scale=[2.5, 2.5, 1], location=[0, -2.5, 2.5], rotation=[-1.570796, 0, 0]),
        bproc.object.create_primitive('PLANE', scale=[2.5, 2.5, 1], location=[0, 2.5, 2.5], rotation=[1.570796, 0, 0]),
        bproc.object.create_primitive('PLANE', scale=[2.5, 2.5, 1], location=[2.5, 0, 2.5], rotation=[0, -1.570796, 0]),
        bproc.object.create_primitive('PLANE', scale=[2.5, 2.5, 1], location=[-2.5, 0, 2.5], rotation=[0, 1.570796, 0])]


# sample point light on shell
light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')
light_plane_material.make_emissive(emission_strength=np.random.uniform(3,6), 
                                emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))    
light_plane.replace_materials(light_plane_material)

# sample CC Texture and assign to room planes
if cfg['cc_textures_path'] is not None:
    cc_textures = bproc.loader.load_ccmaterials(cfg['cc_textures_path'])
    for plane in room:
        random_cc_texture = np.random.choice(cc_textures)
        plane.replace_materials(random_cc_texture)


# set attributes
room.append(light_plane)
for plane in room:
    plane.enable_rigidbody(False, collision_shape='BOX', friction = 100, linear_damping = 0.0, angular_damping = 0.0)
    plane.set_cp('category_id', 0)

# sample point light on shell
light_point = bproc.types.Light()
light_point.set_energy(np.random.uniform(50, 200))
# lights = [[0,0,1], [1,0,0], [0,1,0]] debug only
light_point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))
location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 1, radius_max = 1.5,
                        elevation_min = 5, elevation_max = 89, uniform_volume = False)
light_point.set_location(location)


def sample_initial_pose(obj: bproc.types.MeshObject):
    # might need scale a bit for some specific class, currently fine with mugs and bowls
    obj.set_scale(np.ones(3) *  np.random.uniform(0.15, 0.25))
    if mode == 'train': 
        obj.set_rotation_euler([0, 0, np.random.uniform(0, 2*np.pi)])
    obj.set_location(bproc.sampler.upper_region(objects_to_sample_on=room[0:1],
                                                min_height=1, max_height=4, 
                                                face_sample_range=[0.4, 0.6]))

# Spawn object one by one with collision check
begin = time.time()
objects_ref_buffer = [room[0]]


# Sample objects on the given surface
placed_objects = bproc.object.sample_poses_on_surface(objects_to_sample=target_objects+distractor_objects,
                                         surface=room[0],
                                         sample_pose_func=sample_initial_pose,
                                         min_distance=min_dist[mode],
                                         max_distance=max_dist[mode])


# BVH tree used for camera obstacle checks
bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(placed_objects)

# setup camera
i = 0
radius_min, radius_max = (1.2, 1.5)
_radius = np.random.uniform(low=radius_min, high=radius_max) 
while i < num_views:

    # Determine point of interest in scene as the object closest to the mean of a subset of objects
    poi = bproc.object.compute_poi(np.random.choice(target_objects, 
                                    size=min(len(source_ids), 5)))

    # # Sample location --> for pose estimation
    # location = bproc.sampler.shell(center = [0, 0, 0],
    #                             radius_min = 0.8,
    #                             radius_max = 1.1,
    #                             elevation_min = 5,
    #                             elevation_max = 80,
    #                             uniform_volume = False)

    # # Compute rotation based on vector going from location towards poi
    # rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
    noise =  np.random.randn() * (_radius / 10) if mode == 'train' else 0
    inplane_rot = np.random.uniform(-0.7854, 0.7854) if mode == 'train' else 0

    radius = _radius + noise
    # Sample on sphere around ShapeNet object
    location = bproc.sampler.part_sphere(poi, radius=radius, dist_above_center=0.05, mode="SURFACE") # dist_above_center: hight of the ring


    # Compute rotation based on vector going from location towards ShapeNet object
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=inplane_rot)

    # Add homog cam pose based on location an rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    
    # Check that obstacles are at least 0.5 meter away from the camera and make sure the view interesting enough
    if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.5}, bop_bvh_tree):
        # Persist camera pose
        bproc.camera.add_camera_pose(cam2world_matrix)
        i += 1


# render the whole pipeline
bproc.renderer.enable_depth_output(activate_antialiasing=False)
# bproc.renderer.enable_normals_output()
bproc.renderer.set_max_amount_of_samples(50)
bproc.renderer.enable_segmentation_output(map_by=["instance", "category_id"])

data = bproc.renderer.render()
# postprocess depth using the kinect azure noise model
data["depth_kinect"] = bproc.postprocessing.add_kinect_azure_noise(data["depth"], data["colors"], 10)


# write_coco_annotations(out_dir,
#                         instance_segmaps=data["instance_segmaps"],
#                         instance_attribute_maps=data["instance_attribute_maps"],
#                         colors=data["colors"],
#                         color_file_format="JPEG")
# print(len(data['instance_attribute_maps']))
# Might need to change out_dir path every re-run_myenv
write_my(out_dir, 
        chunk_name=mode+'_pbr',
        dataset='',
        target_objects=target_objects,
        depths = data["depth"],
        depths_noise = data["depth_kinect"],
        colors = data["colors"], 
        instance_masks=data['instance_segmaps'],
        category_masks=data['category_id_segmaps'],
        instance_attribute_maps=data["instance_attribute_maps"],
        color_file_format = "JPEG",
        ignore_dist_thres = 10,
        frames_per_chunk=num_views)
        