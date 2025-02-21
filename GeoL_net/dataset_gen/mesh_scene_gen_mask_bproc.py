import blenderproc as bproc
import open3d as o3d
import numpy as np
import glob
import json
import os
import math
from blenderproc.python.writer.MyWriterUtility import write_my, write_my_zhoy
from blenderproc.python.writer.CocoWriterUtility import write_coco_annotations
import argparse
import time
from tqdm import tqdm
import random
import math



""""
    almost same as the mesh_scene_gen_bproc.py
    the only difference is generating a mask.json containing the bbox parameters of obj.
    json to RGBD(mask)

"""
def calculate_distance(pos1, pos2):
    """计算两个点之间的欧几里得距离"""
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

def key2phrase(key:str):
    key_parts = key.split('/')
    obj_descpt = key_parts[-2]
    parts = obj_descpt.split('_')
    obj_name = '_'.join(key_parts[3:-2]) # pay attention
    obj_descpt = parts[-1]
    phrase = f"the {obj_descpt} {obj_name}"
    return phrase

def determine_direction(p1, p2):
    x_diff = p2[0] - p1[0]
    y_diff = p2[1] - p1[1]
    
    angle = math.degrees(math.atan2(y_diff, x_diff))
    if angle < 0:
        angle += 360
    
    if 0 <= angle < 22.5 or 337.5 <= angle < 360:
        return "Left"
    elif 22.5 <= angle < 67.5:
        return "Front Left"
    elif 67.5 <= angle < 112.5:
        return "Front"
    elif 112.5 <= angle < 157.5:
        return "Front Right"
    elif 157.5 <= angle < 202.5:
        return "Right"
    elif 202.5 <= angle < 247.5:
        return "Right Behind"
    elif 247.5 <= angle < 292.5:
        return "Behind"
    elif 292.5 <= angle < 337.5:
        return "Left Behind"

def bproc_gen_mask(scene_mesh_json, RGBD_out_dir, removed_obj = None):
    "still keep the removed obj but mask it"

    # read some parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', nargs='?', help="Path to the configuration files")
    min_dist = {'train': 0.05, 'val': 0.20, 'test': 0.20}
    max_dist = {'train': 0.20, 'val': 0.25, 'test': 0.25}
    num_objects = {'train': 2, 'val': 2, 'test':2}
    num_objects_distract = {'train': 0, 'val': 0, 'test':0}
    #catId2name = {2:'Bowl', 6: 'Mug', 1: 'Bottle'}
    #catId2dir  = {2:'02880940', 6: '03797390', 1:'02876657'}
    all_samples = {}
    num_views = 50


    json_file_path = scene_mesh_json
    with open(json_file_path, 'r') as f:
        data = json.load(f)


    # slurm
    mode = 'train'
    cfg = dict(H=480, W=640, K=[591.0125, 0, 322.525, 0, 590.16775, 244.11084, 0, 0, 1], 
            cc_textures_path='resources'
            )

    K = np.array(cfg['K']).reshape(3, 3)
    num_target_objects_perclass = num_objects[mode]
    num_distractor_objects_perclass = num_objects_distract[mode]

    #bproc.init()
    bproc.camera.set_intrinsics_from_K_matrix(K,  cfg['W'],  cfg['H'])

    # generate whole scene in blenderproc
    table_height = 0
    #print("len:", len(data))
    obj_amount = len(data)
    exist_obj_amount = 0
    x_min = 100
    x_max = -100
    y_min = 100
    y_max = -100
    target_objects = []

    for obj_file_name, obj_pose in data.items():
        exist_obj_amount+=1
        if exist_obj_amount == obj_amount:
            obj_mesh = obj_mesh = bproc.loader.load_obj(obj_file_name)
            desk = obj_mesh[0]
            desk.blender_obj.rotation_euler = (0,0,0)  
            desk.blender_obj.scale = (obj_pose[1][0],obj_pose[1][1], obj_pose[1][2])
            bbox = desk.get_bound_box()
            z_height = bbox[1][2] - bbox[0][2]
            x_width = bbox[4][0]-bbox[3][0]
            y_width = bbox[2][1]-bbox[0][1]
            desk.blender_obj.location = ((x_min+x_max)/2, (y_min+y_max)/2,0 - z_height/2)

            desk_mat = desk.get_materials()[0]
            desk_mat.set_principled_shader_value("Roughness", np.random.uniform(0.8, 1.0))
            desk_mat.set_principled_shader_value("Specular", np.random.uniform(0.8, 1.0))

            #set color
            material = desk.get_materials()[0]
            #material.set_principled_shader_value("Base Color", [0.0, 0.0, 0.0, 0.0])
            #material.make_emissive(emission_strength=0, emission_color=[1, 1, 1, 1.0])
            desk.replace_materials(material)

            desk.set_cp("category_id", 2) # default 0 
            desk.set_cp("scene_id", 2) #default 0

            target_objects.append(desk)
                # Update JSON data with bounding box parameters

            break
        obj_mesh = bproc.loader.load_obj(obj_file_name)
        #print(f"{obj_file_name}")
        bbox = obj_mesh[0].get_bound_box()
        bbox_center = np.mean(bbox, axis=0)
        #print(bbox_center)
        z_angle = math.radians(obj_pose[2])
        obj_mesh[0].blender_obj.rotation_euler = (0,0,z_angle)


            
        obj_mesh[0].blender_obj.scale = (obj_pose[1][0], obj_pose[1][1], obj_pose[1][2]) 
        bbox = obj_mesh[0].get_bound_box()
        z_height = bbox[1][2] - bbox[0][2]
        x_width = bbox[4][0]-bbox[3][0]
        y_width = bbox[2][1]-bbox[0][1]

        obj_mesh[0].blender_obj.location = (obj_pose[0][0],obj_pose[0][1],0+z_height/2)
        mass, fiction_coeff = (0.4, 0.5)
        obj_mesh[0].enable_rigidbody(True, mass=mass, friction=mass * fiction_coeff, 
        linear_damping = 1.99, angular_damping = 0, collision_margin=0.0001)
        obj_mesh[0].set_shading_mode('auto')

        # set material    
        mat = obj_mesh[0].get_materials()[0]
        mat.set_principled_shader_value("Roughness", np.random.uniform(0.8, 1.0))
        mat.set_principled_shader_value("Specular", np.random.uniform(0.8, 1.0))
        #mat.set_principled_shader_value("Base Color", [0.0, 0.0, 1.0, 1.0])

        #print(obj_file_name)
        if removed_obj in obj_file_name:
            #print("++++++++++++++++++")
            obj_mesh[0].set_cp("category_id", 4) # default 0 
            obj_mesh[0].set_cp("scene_id", 4) #default 0
        else:
            obj_mesh[0].set_cp("category_id", 1) # default 0 
            obj_mesh[0].set_cp("scene_id", 1) #default 0


   

        target_objects.append(obj_mesh[0])


        # update x min max y min max
        bbox = obj_mesh[0].get_bound_box()
        if x_min > bbox[0][0]:
            x_min = bbox[0][0]
        if x_max < bbox[6][0]:
            x_max = bbox[6][0]
        if y_min > bbox[0][1]:
            y_min = bbox[0][1]
        if y_max < bbox[6][1]:
            y_max = bbox[6][1]
        
    with open(json_file_path, 'w') as f:
        json.dump(data, f, indent=4)

    # create room
    room_coeff = 10
    room = [bproc.object.create_primitive('PLANE', scale=[room_coeff, room_coeff, 1], location=[0, 0, obj_pose[0][2] - z_height]),
            bproc.object.create_primitive('PLANE', scale=[room_coeff, room_coeff, 1], location=[0, -room_coeff, room_coeff + obj_pose[0][2] - z_height], rotation=[-1.570796, 0, 0]),
            bproc.object.create_primitive('PLANE', scale=[room_coeff, room_coeff, 1], location=[0, room_coeff, room_coeff + obj_pose[0][2] - z_height], rotation=[1.570796, 0, 0]),
            bproc.object.create_primitive('PLANE', scale=[room_coeff, room_coeff, 1], location=[room_coeff, 0, room_coeff + obj_pose[0][2] - z_height], rotation=[0, -1.570796, 0]),
            bproc.object.create_primitive('PLANE', scale=[room_coeff, room_coeff, 1], location=[-room_coeff, 0, room_coeff+obj_pose[0][2] - z_height], rotation=[0, 1.570796, 0])]

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
        plane.set_cp('category_id', 3)
    

    # sample point light on shell
    '''
    print("-----start setting light point ------")
    light_point = bproc.types.Light()
    light_point.set_energy(np.random.uniform(200,250))
    # lights = [[0,0,1], [1,0,0], [0,1,0]] debug only
    light_point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))
    location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 1, radius_max = 1.5,
                            elevation_min = 5, elevation_max = 89, uniform_volume = False)
    location = [(x_min+x_max)/2 - 1, (y_min+y_max)/2,obj_pose[0][2] + z_height/2]
    light_point.set_location(location)
    '''
    # setup camera
    #print("-----start setting camera ------")
    i = 0
    radius_min, radius_max = (1.2, 1.5)
    _radius = np.random.uniform(low=radius_min, high=radius_max) 
    num_views= 1

    while i < num_views:

        poi = bproc.object.compute_poi(np.random.choice(target_objects, size=5))

        noise =  np.random.randn() * (_radius / 10) 
        inplane_rot = np.random.uniform(-0.7854, 0.7854) 

        radius = _radius + noise
        # Sample on sphere around ShapeNet object
        location = bproc.sampler.part_sphere(poi, radius=radius, dist_above_center=0.05, mode="SURFACE") # dist_above_center: hight of the ring


        # Compute rotation based on vector going from location towards ShapeNet object
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=inplane_rot)

        # Add homog cam pose based on location an rotation
        #cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        rotation_matrix = [[1, 0, 0], [0,0.8,-0.6], [0,0.6,0.8]]
        cam2world_matrix = bproc.math.build_transformation_mat([(x_min+x_max)/2, (y_min+y_max)/2 - 1, 1], rotation_matrix)
        print(location, rotation_matrix)
        bproc.camera.add_camera_pose(cam2world_matrix)
        i += 1

        
    # render the whole pipeline
    # print("-----start rendering ------")
    try:
        bproc.renderer.enable_depth_output(activate_antialiasing=False)
    except RuntimeError:
        pass
    bproc.renderer.set_max_amount_of_samples(num_views)
    bproc.renderer.enable_segmentation_output(map_by=["instance", "category_id"], default_values={'category_id': 0})
    data = bproc.renderer.render()

    data["depth_kinect"] = bproc.postprocessing.add_kinect_azure_noise(data["depth"], data["colors"], 10)
    #print(data)
    # Save the rendered images
    out_dir = RGBD_out_dir
    #print(data['instance_segmaps'])
    # print(len(data['category_id_segmaps']))
    '''
    write_my_zhoy(out_dir, 
            chunk_name='test_pbr',
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
    '''
    write_my(out_dir, 
        chunk_name='test_pbr',
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
        frames_per_chunk=num_views,
        is_shapenet=False)
        
    bproc.writer.write_hdf5(out_dir, data)

def bproc_gen_mask_removw_obj(scene_mesh_json, RGBD_out_dir, removed_obj = None):
    "remove the obj"

    # read some parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', nargs='?', help="Path to the configuration files")
    min_dist = {'train': 0.05, 'val': 0.20, 'test': 0.20}
    max_dist = {'train': 0.20, 'val': 0.25, 'test': 0.25}
    num_objects = {'train': 2, 'val': 2, 'test':2}
    num_objects_distract = {'train': 0, 'val': 0, 'test':0}
    #catId2name = {2:'Bowl', 6: 'Mug', 1: 'Bottle'}
    #catId2dir  = {2:'02880940', 6: '03797390', 1:'02876657'}
    all_samples = {}
    num_views = 50


    json_file_path = scene_mesh_json
    with open(json_file_path, 'r') as f:
        data = json.load(f)


    # slurm
    mode = 'train'
    cfg = dict(H=480, W=640, K=[591.0125, 0, 322.525, 0, 590.16775, 244.11084, 0, 0, 1], 
            cc_textures_path='resources'
            )

    K = np.array(cfg['K']).reshape(3, 3)
    num_target_objects_perclass = num_objects[mode]
    num_distractor_objects_perclass = num_objects_distract[mode]


    #bproc.init()
    bproc.camera.set_intrinsics_from_K_matrix(K,  cfg['W'],  cfg['H'])

    # generate whole scene in blenderproc
    table_height = 0
    # print("len:", len(data))
    obj_amount = len(data)
    exist_obj_amount = 0
    x_min = 100
    x_max = -100
    y_min = 100
    y_max = -100
    target_objects = []

    for obj_file_name, obj_pose in data.items():
        exist_obj_amount+=1
        if exist_obj_amount == obj_amount:
            obj_mesh = obj_mesh = bproc.loader.load_obj(obj_file_name)
            desk = obj_mesh[0]
            desk.blender_obj.rotation_euler = (0,0,0)  
            desk.blender_obj.scale = (obj_pose[1][0],obj_pose[1][1], obj_pose[1][2])
            bbox = desk.get_bound_box()
            z_height = bbox[1][2] - bbox[0][2]
            x_width = bbox[4][0]-bbox[3][0]
            y_width = bbox[2][1]-bbox[0][1]
            desk.blender_obj.location = ((x_min+x_max)/2, (y_min+y_max)/2,0 - z_height/2)

            desk_mat = desk.get_materials()[0]
            desk_mat.set_principled_shader_value("Roughness", np.random.uniform(0.8, 1.0))
            desk_mat.set_principled_shader_value("Specular", np.random.uniform(0.8, 1.0))

            #set color
            material = desk.get_materials()[0]
            #material.set_principled_shader_value("Base Color", [0.0, 0.0, 0.0, 0.0])
            #material.make_emissive(emission_strength=0, emission_color=[1, 1, 1, 1.0])
            desk.replace_materials(material)

            desk.set_cp("category_id", 2) # default 0 
            desk.set_cp("scene_id", 2) #default 0

            target_objects.append(desk)
                # Update JSON data with bounding box parameters

            break

        if removed_obj in obj_file_name:
            continue

        obj_mesh = bproc.loader.load_obj(obj_file_name)
        z_rotation = math.radians(obj_pose[2])
        obj_mesh[0].blender_obj.rotation_euler = (0, 0, z_rotation)
         
        obj_mesh[0].blender_obj.scale = (obj_pose[1][0], obj_pose[1][1], obj_pose[1][2]) 
        bbox = obj_mesh[0].get_bound_box()
        z_height = bbox[1][2] - bbox[0][2]
        x_width = bbox[4][0]-bbox[3][0]
        y_width = bbox[2][1]-bbox[0][1]
        #obj_mesh[0].blender_obj.location = (obj_pose[0][0]-x_width/2,obj_pose[0][1]-y_width/2,obj_pose[0][2]+z_height/2)
        #print(f"{obj_file_name} oringal position: {obj_mesh[0].blender_obj.location}")
        obj_mesh[0].blender_obj.location = (obj_pose[0][0],obj_pose[0][1],0+z_height/2)
        #obj_mesh[0].blender_obj.location= (0,0,obj_pose[0][2]+z_height/2)
        #print(f"{obj_file_name} new position: {obj_mesh[0].blender_obj.location}")
        mass, fiction_coeff = (0.4, 0.5)
        obj_mesh[0].enable_rigidbody(True, mass=mass, friction=mass * fiction_coeff, 
        linear_damping = 1.99, angular_damping = 0, collision_margin=0.0001)
        obj_mesh[0].set_shading_mode('auto')

        # set material    
        mat = obj_mesh[0].get_materials()[0]
        mat.set_principled_shader_value("Roughness", np.random.uniform(0.8, 1.0))
        mat.set_principled_shader_value("Specular", np.random.uniform(0.8, 1.0))
        #mat.set_principled_shader_value("Base Color", [0.0, 0.0, 1.0, 1.0])

        # print(obj_file_name)
        if removed_obj in obj_file_name:
            print("++++++++++++++++++")
            obj_mesh[0].set_cp("category_id", 4) # default 0 
            obj_mesh[0].set_cp("scene_id", 4) #default 0
        else:
            obj_mesh[0].set_cp("category_id", 1) # default 0 
            obj_mesh[0].set_cp("scene_id", 1) #default 0


   

        target_objects.append(obj_mesh[0])


        # update x min max y min max
        bbox = obj_mesh[0].get_bound_box()
        if x_min > bbox[0][0]:
            x_min = bbox[0][0]
        if x_max < bbox[6][0]:
            x_max = bbox[6][0]
        if y_min > bbox[0][1]:
            y_min = bbox[0][1]
        if y_max < bbox[6][1]:
            y_max = bbox[6][1]
        
    with open(json_file_path, 'w') as f:
        json.dump(data, f, indent=4)

    # create room
    room_coeff = 10
    room = [bproc.object.create_primitive('PLANE', scale=[room_coeff, room_coeff, 1], location=[0, 0, obj_pose[0][2] - z_height]),
            bproc.object.create_primitive('PLANE', scale=[room_coeff, room_coeff, 1], location=[0, -room_coeff, room_coeff + obj_pose[0][2] - z_height], rotation=[-1.570796, 0, 0]),
            bproc.object.create_primitive('PLANE', scale=[room_coeff, room_coeff, 1], location=[0, room_coeff, room_coeff + obj_pose[0][2] - z_height], rotation=[1.570796, 0, 0]),
            bproc.object.create_primitive('PLANE', scale=[room_coeff, room_coeff, 1], location=[room_coeff, 0, room_coeff + obj_pose[0][2] - z_height], rotation=[0, -1.570796, 0]),
            bproc.object.create_primitive('PLANE', scale=[room_coeff, room_coeff, 1], location=[-room_coeff, 0, room_coeff+obj_pose[0][2] - z_height], rotation=[0, 1.570796, 0])]

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
        plane.set_cp('category_id', 3)
    

    # sample point light on shell
    '''
    print("-----start setting light point ------")
    light_point = bproc.types.Light()
    light_point.set_energy(np.random.uniform(200,250))
    # lights = [[0,0,1], [1,0,0], [0,1,0]] debug only
    light_point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))
    location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 1, radius_max = 1.5,
                            elevation_min = 5, elevation_max = 89, uniform_volume = False)
    location = [(x_min+x_max)/2 - 1, (y_min+y_max)/2,obj_pose[0][2] + z_height/2]
    light_point.set_location(location)
    '''
    # setup camera
    # print("-----start setting camera ------")
    i = 0
    radius_min, radius_max = (1.2, 1.5)
    _radius = np.random.uniform(low=radius_min, high=radius_max) 
    num_views= 1

    while i < num_views:

        poi = bproc.object.compute_poi(np.random.choice(target_objects, size=5))

        noise =  np.random.randn() * (_radius / 10) 
        inplane_rot = np.random.uniform(-0.7854, 0.7854) 

        radius = _radius + noise
        # Sample on sphere around ShapeNet object
        location = bproc.sampler.part_sphere(poi, radius=radius, dist_above_center=0.05, mode="SURFACE") # dist_above_center: hight of the ring


        # Compute rotation based on vector going from location towards ShapeNet object
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=inplane_rot)

        # Add homog cam pose based on location an rotation
        #cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        rotation_matrix = [[1, 0, 0], [0,0.8,-0.6], [0,0.6,0.8]]
        cam2world_matrix = bproc.math.build_transformation_mat([(x_min+x_max)/2, (y_min+y_max)/2 - 1, 1], rotation_matrix)
        # print(location, rotation_matrix)
        bproc.camera.add_camera_pose(cam2world_matrix)
        i += 1

        
    # render the whole pipeline
    # print("-----start rendering ------")
    
    #bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.set_max_amount_of_samples(num_views)
    bproc.renderer.enable_segmentation_output(map_by=["instance", "category_id"])
    bproc.renderer.enable_segmentation_output(map_by=["instance", "category_id"], default_values={'category_id': 0})
    data = bproc.renderer.render()

    data["depth_kinect"] = bproc.postprocessing.add_kinect_azure_noise(data["depth"], data["colors"], 10)

    # Save the rendered images
    out_dir = RGBD_out_dir
    # print(data['instance_segmaps'])
    # print(len(data['category_id_segmaps']))
    '''
    write_my_zhoy(out_dir, 
            chunk_name='test_pbr',
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
    '''
    write_my(out_dir, 
        chunk_name='test_pbr',
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
        frames_per_chunk=num_views,
        is_shapenet=False)
        
    bproc.writer.write_hdf5(out_dir, data)

def bproc_gen_mask_with_and_without_obj(scene_mesh_json, RGBD_out_dir, removed_obj=None):
    "Render both the scene with and without the removed object."

    # Step 1: 加载场景数据
    json_file_path = scene_mesh_json
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    mode = 'train'
    cfg = dict(
        H=480, 
        W=640, 
        K=[591.0125, 0, 322.525, 0, 590.16775, 244.11084, 0, 0, 1], 
        cc_textures_path='resources'
    )
    K = np.array(cfg['K']).reshape(3, 3)

    # Step 2: 定义相机内参
    bproc.camera.set_intrinsics_from_K_matrix(K, cfg['W'], cfg['H'])

    target_objects = []
    x_min, x_max, y_min, y_max = 100, -100, 100, -100
    exist_obj_amount = 0
    obj_amount = len(data)

    # Step 3: 一次性加载所有对象
    for obj_file_name, obj_pose in data.items():
        exist_obj_amount += 1
        if exist_obj_amount == obj_amount:
            # 最后一个对象，假设为桌面对象
            obj_mesh = bproc.loader.load_obj(obj_file_name)
            desk = obj_mesh[0]
            desk.blender_obj.rotation_euler = (0, 0, 0)
            desk.blender_obj.scale = (obj_pose[1][0], obj_pose[1][1], obj_pose[1][2])
            bbox = desk.get_bound_box()
            z_height = bbox[1][2] - bbox[0][2]
            desk.blender_obj.location = ((x_min + x_max) / 2, (y_min + y_max) / 2, 0 - z_height / 2)

            desk_mat = desk.get_materials()[0]
            desk_mat.set_principled_shader_value("Roughness", np.random.uniform(0.8, 1.0))
            desk_mat.set_principled_shader_value("Specular", np.random.uniform(0.8, 1.0))

            material = desk.get_materials()[0]
            desk.replace_materials(material)

            desk.set_cp("category_id", 2)
            desk.set_cp("scene_id", 2)

            target_objects.append(desk)
            break

        # 加载每个对象
        obj_mesh = bproc.loader.load_obj(obj_file_name)
        z_rotation = math.radians(obj_pose[2])
        obj_mesh[0].blender_obj.rotation_euler = (0, 0, z_rotation)
        obj_mesh[0].blender_obj.scale = (obj_pose[1][0], obj_pose[1][1], obj_pose[1][2])
        obj_mesh[0].set_name(obj_file_name)
        bbox = obj_mesh[0].get_bound_box()
        z_height = bbox[1][2] - bbox[0][2]
        obj_mesh[0].blender_obj.location = (obj_pose[0][0], obj_pose[0][1], 0 + z_height / 2)

        # 设置材质
        mat = obj_mesh[0].get_materials()[0]
        mat.set_principled_shader_value("Roughness", np.random.uniform(0.8, 1.0))
        mat.set_principled_shader_value("Specular", np.random.uniform(0.8, 1.0))

        # 设置对象属性
        if removed_obj in obj_file_name:
            obj_mesh[0].set_cp("category_id", 4)
            obj_mesh[0].set_cp("scene_id", 4)
        else:
            obj_mesh[0].set_cp("category_id", 1)
            obj_mesh[0].set_cp("scene_id", 1)

        target_objects.append(obj_mesh[0])

        # 更新场景边界
        x_min = min(x_min, bbox[0][0])
        x_max = max(x_max, bbox[6][0])
        y_min = min(y_min, bbox[0][1])
        y_max = max(y_max, bbox[6][1])

    # Step 4: 创建房间和光源
    room_coeff = 10
    room = [bproc.object.create_primitive('PLANE', scale=[room_coeff, room_coeff, 1], location=[0, 0, obj_pose[0][2] - z_height]),
            bproc.object.create_primitive('PLANE', scale=[room_coeff, room_coeff, 1], location=[0, -room_coeff, room_coeff + obj_pose[0][2] - z_height], rotation=[-1.570796, 0, 0]),
            bproc.object.create_primitive('PLANE', scale=[room_coeff, room_coeff, 1], location=[0, room_coeff, room_coeff + obj_pose[0][2] - z_height], rotation=[1.570796, 0, 0]),
            bproc.object.create_primitive('PLANE', scale=[room_coeff, room_coeff, 1], location=[room_coeff, 0, room_coeff + obj_pose[0][2] - z_height], rotation=[0, -1.570796, 0]),
            bproc.object.create_primitive('PLANE', scale=[room_coeff, room_coeff, 1], location=[-room_coeff, 0, room_coeff+obj_pose[0][2] - z_height], rotation=[0, 1.570796, 0])]

    # sample point light on shell
    light_plane = bproc.object.create_primitive('SPHERE', scale=[1, 1, 1], location=[-1, -1, 2])
    light_plane.set_name('light_point')
    light_plane_material = bproc.material.create('light_material')
    light_plane_material.make_emissive(emission_strength=np.random.uniform(3,6), emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))     
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
        plane.set_cp('category_id', 3)

    # Step 5: 设置相机视角
    i = 0
    radius_min, radius_max = (1.2, 1.5)
    _radius = np.random.uniform(low=radius_min, high=radius_max) 
    num_views= 1

    while i < num_views:

        poi = bproc.object.compute_poi(np.random.choice(target_objects, size=5))

        noise =  np.random.randn() * (_radius / 10) 
        inplane_rot = np.random.uniform(-0.7854, 0.7854) 

        radius = _radius + noise
        # Sample on sphere around ShapeNet object
        location = bproc.sampler.part_sphere(poi, radius=radius, dist_above_center=0.05, mode="SURFACE") # dist_above_center: hight of the ring


        # Compute rotation based on vector going from location towards ShapeNet object
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=inplane_rot)

        # Add homog cam pose based on location an rotation
        #cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        rotation_matrix = [[1, 0, 0], [0,0.8,-0.6], [0,0.6,0.8]]
        cam2world_matrix = bproc.math.build_transformation_mat([(x_min+x_max)/2, (y_min+y_max)/2 - 1, 1], rotation_matrix)
        # print(location, rotation_matrix)
        bproc.camera.add_camera_pose(cam2world_matrix)
        i += 1

    # Step 6: 渲染包含和不包含 removed_obj 的场景
    for remove_flag in [False, True]:
        # 控制 removed_obj 的可见性
        for obj in target_objects:
            print(f"-------{obj.get_name()}-----")
            if removed_obj and removed_obj in obj.get_name():
                obj.blender_obj.hide_render = remove_flag  # True: 不渲染该对象, False: 渲染该对象
                print(f"-------not rendering {obj.get_name()}-----")
            else:
                obj.blender_obj.hide_render = False  # 其他对象始终渲染
        try:
            bproc.renderer.enable_depth_output(activate_antialiasing=False)
        except RuntimeError:
            pass
        bproc.renderer.set_max_amount_of_samples(1)
        bproc.renderer.enable_segmentation_output(map_by=["instance", "category_id"])
        data = bproc.renderer.render()

        data["depth_kinect"] = bproc.postprocessing.add_kinect_azure_noise(data["depth"], data["colors"], 10)
        # Step 7: 保存渲染结果
        out_dir = RGBD_out_dir
        if remove_flag:
            out_dir = os.path.join(RGBD_out_dir, "no_obj")
        else:
            out_dir = os.path.join(RGBD_out_dir, "with_obj")

        # 保存文件
        write_my(out_dir,
            chunk_name='test_pbr',
            dataset='',
            target_objects=target_objects,
            depths=data["depth"],
            depths_noise=data["depth_kinect"],
            colors=data["colors"],
            instance_masks=data['instance_segmaps'],
            category_masks=data['category_id_segmaps'],
            instance_attribute_maps=data["instance_attribute_maps"],
            color_file_format="JPEG",
            ignore_dist_thres=10,
            frames_per_chunk=1,
            is_shapenet=False
        )

        bproc.writer.write_hdf5(out_dir, data)


def bproc_gen_mask_with_and_without_obj_sequnce(data, RGBD_out_dir, removed_obj=None):
    "Render both the scene with and without the removed object."

    # Step 1: 加载场景数据
    data = data

    mode = 'train'
    cfg = dict(
        H=480, 
        W=640, 
        K=[591.0125, 0, 322.525, 0, 590.16775, 244.11084, 0, 0, 1], 
        cc_textures_path='resources'
    )
    K = np.array(cfg['K']).reshape(3, 3)

    # Step 2: 定义相机内参
    bproc.camera.set_intrinsics_from_K_matrix(K, cfg['W'], cfg['H'])

    target_objects = []
    x_min, x_max, y_min, y_max = 100, -100, 100, -100
    exist_obj_amount = 0
    obj_amount = len(data)

    # Step 3: 一次性加载所有对象
    for obj_file_name, obj_pose in data.items():
        exist_obj_amount += 1
        if exist_obj_amount == obj_amount:
            # 最后一个对象，假设为桌面对象
            obj_mesh = bproc.loader.load_obj(obj_file_name)
            desk = obj_mesh[0]
            desk.blender_obj.rotation_euler = (0, 0, 0)
            desk.blender_obj.scale = (obj_pose[1][0], obj_pose[1][1], obj_pose[1][2])
            bbox = desk.get_bound_box()
            z_height = bbox[1][2] - bbox[0][2]
            desk.blender_obj.location = ((x_min + x_max) / 2, (y_min + y_max) / 2, 0 - z_height / 2)

            desk_mat = desk.get_materials()[0]
            desk_mat.set_principled_shader_value("Roughness", 0.9)
            desk_mat.set_principled_shader_value("Specular", 0.9)

            material = desk.get_materials()[0]
            desk.replace_materials(material)

            desk.set_cp("category_id", 2)
            desk.set_cp("scene_id", 2)

            target_objects.append(desk)
            break

        # 加载每个对象
        obj_mesh = bproc.loader.load_obj(obj_file_name)
        z_rotation = math.radians(obj_pose[2])
        obj_mesh[0].blender_obj.rotation_euler = (0, 0, z_rotation)
        obj_mesh[0].blender_obj.scale = (obj_pose[1][0], obj_pose[1][1], obj_pose[1][2])
        obj_mesh[0].set_name(obj_file_name)
        bbox = obj_mesh[0].get_bound_box()
        z_height = bbox[1][2] - bbox[0][2]
        obj_mesh[0].blender_obj.location = (obj_pose[0][0], obj_pose[0][1], 0 + z_height / 2)

        # 设置材质
        mat = obj_mesh[0].get_materials()[0]
        mat.set_principled_shader_value("Roughness", 0.9)
        mat.set_principled_shader_value("Specular", 0.9)

        # 设置对象属性
        if removed_obj in obj_file_name:
            obj_mesh[0].set_cp("category_id", 4)
            obj_mesh[0].set_cp("scene_id", 4)
        else:
            obj_mesh[0].set_cp("category_id", 1)
            obj_mesh[0].set_cp("scene_id", 1)

        target_objects.append(obj_mesh[0])

        # 更新场景边界
        x_min = min(x_min, bbox[0][0])
        x_max = max(x_max, bbox[6][0])
        y_min = min(y_min, bbox[0][1])
        y_max = max(y_max, bbox[6][1])

    # Step 4: 创建房间和光源
    room_coeff = 10
    room = [bproc.object.create_primitive('PLANE', scale=[room_coeff, room_coeff, 1], location=[0, 0, obj_pose[0][2] - z_height]),
            bproc.object.create_primitive('PLANE', scale=[room_coeff, room_coeff, 1], location=[0, -room_coeff, room_coeff + obj_pose[0][2] - z_height], rotation=[-1.570796, 0, 0]),
            bproc.object.create_primitive('PLANE', scale=[room_coeff, room_coeff, 1], location=[0, room_coeff, room_coeff + obj_pose[0][2] - z_height], rotation=[1.570796, 0, 0]),
            bproc.object.create_primitive('PLANE', scale=[room_coeff, room_coeff, 1], location=[room_coeff, 0, room_coeff + obj_pose[0][2] - z_height], rotation=[0, -1.570796, 0]),
            bproc.object.create_primitive('PLANE', scale=[room_coeff, room_coeff, 1], location=[-room_coeff, 0, room_coeff+obj_pose[0][2] - z_height], rotation=[0, 1.570796, 0])]

    # sample point light on shell
    light_plane = bproc.object.create_primitive('SPHERE', scale=[1, 1, 1], location=[-1, -1, 2])
    light_plane.set_name('light_point')
    light_plane_material = bproc.material.create('light_material')
    #light_plane_material.make_emissive(emission_strength=np.random.uniform(3,6), emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))     
    light_plane_material.make_emissive(emission_strength=4.5, emission_color=[0.75, 0.75, 0.75, 1.0])   
    light_plane.replace_materials(light_plane_material)

    # sample CC Texture and assign to room planes
    
    if cfg['cc_textures_path'] is not None:
        cc_textures = bproc.loader.load_ccmaterials(cfg['cc_textures_path'])
        for plane in room:
            #random_cc_texture = np.random.choice(cc_textures)
            random_cc_texture = cc_textures[0]
            plane.replace_materials(random_cc_texture)
    
    # set attributes
    room.append(light_plane)
    for plane in room:
        plane.set_cp('category_id', 3)

    # Step 5: 设置相机视角
    i = 0
    radius_min, radius_max = (1.2, 1.5)
    _radius = np.random.uniform(low=radius_min, high=radius_max) 
    num_views= 1

    while i < num_views:

        poi = bproc.object.compute_poi(np.random.choice(target_objects, size=5))

        noise =  np.random.randn() * (_radius / 10) 
        inplane_rot = np.random.uniform(-0.7854, 0.7854) 

        radius = _radius + noise
        # Sample on sphere around ShapeNet object
        location = bproc.sampler.part_sphere(poi, radius=radius, dist_above_center=0.05, mode="SURFACE") # dist_above_center: hight of the ring


        # Compute rotation based on vector going from location towards ShapeNet object
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=inplane_rot)

        # Add homog cam pose based on location an rotation
        #cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        rotation_matrix = [[1, 0, 0], [0,0.8,-0.6], [0,0.6,0.8]]
        cam2world_matrix = bproc.math.build_transformation_mat([(x_min+x_max)/2, (y_min+y_max)/2 - 1, 1], rotation_matrix)
        # print(location, rotation_matrix)
        bproc.camera.add_camera_pose(cam2world_matrix)
        i += 1

    # Step 6: 渲染包含和不包含 removed_obj 的场景
    for remove_flag in [False, True]:
        # 控制 removed_obj 的可见性
        for obj in target_objects:
            print(f"-------{obj.get_name()}-----")
            if removed_obj and removed_obj in obj.get_name():
                obj.blender_obj.hide_render = remove_flag  # True: 不渲染该对象, False: 渲染该对象
                print(f"-------not rendering {obj.get_name()}-----")
            else:
                obj.blender_obj.hide_render = False  # 其他对象始终渲染
        try:
            bproc.renderer.enable_depth_output(activate_antialiasing=False)
        except RuntimeError:
            pass
        bproc.renderer.set_max_amount_of_samples(1)
        bproc.renderer.enable_segmentation_output(map_by=["instance", "category_id"])
        data = bproc.renderer.render()

        data["depth_kinect"] = bproc.postprocessing.add_kinect_azure_noise(data["depth"], data["colors"], 10)
        # Step 7: 保存渲染结果
        out_dir = RGBD_out_dir
        if remove_flag:
            out_dir = os.path.join(RGBD_out_dir, "no_obj")
        else:
            out_dir = os.path.join(RGBD_out_dir, "with_obj")

        # 保存文件
        write_my(out_dir,
            chunk_name='test_pbr',
            dataset='',
            target_objects=target_objects,
            depths=data["depth"],
            depths_noise=data["depth_kinect"],
            colors=data["colors"],
            instance_masks=data['instance_segmaps'],
            category_masks=data['category_id_segmaps'],
            instance_attribute_maps=data["instance_attribute_maps"],
            color_file_format="JPEG",
            ignore_dist_thres=10,
            frames_per_chunk=1,
            is_shapenet=False
        )

        bproc.writer.write_hdf5(out_dir, data)
    
def objs_keep_in_scene(json_file, number_of_objects):
    scene_id = json_file.split("/")[-1].split(".")[0]
    new_json_file = os.path.join("dataset/scene_RGBD_mask_sequence",scene_id, "text_guidance.json")

        

    obj_keep_list = []
    
    # 打开JSON文件
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 创建用于分类的列表
    high_priority = []
    low_priority = []

    # 遍历JSON文件中的所有物体
    for obj_file_name, obj_pose in data.items():
        if "desk" in obj_file_name or "table" in obj_file_name:
            obj_keep_list.append(obj_file_name)
        if "monitor" in obj_file_name or "laptop" in obj_file_name:
            # 优先保留 monitor 和 laptop
            high_priority.append((obj_file_name, obj_pose[0]))  # 存储物体名和位置信息
        elif any(x in obj_file_name for x in ["pencil", "eraser", "clock"]):
            # 不保留 pencil, eraser, clock
            continue
        else:
            # 其他物体保留到低优先级列表
            low_priority.append((obj_file_name, obj_pose[0]))

    # 循环选择物体
    while len(obj_keep_list) < number_of_objects+1: #算上桌子
        # 优先选择高优先级的物体
        if high_priority:
            random.shuffle(high_priority)  # 随机打乱顺序
            obj_name, obj_pos = high_priority.pop(0)  # 从 high_priority 列表中取出
            obj_keep_list.append(obj_name)
        # 如果高优先级物体处理完，则处理其他物体
        elif low_priority:
            random.shuffle(low_priority)  # 随机打乱顺序
            obj_name, obj_pos = low_priority.pop(0)  # 从 low_priority 列表中取出
            obj_keep_list.append(obj_name)
        else:
            break  # 如果没有更多的物体可供选择，退出循环

    # 保留data中obj_keep_list中的物体
    filtered_data = {k: v for k, v in data.items() if k in obj_keep_list}

    # 选定anchor物体 从filtered_data中选择一个物体作为anchor物体，但anchor物体不能是桌子
    anchor_obj = None
    
    while True:
        obj_name = random.choice(obj_keep_list)
        if "desk" not in obj_name and "table" not in obj_name:
            anchor_obj = obj_name
            break
    anchor_pos = data[anchor_obj][0] # x, y, z
    
    # 选定anchor 后， 遍历所有不在obj_keep_list中的物体，计算其相对于anchor的位置
    direction_list = []
    text_dict = {}
    removed_obj_list = []
    obj_distances = []

    # 计算不在 obj_keep_list 中的物体到锚点的距离，并保存这些物体
    for obj_name, obj_pose in data.items():
        if obj_name in obj_keep_list:
            continue
        distance = calculate_distance(obj_pose[0][:2], anchor_pos[:2])  # 计算物体与锚点的2D距离
        obj_distances.append((obj_name, obj_pose, distance))

    # 按照距离从近到远排序
    obj_distances.sort(key=lambda x: x[2])


    for obj_name, obj_pose, _ in obj_distances:
        if obj_name in obj_keep_list:
            continue
        direction = determine_direction(obj_pose[0][:2], anchor_pos[:2])
        if direction not in direction_list:
            direction_list.append(direction)
            removed_obj_list.append(obj_name)
        else :
            continue
        processed_obj_name = key2phrase(obj_name)
        processed_anchor_obj = key2phrase(anchor_obj)
        short_anchor_obj = processed_anchor_obj.split(" ")[-1]
        text_dict[processed_obj_name] = [processed_anchor_obj,f"{direction} {processed_anchor_obj}",f"{short_anchor_obj}",direction]
    if not os.path.exists(new_json_file):
    # 如果文件不存在，首先确保目录存在
        os.makedirs(os.path.dirname(new_json_file), exist_ok=True)
    with open(new_json_file, "w") as json_file:
        json.dump(text_dict, json_file, indent=4)
    
    
    return obj_keep_list, filtered_data, anchor_obj, removed_obj_list


if __name__ == "__main__":
    ##### generate one case
    bproc.init()
    json_file = "dataset/scene_gen/picked_scene_mesh_json/id15_0.json"
    parent_dir = "dataset/picked_RGBD"
    
    scene_id = json_file.split("/")[-1].split(".")[0]


    with open(json_file, 'r') as f:
                data = json.load(f)

    for removed_obj in removed_obj_list:
        
        filtered_data = {k: v for k, v in data.items() if k in obj_keep_list or k == removed_obj}
        removed_obj_name = removed_obj.split("/")[-2]
        RGBD_out_dir = parent_dir + "/" + scene_id + "/" + removed_obj_name
        # check if the data already exists


        bproc_gen_mask_with_and_without_obj_sequnce(filtered_data, RGBD_out_dir, removed_obj)
        bproc.clean_up()



    ##### generate scene and obj in sequence
    # bproc.init()
    # json_folder_path = "dataset/scene_gen/scene_mesh_json"
    # json_files = glob.glob(os.path.join(json_folder_path, '*.json'))
    # data_size = 0
    # for json_file_path in json_files:
    #     print(f"-----start process {json_file_path} ------")
    #     start_time = time.time()
    #     parent_dir = "dataset/scene_RGBD_mask_sequence"
    #     json_file = json_file_path
    #     scene_id = json_file.split("/")[-1].split(".")[0]
    #     if os.path.exists(parent_dir + "/" + scene_id):
    #             continue
    #     obj_keep_list, filtered_data, anchor_obj, removed_obj_list = objs_keep_in_scene(json_file=json_file, number_of_objects=1)

    #     # only keep the objects in obj_keep_list and removed_obj_list
    #     with open(json_file, 'r') as f:
    #         data = json.load(f)

    #     for removed_obj in removed_obj_list:
            
    #         filtered_data = {k: v for k, v in data.items() if k in obj_keep_list or k == removed_obj}
    #         removed_obj_name = removed_obj.split("/")[-2]
    #         RGBD_out_dir = parent_dir + "/" + scene_id + "/" + removed_obj_name
    #          # check if the data already exists


    #         bproc_gen_mask_with_and_without_obj_sequnce(filtered_data, RGBD_out_dir, removed_obj)
    #         bproc.clean_up()
    #         data_size += 1

    #     end_time = time.time()
    #     print(f"------Consume: {end_time - start_time} s--------")
    #     print(f"------Data size: {data_size}--------")
    
    '''
    bproc.init()
    
    # gerate depth.png and hdf5
    parent_dir = "dataset/scene_RGBD_mask"

    
    json_folder_path = "dataset/scene_gen/scene_mesh_json"
    json_files = glob.glob(os.path.join(json_folder_path, '*.json'))

    for json_file_path in json_files:
        #json_file_path = "dataset/scene_gen/scene_mesh_json/id1_2.json" # debug
        with open(json_file_path, 'r', encoding='utf-8') as file:
            # if file exists, continue
            scene_id = os.path.splitext(os.path.basename(json_file_path))[0]
            target_folder = os.path.join(parent_dir, scene_id)
            if os.path.exists(target_folder) and os.path.isdir(target_folder):
                print(f"Data '{scene_id}' already exists. next")
                continue


            data = json.load(file)  # 解析JSON文件内容为Python字典
            print(json_file_path, "-----")
            start_time = time.time()
            for removed_obj_path in data.keys():
                if "desk" in removed_obj_path or "table"  in removed_obj_path: # do not remove desk or table
                    continue
    
                scene_mesh_json = json_file_path
                removed_obj_path = removed_obj_path

                scene_path = scene_mesh_json.split("/")[-1]  
                scene_id = scene_path.split(".")[0]  

                removed_obj_name = removed_obj_path.split("/")[-2]

                output_dir_with_obj = parent_dir + "/" + scene_id + "/"+ removed_obj_name +"/with_obj"
                output_dir_no_obj = parent_dir + "/" + scene_id + "/" + removed_obj_name +"/no_obj"
                print(output_dir_no_obj)
                print(output_dir_with_obj)
    
    
                
                bproc_gen_mask(scene_mesh_json=scene_mesh_json,
                            RGBD_out_dir=output_dir_with_obj,
                            removed_obj=removed_obj_path)
                bproc.clean_up()

                bproc_gen_mask_removw_obj(scene_mesh_json=scene_mesh_json,
                            RGBD_out_dir=output_dir_no_obj,
                            removed_obj=removed_obj_path)
                bproc.clean_up()
            end_time = time.time()
            print(f"------Consume: {end_time - start_time} s--------")
    '''
    """
    ########## 生成sceneRGBD mask的版本
    bproc.init()
    parent_dir = "dataset/scene_RGBD_mask"

    json_folder_path = "dataset/scene_gen/scene_mesh_json"
    json_files = glob.glob(os.path.join(json_folder_path, '*.json'))
    for json_file_path in json_files:


        with open(json_file_path, 'r', encoding='utf-8') as file:

            # if file exists, continue
            scene_id = os.path.splitext(os.path.basename(json_file_path))[0]
            target_folder = os.path.join(parent_dir, scene_id)
            if os.path.exists(target_folder) and os.path.isdir(target_folder):
                print(f"Data '{scene_id}' already exists. next")
                continue

            data = json.load(file)  # 解析JSON文件内容为Python字典
            print(json_file_path, "-----")
            start_time = time.time()
            for removed_obj_path in data.keys():
                if "desk" in removed_obj_path or "table"  in removed_obj_path: # do not remove desk or table
                    continue

                scene_mesh_json = json_file_path
                removed_obj_path = removed_obj_path

                scene_path = scene_mesh_json.split("/")[-1]  
                scene_id = scene_path.split(".")[0]  

                removed_obj_name = removed_obj_path.split("/")[-2]

                output_dir_with_obj = parent_dir + "/" + scene_id + "/"+ removed_obj_name +"/with_obj"
                output_dir_no_obj = parent_dir + "/" + scene_id + "/" + removed_obj_name +"/no_obj"
                RGB_out_dir = parent_dir + "/" + scene_id + "/" + removed_obj_name

                bproc_gen_mask_with_and_without_obj(scene_mesh_json=scene_mesh_json,
                                                    RGBD_out_dir=RGB_out_dir,
                                                    removed_obj=removed_obj_path)
                bproc.clean_up()

            end_time = time.time()
            print(f"------Consume: {end_time - start_time} s--------")
        """