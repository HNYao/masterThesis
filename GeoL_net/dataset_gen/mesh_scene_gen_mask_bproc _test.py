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


""""
TODO: realize load 1 and generate 2
    almost same as the mesh_scene_gen_bproc.py
    the only difference is generating a mask.json containing the bbox parameters of obj.
    json to RGBD(mask)

    TEST: only load obj once and generate 2 files (with obj and without obj)

"""

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


if __name__ == "__main__":
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
                    print(removed_obj_path)
    
    #scene_mesh_json = "dataset/scene_gen/scene_mesh_json/id16_7.json"
    #removed_obj_path = "dataset/obj/mesh/cup/cup_0004_white/mesh.obj"
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

        #scene_mesh_json = "dataset/scene_gen/scene_mesh_json/id16_7.json"
    #removed_obj_path = "dataset/obj/mesh/cup/cup_0004_white/mesh.obj"
    scene_mesh_json = "dataset/scene_gen/scene_mesh_json/id1_5.json"
    removed_obj_path = "dataset/obj/mesh/cup/cup_0004_white/mesh.obj"
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
    '''