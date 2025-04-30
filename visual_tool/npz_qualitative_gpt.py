import cv2
import numpy as np
from GeoL_net.dataset_gen.RGBD2PC import backproject, visualize_points
import matplotlib.pyplot as plt
from GeoL_net.gpt.gpt import *
from metrics.utils import *
import os
import open3d as o3d
import random
from glob import glob

def retrieve_obj_mesh(obj_category, target_size=1, obj_mesh_dir="data_and_weights/mesh/"):
    obj_mesh_file_dir_default = os.path.join(obj_mesh_dir, obj_category)
    if not os.path.exists(obj_mesh_file_dir_default):
        obj_mesh_file = os.path.join("data_and_weights/mesh_realworld", "{}.obj".format(obj_category))
    else:
        obj_mesh_files = glob(os.path.join(obj_mesh_file_dir_default, "*", "mesh.obj"))
        obj_mesh_file = obj_mesh_files[random.randint(0, len(obj_mesh_files)-1)]
    print("Selected object mesh file: ", obj_mesh_file)
    obj_mesh = o3d.io.read_triangle_mesh(obj_mesh_file)
    obj_mesh.compute_vertex_normals()
        
    # Compute the bounding box of the mesh, and acquire the diagonal length
    bounding_box = obj_mesh.get_axis_aligned_bounding_box()
    diagonal_length = np.linalg.norm(bounding_box.get_max_bound() - bounding_box.get_min_bound())
    # Compute the scale factor to resize the mesh
    scale = target_size / diagonal_length
    obj_inverse_matrix = np.array(
        [
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ]
    )
    # obj_mesh.transform(obj_scale_matrix)
    obj_mesh.scale(scale, center=[0, 0, 0])  # scale obj mesh
    obj_mesh.rotate(obj_inverse_matrix, center=[0, 0, 0])  # rotate obj mesh
    
    # Move the mesh center to the bottom part of the mesh
    vertices = np.asarray(obj_mesh.vertices)
    # Find the minimum y-coordinate (bottom point)
    min_z = np.min(vertices[:, 2])
    # Create translation vector to move bottom to origin
    translation = np.array([0, 0, min_z])
    # Apply translation to move bottom to origin
    obj_mesh.translate(translation)
    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    # o3d.visualization.draw([obj_mesh, axis])
    return obj_mesh, obj_mesh_file

def predicted_placement_gpt(
        npz_file_path, 
        object_name: str = "object",
        direction:str = "Right",
        anchor_obj_name: str = "monitor",):
    """
    Visualize the predicted placement of objects in a scene using the data from a .npz file.

    Parameters:
        npz_file_path (str): Path to the .npz file containing the prediction data.
    """
    # Load the data from the .npz file
    data = np.load(npz_file_path)
    
    # Extract the relevant data
    pred_xyz_all = data["pred_xyz_all"]
    pred_r_all = data["pred_r_all"]
    pred_cost = data["pred_cost"]
    rgb_image = data["rgb_image"]
    depth_image_raw = data["depth_image_raw"]
    #depth_image_completed = data["depth_image_completed"]
    depth_image = data["depth_image"]
    intrinsics = data["intrinsics"]
    
    # visualize the rgb_image use plt
    # plt.imshow(rgb_image)
    # plt.show()

    # save the image temporarily
    cv2.imwrite(".tmp/chatgpt_temp.png", rgb_image)

    # use chatgpt to get bbox
    bbox = chatgpt_object_placement_bbox(
        gpt_version="gpt-4o",
        image_path=".tmp/chatgpt_temp.png",
        prompts_obj_place=object_name,
        prompts_direction=direction,
        prompts_anchor_obj=anchor_obj_name,
    )
    updated_img = sample_points_in_bbox(rgb_image, bbox, n=5)
    pcd_point, idx = backproject(
        depth_image/1000,
        intrinsics,
        np.logical_and(depth_image/1000 > 0, depth_image/1000 < 2),
        NOCS_convention=False,
    )
    colors = rgb_image[idx[0], idx[1]] / 255
    pcd = visualize_points(pcd_point, colors)
    o3d.visualization.draw_geometries([pcd])    

    
    colors_sampled = updated_img[idx[0], idx[1]] / 255
    pcd_sampled = visualize_points(pcd_point, colors_sampled)
    o3d.visualization.draw_geometries([pcd_sampled])

    colors = np.asarray(pcd_sampled.colors)
    points = np.asarray(pcd_sampled.points)

    is_red = (colors[:, 0] >= 0.999) & (colors[:, 1] <= 0.1) & (colors[:, 2] <= 0.1)
    sampled_points = points[is_red]

    position_to_place = sampled_points

    # add the position to place into the npz

    data_dict = dict(data)
    data_dict["pred_xyz_all_gpt"] = position_to_place
    print("pred_xyz_all_gpt:", position_to_place)


    # save the npz file
    np.savez_compressed(npz_file_path, **data_dict)


if __name__ == "__main__":
    # Example usage
    npz_file_paths = {
         "qualitative_demo/qualitative_npz/cactus_workingdesk_rw.npz": ["cactus", "On", "Put it on the left of the keyboard"],
        "qualitative_demo/qualitative_npz/box_shelf_rw.npz": ["box", "On", "relatvely empty shelf"],
        "qualitative_demo/qualitative_npz/cup_drinkwine_rw.npz": ["cup", "Left", "white wine"],
        "qualitative_demo/qualitative_npz/detergent_sink_rw.npz":["detergent", "Right Behind", "left sink"],
        "qualitative_demo/qualitative_npz/keyboard_desk_rw.npz": ["keyboard", "Left", "Mouse"],
        "qualitative_demo/qualitative_npz/knife_draw_rw.npz":["knife", "On", "Knives"],
        "qualitative_demo/qualitative_npz/mouse_lefthanded_rw.npz": ["mouse", "Left", "keyboard"],
        "qualitative_demo/qualitative_npz/mouse_righthanded_rw.npz": ["mouse", "Right", "keyboard"],
        "qualitative_demo/qualitative_npz/plate_dinningtable_rw.npz": ["plate", "Left, Right", "knife, fork"],
        "qualitative_demo/qualitative_npz/cake_plate_rw.npz": ["cake", "On", "plate"],
       

    }
    for npz_file_path, item in npz_file_paths.items():
        object_name, direction, anchor_obj_name = item

        predicted_placement_gpt(npz_file_path, object_name="object")
        print("Finished processing: ", npz_file_path)

    