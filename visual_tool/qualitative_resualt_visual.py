import os
import numpy as np
import open3d as o3d
from GeoL_net.dataset_gen.RGBD2PC import backproject, visualize_points
from GeoL_diffuser.models.utils.fit_plane import *
import matplotlib.cm as cm
from scipy.spatial.transform import Rotation as SciR
import blendertoolbox as bt
import bpy
import argparse
from visual_tool.bpy_utils import setMat_pointCloudColoredEmission

OBJ_COLOR_BLUE = [153.0 / 255, 203.0 / 255, 67.0 / 255, 1.0]
OBJ_COLOR_RED = [250.0 / 255, 114.0 / 255, 104.0 / 255, 0.5]

MESH_COLOR = bt.colorObj(OBJ_COLOR_RED, 0.5, 1.0, 1.0, 0.0, 2.0)
POINT_COLOR = bt.colorObj([], 0.5, 1.0, 1.0, 0.0, 0.2)
SCENE_POINT_SIZE = 0.005


def get_obj_mesh(obj_mesh_file_path, target_size=1):
    obj_mesh = o3d.io.read_triangle_mesh(obj_mesh_file_path)
    obj_mesh.compute_vertex_normals()
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
    return obj_mesh

def get_heatmap(values, cmap_name="turbo", invert=False):
    if invert:
        values = -values
    values = (values - values.min()) / (values.max() - values.min())
    colormaps = cm.get_cmap(cmap_name)
    rgb = colormaps(values)[..., :3]  # don't need alpha channel
    return rgb

def initialize_renderer(height=720, width=1280, exposure=1, numSamples=64):
    bt.blenderInit(width, height, numSamples, exposure, True)
    bpy.data.scenes[0].view_layers[0]["cycles"]["use_denoising"] = 1

def visualize_npz(args):
    data = np.load(args.npz_path)
    
    pred_xyz_all = data["pred_xyz_all"]
    pred_r_all  = data["pred_r_all"]
    pred_cost = data["pred_cost"]
    rgb_image = data["rgb_image"]
    depth_image_raw = data["depth_image_raw"]
    depth_image = data["depth_image"] # in mm
    intr = data["intrinsics"]
    mesh_category = data["mesh_category"]
    target_size = data["target_size"].item()
    obj_mesh_file = data["obj_mesh_file"].item()
    

    # get processed obj mesh
    print(f"Processing the obj mesh: {obj_mesh_file}")
    obj_mesh = get_obj_mesh(obj_mesh_file, target_size)


    # built scene point cloud
    points_scene, idx= backproject(
        depth_image/1000,
        intr,
        depth_image > 0,
        NOCS_convention=False,
    )
    rgb_colors = rgb_image[idx[0], idx[1], :] / 255
    scene_pcd = visualize_points(points_scene, rgb_colors)
    T_plane, plane_model = get_tf_for_scene_rotation(points_scene)
    
    vis_o3d = [scene_pcd]
    
    guide_loss_color = get_heatmap(pred_cost[None])[0] # [N, 3]
    
    # visualize the point clouds with objects mesh directly
    for i in range(len(pred_xyz_all)):
        guide_loss_color_i = guide_loss_color[i]
        # Create a mesh by copying the vertices and faces of the original mesh
        obj_mesh_i = o3d.geometry.TriangleMesh()
        obj_mesh_i.vertices = obj_mesh.vertices
        obj_mesh_i.triangles = obj_mesh.triangles
        obj_mesh_i.compute_vertex_normals()
        
        obj_mesh_i.paint_uniform_color(guide_loss_color_i)
        dR_object = SciR.from_euler("Z", pred_r_all[i], degrees=True).as_matrix()

        obj_mesh_i.rotate(dR_object, center=[0, 0, 0])
        obj_mesh_i.rotate(T_plane[:3, :3], center=[0, 0, 0])  # rotate obj mesh
        obj_mesh_i.translate(pred_xyz_all[i])  # move obj

        vis_o3d.append(obj_mesh_i)  

    o3d.visualization.draw(vis_o3d)
    
    
    if args.blender_render == True:
        cam_location = args.cam_location
        cam_rotation = args.cam_rotation
        ground_location = args.ground_location
        focal_length = args.focal_length
        light_rotation = args.light_rotation
        light_strength = args.light_strength      


        # Prepare the blender project file
        project_fpath = "selected_scene/project.blend"
        print("... Initialize new project file and set camera parameters")
        initialize_renderer()
        bt.setLight_sun(light_rotation, light_strength, 1)
        bt.invisibleGround(ground_location, 100, shadowBrightness=0.05)
        bt.shadowThreshold(alphaThreshold=0.05, interpolationMode="CARDINAL")
        cam = bt.setCamera_from_UI(cam_location, cam_rotation, focalLength=focal_length)  

        scene_mesh = bt.readNumpyPoints(
            np.asarrray(scene_pcd.points), (0, 0, 0), (0, 0, 0), (1, 1, 1)
        )
        scene_mesh = bt.setPointColors(scene_mesh, np.asarray(scene_pcd.colors))
        # bt.setMat_pointCloudColored(scene_mesh, POINT_COLOR, SCENE_POINT_SIZE)
        setMat_pointCloudColoredEmission(scene_mesh, POINT_COLOR, SCENE_POINT_SIZE)
        scene_mesh.name = "SceneGeo"
        
        o3d.visualization.draw([scene_mesh])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--npz_path",
        type=str,
        default="qualitative_demo/qualitative_npz/mouse_righthanded.npz",
    )
    parser.add_argument(
        "--overwrite_project",
        action="store_true",
        help="overwrite the project file",
    )
    parser.add_argument(
        "--cam_location", type=float, nargs=3, default=[0.0, 0.79, -0.94]
    )
    parser.add_argument("--cam_rotation", type=float, nargs=3, default=[-140.0, 0, 0])
    parser.add_argument("--ground_location", type=float, nargs=3, default=[0, 0, 4.5])
    parser.add_argument(
        "--light_rotation", type=float, nargs=3, default=[-18, 200, -50]
    )
    parser.add_argument("--light_strength", type=float, default=5)
    parser.add_argument("--focal_length", type=float, default=30)
    parser.add_argument("--visual_direct", type=bool, default=True)
    parser.add_argument("--blender_render", type=bool, default=True)
    
    args = parser.parse_args()
    visualize_npz(args)