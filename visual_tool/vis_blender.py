import trimesh
import open3d as o3d
import numpy as np
import cv2
from sklearn.neighbors import KDTree
import open3d as o3d
import numpy as np
import cv2
import os
import blendertoolbox as bt
import bpy
import argparse
from visual_tool.bpy_utils import setMat_pointCloudColoredEmission

OBJ_COLOR_BLUE = [153.0 / 255, 203.0 / 255, 67.0 / 255, 1.0]
OBJ_COLOR_RED = [250.0 / 255, 114.0 / 255, 104.0 / 255, 0.5]

MESH_COLOR = bt.colorObj(OBJ_COLOR_RED, 0.5, 1.0, 1.0, 0.0, 2.0)
POINT_COLOR = bt.colorObj([], 0.5, 1.0, 1.0, 0.0, 0.2)
SCENE_POINT_SIZE = 0.005


def initialize_renderer(height=720, width=1280, exposure=1, numSamples=64):
    bt.blenderInit(width, height, numSamples, exposure, True)
    bpy.data.scenes[0].view_layers[0]["cycles"]["use_denoising"] = 1


# read npy as dictionary
npy_file = np.load(
    "selected_scene/align_pointcloud_with_mesh.npy", allow_pickle=True
).item()

intrinsics = np.array(
    [
        [607.09912 / 2, 0.0, 636.85083 / 2],
        [0.0, 607.05212 / 2, 367.35952 / 2],
        [0.0, 0.0, 1.0],
    ]
)


def main(args):

    # Processing the scene points
    pts_without_obj = o3d.geometry.PointCloud()
    pts_without_obj.points = o3d.utility.Vector3dVector(npy_file["scene_pcd_point"])
    pts_without_obj.colors = o3d.utility.Vector3dVector(npy_file["scene_pcd_color"])

    # Processing the object mesh
    obj_mesh_path = "selected_scene/mesh.obj" # get the mesh path
    obj_mesh = o3d.io.read_triangle_mesh(obj_mesh_path)
    obj_mesh.compute_vertex_normals()
    obj_scale = np.array(npy_file["obj_scale"]) # get the obj mesh scale
    obj_mesh.scale(obj_scale[0], center=[0, 0, 0])
    obj_pcd = obj_mesh.sample_points_uniformly(number_of_points=10000)
    obj_rotation = 0
    obj_rotation_matrix = np.array(npy_file["obj_rotation_matrix"]) # get the obj full rotation matrix
    obj_mesh.rotate(obj_rotation_matrix, center=[0, 0, 0])  # rotate obj mesh
    obj_mesh.translate([0, -1, 1])
    obj_pcd.rotate(obj_rotation_matrix, center=[0, 0, 0])  # rotate obj mesh
    obj_pcd.translate([0, -1, 1])
    obj_pcd_target_point = [-0.2, -0.75, 1]
    obj_max_bound = obj_pcd.get_max_bound()
    obj_min_bound = obj_pcd.get_min_bound()
    obj_bottom_center = (obj_max_bound + obj_min_bound) / 2
    obj_bottom_center[2] = obj_max_bound[2]  # attention: the z axis is reversed
    obj_pcd.translate(
        obj_pcd_target_point - obj_bottom_center
    )  # move obj mesh to target point
    obj_mesh.translate(obj_pcd_target_point - obj_bottom_center)  # move obj
    # obj_mesh.paint_uniform_color([OBJECT_COLOR[0], OBJECT_COLOR[1], OBJECT_COLOR[2]])
    # # Visualize the scene points and object mesh
    # o3d.visualization.draw([pts_without_obj, obj_pcd, obj_mesh])

    # Parse parameters, dont need to change
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

    ################################ BLENDER RENDERING ################################
    # Prepare the scene points
    scene_mesh = bt.readNumpyPoints(
        npy_file["scene_pcd_point"], (0, 0, 0), (0, 0, 0), (1, 1, 1)
    )
    scene_mesh = bt.setPointColors(scene_mesh, npy_file["scene_pcd_color"])
    # bt.setMat_pointCloudColored(scene_mesh, POINT_COLOR, SCENE_POINT_SIZE)
    setMat_pointCloudColoredEmission(scene_mesh, POINT_COLOR, SCENE_POINT_SIZE)
    scene_mesh.name = "SceneGeo"

    # Prepare the object mesh
    o3d.io.write_triangle_mesh("visual_tool/obj.ply", obj_mesh)  # Save the mesh ..
    mesh = bt.readMesh("visual_tool/obj.ply", (0, 0, 0), (0, 0, 0), (1, 1, 1))
    bpy.ops.object.shade_smooth()
    # bt.setMat_plastic(mesh, MESH_COLOR)
    C1 = bt.colorObj(bt.coralRed, 0.5, 1.0, 1.0, 0.0, 0.5)
    C2 = bt.colorObj(bt.caltechOrange, 0.5, 1.0, 1.0, 0.0, 0.0)
    bt.setMat_carPaint(mesh, C1, C2)
    os.remove("visual_tool/obj.ply")  # .. and remove it
    ################################ BLENDER RENDERING ################################

    # Tweak the ground rotation
    # bpy.data.objects["Plane"].rotation_euler[1] = np.pi / 2
    # bpy.data.objects["Plane"].rotation_euler[2] = np.pi / 2

    # Do the rendering
    bpy.ops.wm.save_as_mainfile(filepath=project_fpath)
    render_fpath = os.path.join("selected_scene/render.png")
    bt.renderImage(render_fpath, cam)
    print("... Rendered image saved at: ", render_fpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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
    args = parser.parse_args()

    main(args)
