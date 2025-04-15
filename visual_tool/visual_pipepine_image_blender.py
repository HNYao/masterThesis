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



def main(args):
    cam_location = args.cam_location
    cam_rotation = args.cam_rotation
    ground_location = args.ground_location
    focal_length = args.focal_length
    light_rotation = args.light_rotation
    light_strength = args.light_strength      


    # Prepare the blender project file
    print("... Initialize new project file and set camera parameters")
    initialize_renderer()
    bt.setLight_sun(light_rotation, light_strength, 1)
    bt.invisibleGround(ground_location, 100, shadowBrightness=0.05)
    bt.shadowThreshold(alphaThreshold=0.05, interpolationMode="CARDINAL")
    cam = bt.setCamera_from_UI(cam_location, cam_rotation, focalLength=focal_length)  

    scene_pcd = o3d.io.read_point_cloud(".tmp/pipepine_depth.ply")

    scene_mesh = bt.readNumpyPoints(
        np.asarray(scene_pcd.points), (0, 0, 0), (0, 0, 0), (1, 1, 1)
    )
    scene_mesh = bt.setPointColors(scene_mesh, np.asarray(scene_pcd.colors))
    #bt.setMat_pointCloudColored(scene_mesh, POINT_COLOR, SCENE_POINT_SIZE) # TODO: try this mode
    setMat_pointCloudColoredEmission(scene_mesh, POINT_COLOR, SCENE_POINT_SIZE)
    scene_mesh.name = "SceneGeo"


    project_fpath = os.path.join("qualitative_demo/pipeline_img/depth", "project.blend")
    render_fpath = os.path.join("qualitative_demo/pipeline_img/depth", "render.png")



    bpy.ops.wm.save_as_mainfile(filepath=project_fpath)
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
