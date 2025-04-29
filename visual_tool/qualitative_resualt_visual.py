import os
import numpy as np
import open3d as o3d
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

def get_tf_for_scene_rotation(points, axis="z"):
    """
    Get the transformation matrix for rotating the scene to align with the plane normal.

    Args:
        points (np.ndarray): The points of the scene.
        axis (str): The axis to align with the plane normal.
    
    Returns:
        T_plane: The transformation matrix.
        plane_model: The plane model.

    """

    points_filtered = points
    plane_model = fit_plane_from_points(points_filtered)
    pcd_filtered = o3d.geometry.PointCloud()
    pcd_filtered.points = o3d.utility.Vector3dVector(points_filtered)


    plane_dir = -plane_model[:3]
    plane_dir = plane_dir / np.linalg.norm(plane_dir)
    T_plane = np.eye(4)
    if axis == "y":
        T_plane[:3, 1] = -plane_dir
        T_plane[:3, 1] /= np.linalg.norm(T_plane[:3, 1])
        T_plane[:3, 2] = -np.cross([1, 0, 0], plane_dir)
        T_plane[:3, 2] /= np.linalg.norm(T_plane[:3, 2])
        T_plane[:3, 0] = np.cross(T_plane[:3, 1], T_plane[:3, 2])
    elif axis == "z":
        T_plane[:3, 2] = -plane_dir 
        T_plane[:3, 2] /= np.linalg.norm(T_plane[:3, 2]) 
        T_plane[:3, 0] = -np.cross([0, 1, 0], plane_dir) 
        T_plane[:3, 0] /= np.linalg.norm(T_plane[:3, 0])
        T_plane[:3, 1] = np.cross(T_plane[:3, 2], T_plane[:3, 0])

    return T_plane, plane_model

def fit_plane_from_points(points, threshold=0.01, ransac_n=3, num_iterations=2000):
    """
    Fit a plane from the points.

    Args:
        points (np.ndarray): The points.
        threshold (float): The threshold for RANSAC.
        ransac_n (int): The number of points to sample for RANSAC.
        num_iterations (int): The number of iterations for RANSAC.
    
    Returns:
        plane_model: The plane model.
    
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    plane_model, inliers = o3d.geometry.PointCloud.segment_plane(
    pcd,
    distance_threshold=threshold,
    ransac_n=ransac_n,
    num_iterations=num_iterations,
    )
    return plane_model

def backproject(depth, intrinsics, instance_mask, NOCS_convention=False):
    """
        depth: np.array, [H,W]
        intrinsics: np.array, [3, 3]
        instance_mask: np.array, [H, W]; (np.logical_and(depth>0, depth<2))
    """
    intrinsics_inv = np.linalg.inv(intrinsics)
    non_zero_mask = depth > 0
    final_instance_mask = np.logical_and(instance_mask, non_zero_mask)

    idxs = np.where(final_instance_mask)
    grid = np.array([idxs[1], idxs[0]])

    length = grid.shape[1]
    ones = np.ones([1, length])
    uv_grid = np.concatenate((grid, ones), axis=0)  # [3, num_pixel]

    xyz = intrinsics_inv @ uv_grid  # [3, num_pixel]
    xyz = np.transpose(xyz)  # [num_pixel, 3]

    z = depth[idxs[0], idxs[1]]

    # print(np.amax(z), np.amin(z))
    pts = xyz * z[:, np.newaxis] / xyz[:, -1:]
    if NOCS_convention:
        pts[:, 1] = -pts[:, 1]
        pts[:, 2] = -pts[:, 2]

    return pts, idxs

def visualize_points(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def get_obj_mesh(obj_mesh_file_path, target_size=1):
    obj_mesh = o3d.io.read_triangle_mesh(obj_mesh_file_path)
    obj_mesh.compute_vertex_normals()
    bounding_box = obj_mesh.get_axis_aligned_bounding_box()
    center = obj_mesh.get_center()
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
    obj_mesh.translate(translation)  # move obj mesh
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
    data_filename = os.path.basename(args.npz_path).strip(".npz")
    render_dir = "qualitative_demo/blender_output/{}".format(data_filename)
    os.makedirs(render_dir, exist_ok=True)
    project_fpath = os.path.join(render_dir, "project.blend")
    if args.overwrite_project and os.path.exists(project_fpath):
        os.remove(project_fpath)
    render_fpath = os.path.join(render_dir, "render.png")

    pred_xyz_all = data["pred_xyz_all"]
    pred_r_all  = data["pred_r_all"]
    pred_cost = data["pred_cost"]
    rgb_image = data["rgb_image"] # [H, W, 3]

    # save the image use PIL
    from PIL import Image
    rgb_image = Image.fromarray(rgb_image.astype(np.uint8))
    rgb_image.save(os.path.join(".tmp", "cactus_scene.png"))



    depth_image_raw = data["depth_image_raw"]
    depth_image = data["depth_image"] # in mm
    #depth_image_completed = data["depth_image_completed"]
    intr = data["intrinsics"]
    mesh_category = data["mesh_category"]
    target_size = data["target_size"].item()

    if args.obj_mesh is not None:
        obj_mesh_file = args.obj_mesh
    else:
        obj_mesh_file = data["obj_mesh_file"].item()
    pred_xyz_all[..., 2] = np.min(pred_xyz_all[..., 2]) 

    # get processed obj mesh
    print(f"Processing the obj mesh: {obj_mesh_file}")
    obj_mesh = get_obj_mesh(obj_mesh_file, target_size)

    # visuliaze the image
    # import matplotlib.pyplot as plt
    # plt.imshow(rgb_image)
    # plt.axis("off")
    # plt.show()


    # built scene point cloud
    points_scene, idx= backproject(
        depth_image/1000,
        intr,
        depth_image> 0,
        NOCS_convention=False,
    )
    rgb_colors = rgb_image[idx[0], idx[1], :] / 255
    scene_pcd = visualize_points(points_scene, rgb_colors)
    o3d.io.write_point_cloud(".tmp/pipepine_depth.ply", scene_pcd)
    # o3d.visualization.draw_geometries([scene_pcd])
    T_plane, plane_model = get_tf_for_scene_rotation(points_scene)
    #visuallize plane coordinates frame

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    axis.transform(T_plane)
    vis_o3d = [scene_pcd, axis]
    
    guide_loss_color = get_heatmap(pred_cost[None])[0] # [N, 3]
    plane_z_axis = T_plane[:3, 2]

    coord_z_plane = []
    for i in range(len(pred_xyz_all)):
        coordinate_on_z = np.dot(pred_xyz_all[i], plane_z_axis) * plane_z_axis
        coord_z_plane.append(coordinate_on_z)
    coord_z_plane = np.array(coord_z_plane) # [N, 3]
    coord_z_plane_final = np.min(coord_z_plane, axis=0)

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
        xyz_object = pred_xyz_all[i] - plane_z_axis * (coord_z_plane[i] - coord_z_plane_final)
        pred_xyz_all[i] = xyz_object

        obj_mesh_i.rotate(dR_object, center=[0, 0, 0])
        obj_mesh_i.rotate(T_plane[:3, :3], center=[0, 0, 0])  # rotate obj mesh
        obj_mesh_i.translate(xyz_object)  # move obj

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
        print("... Initialize new project file and set camera parameters")
        initialize_renderer()
        bt.setLight_sun(light_rotation, light_strength, 1)
        bt.invisibleGround(ground_location, 100, shadowBrightness=0.05)
        bt.shadowThreshold(alphaThreshold=0.05, interpolationMode="CARDINAL")
        cam = bt.setCamera_from_UI(cam_location, cam_rotation, focalLength=focal_length)  

        scene_mesh = bt.readNumpyPoints(
            np.asarray(scene_pcd.points), (0, 0, 0), (0, 0, 0), (1, 1, 1)
        )
        scene_mesh = bt.setPointColors(scene_mesh, np.asarray(scene_pcd.colors))
        #bt.setMat_pointCloudColored(scene_mesh, POINT_COLOR, SCENE_POINT_SIZE) # TODO: try this mode
        setMat_pointCloudColoredEmission(scene_mesh, POINT_COLOR, SCENE_POINT_SIZE)
        scene_mesh.name = "SceneGeo"
        obj_mesh_combined = o3d.geometry.TriangleMesh()
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

            # Prepare the object mesh
            o3d.io.write_triangle_mesh("visual_tool/obj.ply", obj_mesh_i)  # Save the mesh ..
            mesh = bt.readMesh("visual_tool/obj.ply", (0, 0, 0), (0, 0, 0), (1, 1, 1))
            bpy.ops.object.shade_smooth()
            #bt.setMat_plastic(mesh, MESH_COLOR)

            color = list(guide_loss_color_i) + [0.0] 
            # C1 = bt.colorObj(bt.coralRed, 0.5, 1.0, 1.0, 0.0, 0.5)
            # C2 = bt.colorObj(bt.caltechOrange, 0.5, 1.0, 1.0, 0.0, 0.0)
            # C1 = bt.colorObj(color, 0.5, 1.0, 1.0, 0.0, 0.5)
            # C2 = bt.colorObj(color, 0.5, 1.0, 1.0, 0.0, 0.0)
            # bt.setMat_carPaint(mesh, C1, C2)
            meshColor = bt.colorObj(color, 0.5, 1.0, 1.0, 0.0, 0.0)
            AOStrength = 0.5
            metalVal = 0.9
            bt.setMat_metal(mesh, meshColor, AOStrength, metalVal)
            # bt.setMat_emission(mesh, C1, emission_strength=0.5)
            os.remove("visual_tool/obj.ply")  # .. and remove it
        ################################ BLENDER RENDERING ################################

        # Tweak the ground rotation
        # bpy.data.objects["Plane"].rotation_euler[1] = np.pi / 2
        # bpy.data.objects["Plane"].rotation_euler[2] = np.pi / 2

        # Do the rendering
        bpy.ops.wm.save_as_mainfile(filepath=project_fpath)
        bt.renderImage(render_fpath, cam)
        print("... Rendered image saved at: ", render_fpath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--npz_path",
        type=str,
        default="qualitative_demo/qualitative_npz/cactus_workingdesk_rw.npz",
    )
    parser.add_argument(
        "--obj_mesh",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--overwrite_project",
        action="store_true",
        help="overwrite the project file",
    )
    parser.add_argument(
        "--cam_location", type=float, nargs=3, default=[0.0, 0.074, -0.85]
    )
    parser.add_argument("--cam_rotation", type=float, nargs=3, default=[-180.0, 0, 0])
    parser.add_argument("--ground_location", type=float, nargs=3, default=[0, 0, 4.5])
    parser.add_argument(
        "--light_rotation", type=float, nargs=3, default=[9, 170, -50]
    )
    parser.add_argument("--light_strength", type=float, default=5)
    parser.add_argument("--focal_length", type=float, default=55)
    parser.add_argument("--visual_direct", type=bool, default=True)
    parser.add_argument("--blender_render", type=bool, default=True)
    
    args = parser.parse_args()
    visualize_npz(args)