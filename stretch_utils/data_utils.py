
import cv2
import numpy as np
from transformations import rotation_matrix
import torch
from easydict import EasyDict as edict
from stretch_utils.layers import BackprojectDepth
import open3d as o3d
import json
import matplotlib.pyplot as plt

def pick_points_in_viewer(points, scene_colors=None, verbose=False):
    def pick_points(pcd):
        print("")
        print(
            "1) Please pick at least three correspondences using [shift + left click]"
        )
        print("   Press [shift + right click] to undo point picking")
        print("2) After picking points, press 'Q' to close the window")
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()  # user picks points
        vis.destroy_window()
        print("")
        return vis.get_picked_points()

    if isinstance(points, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if scene_colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(scene_colors)
    else:
        pcd = points

    picked_ids = pick_points(pcd)
    final_points = np.asarray(pcd.points)[picked_ids]

    if verbose:
        print("Final points: ")
        for i in range(len(final_points)):
            print(final_points[i])

    return final_points

def backproject(depth, intrinsics, instance_mask, NOCS_convention=True):
    intrinsics_inv = np.linalg.inv(intrinsics)
    # image_shape = depth.shape
    # width = image_shape[1]
    # height = image_shape[0]

    # x = np.arange(width)
    # y = np.arange(height)

    # non_zero_mask = np.logical_and(depth > 0, depth < 5000)
    non_zero_mask = depth > 0
    final_instance_mask = np.logical_and(instance_mask, non_zero_mask)

    idxs = np.where(final_instance_mask)
    grid = np.array([idxs[1], idxs[0]])

    # shape: height * width
    # mesh_grid = np.meshgrid(x, y) #[height, width, 2]
    # mesh_grid = np.reshape(mesh_grid, [2, -1])
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


def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def save_json(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f)



def get_heatmap(values, cmap_name="turbo", invert=False):
    if invert:
        values = -values
    values = (values - values.min()) / (values.max() - values.min())
    colormaps = plt.cm.get_cmap(cmap_name)
    rgb = colormaps(values)[..., :3]  # don't need alpha channel
    return rgb

def visualize_scores(color, score):
    score_vis = get_heatmap(score)
    score_vis = (score_vis * 255).astype(np.uint8)[..., [2, 1, 0]].copy()
    score_vis = cv2.blur(score_vis, (5, 5))
    vis = cv2.addWeighted(color.copy(), 0.5, score_vis, 0.5, 0)
    return vis

def procrustes_analysis(X0,X1): # [N,3]
    is_np = False
    if isinstance(X0, np.ndarray):
        X0 = torch.from_numpy(X0)
        X1 = torch.from_numpy(X1)
        is_np = True
    # translation
    t0 = X0.mean(dim=0,keepdim=True)
    t1 = X1.mean(dim=0,keepdim=True)
    X0c = X0-t0
    X1c = X1-t1
    # scale
    s0 = (X0c**2).sum(dim=-1).mean().sqrt()
    s1 = (X1c**2).sum(dim=-1).mean().sqrt()
    X0cs = X0c/s0
    X1cs = X1c/s1
    # rotation (use double for SVD, float loses precision)
    U,S,V = (X0cs.t()@X1cs).double().svd(some=True)
    R = (U@V.t()).float()
    if R.det()<0: R[2] *= -1
    # align X1 to X0: X1to0 = (X1-t1)/s1@R.t()*s0+t0
    if is_np:
        R = R.numpy()
        t0 = t0.numpy()
        t1 = t1.numpy()
        s0 = s0.numpy()
        s1 = s1.numpy()
        
    return R, t0, t1, s0, s1

def preprocess_stretch_head_image(color, depth, intr, verbose=False, cut_mode="center"):
    
    # Rotate the image, but now we need the relative transformation!
    new_color = np.rot90(color.copy(), -1)
    new_depth = np.rot90(depth.copy(), -1)
    new_intr = intr.copy()
    fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
    new_intr[0, 0], new_intr[1, 1], new_intr[0, 2], new_intr[1, 2] = fy, fx, cy, cx 

    # Computet the relative transformation
    depth_pt = torch.from_numpy(depth.copy())[None, None]
    new_depth_pt = torch.from_numpy(new_depth.copy())[None, None]
    points = BackprojectDepth(depth.shape[-2], depth.shape[-1])(depth_pt, intr) # [B, 3, HW]
    new_points = BackprojectDepth(new_depth_pt.shape[-2], new_depth.shape[-1])(new_depth_pt, new_intr) # [B, 3, HW]
    points = points[0].T.view(depth.shape[-2], depth.shape[-1], 3)
    points = torch.rot90(points, -1) # [H, W, 3]
    new_points = new_points[0].T.view(new_depth.shape[-2], new_depth.shape[-1], 3)  # [H, W, 3]
    points = points.reshape(-1, 3)
    new_points = new_points.reshape(-1, 3)

    # USV decomposition to know the relative rotation as we are doing rotation! (procrustes_analysis, but no translation)
    # R, t0, t1, s0, s1 = procrustes_analysis(points, new_points)
    # R, t1, t0, s0, s1 = R.numpy(), t1.numpy(), t0.numpy(), s0.numpy(), s1.numpy()
    U,_,V = (points.t()@new_points).double().svd(some=True)
    R = (U@V.t()).float()
    if R.det()<0: 
        R[2] *= -1
    R = R.numpy()
    T_head_crop = np.eye(4)
    T_head_crop[:3, :3] = R
    
    if verbose:
        points_np = points.numpy()
        new_points_np = new_points.numpy()
        # new_points_np = (new_points_np - t1) / s1 @ R.T * s0 + t0
        new_points_np = new_points_np @ T_head_crop[:3, :3].T
        dist = np.linalg.norm(points_np-new_points_np, axis=-1)

        print("===> Distance after solving: {}".format(dist.mean()))

        points_vis, scene_ids = backproject(depth, intr, depth<5, False)
        new_points_vis, _ = backproject(new_depth, new_intr, new_depth<5, False)
        # new_points_vis = (new_points_vis - t1) / s1 @ R.T * s0 + t0
        new_points_vis = new_points_vis @ T_head_crop[:3, :3].T
        point_colors_vis = color[scene_ids[0], scene_ids[1]] / 255.
        pcd = visualize_points(points_vis, point_colors_vis)
        pcd_new = visualize_points(new_points_vis)
        pcd_new.paint_uniform_color([0,1,0])
        o3d.visualization.draw([pcd, pcd_new])
    
    # Start doing some crop
    crop_height, crop_width = color.shape[0], color.shape[1] # [720, 1280]
    crop_color = np.zeros((crop_height, crop_width, 3), dtype=color.dtype)
    crop_depth = np.zeros((crop_height, crop_width), dtype=depth.dtype)
    crop_intr = new_intr.copy()
    new_cx, new_cy = new_intr[0, 2], new_intr[1, 2]

    padding = (crop_width - crop_height) // 2

    if cut_mode == "bottom":
        color_cut = new_color[crop_width - crop_height : crop_width]
        depth_cut = new_depth[crop_width - crop_height : crop_width]
        crop_cx, crop_cy = new_cx + padding, new_cy - padding - padding
    elif cut_mode == "center":
        color_cut = new_color[padding : crop_width - padding]
        depth_cut = new_depth[padding : crop_width - padding]
        crop_cx, crop_cy = new_cx + padding, new_cy - padding
    elif cut_mode == "top":
        color_cut = new_color[0 : crop_width - 2 * padding]
        depth_cut = new_depth[0 : crop_width - 2 * padding]
        crop_cx, crop_cy = new_cx + padding, new_cy - padding + padding
    else:
        raise NotImplementedError
    crop_color[:, padding : crop_width - padding ] = color_cut
    crop_depth[:, padding : crop_width - padding ] = depth_cut
    crop_intr[0, 2] = crop_cx
    crop_intr[1, 2] = crop_cy
    return crop_color, crop_depth, crop_intr, T_head_crop


def visualize_sphere_o3d(center, color=[1, 0, 0], size=0.03):
    # center
    center_o3d = o3d.geometry.TriangleMesh.create_sphere()
    center_o3d.compute_vertex_normals()
    center_o3d.scale(size, [0, 0, 0])
    center_o3d.translate(center)
    center_o3d.paint_uniform_color(color)
    return center_o3d

def visualize_3d_trajectory(trajectory, size=0.03, cmap_name="plasma", invert=False):
    vis_o3d = []
    traj_color = get_heatmap(
        np.arange(len(trajectory)), cmap_name=cmap_name, invert=invert
    )
    for i, traj_point in enumerate(trajectory):
        vis_o3d.append(visualize_sphere_o3d(traj_point, color=traj_color[i], size=size))
    return vis_o3d


def visualize_points(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd