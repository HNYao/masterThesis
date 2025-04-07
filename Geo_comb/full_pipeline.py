"""
    predict one case
    1. GeoL_net predicts affordance heatmap
    2. GeoL_diffuser predicts 4d pose
"""
import sys
sys.path.append("thirdpart/GroundingDINO")
from GeoL_net.core.registry import registry
import open3d as o3d
import torch
from PIL import Image
from pointnet2_ops import pointnet2_utils
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from groundingdino.util.inference import load_model, load_image, predict, annotate # type: ignore
import groundingdino.datasets.transforms as GDinoT # type: ignore
import cv2
import numpy as np
import torch
import json
import os
from GeoL_net.dataset_gen.RGBD2PC import backproject, visualize_points
from GeoL_net.models.modules import ProjectColorOntoImage_v3, ProjectColorOntoImage
from GeoL_diffuser.models.guidance import *
from GeoL_diffuser.models.utils.fit_plane import *
from scipy.spatial.distance import cdist
from matplotlib import cm
import torchvision.transforms as T
from GeoL_net.gpt.gpt import chatgpt_condition, chatgpt_select_id, chatgpt_selected_plan
from GeoL_diffuser.algos.pose_algos import PoseDiffusionModel
import yaml
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as SciR
from torchvision.ops import box_convert
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import copy
import random
from glob import glob

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def predict_depth(depth_model, rgb_origin, intr, input_size = (616, 1064)):
    intrinsic = [intr[0, 0], intr[1, 1],
                 intr[0, 2], intr[1, 2]]  # fx, fy, cx, cy
    # ajust input size to fit pretrained model
    # keep ratio resize
    # input_size = (544, 1216) # for convnext model
    h, w = rgb_origin.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)),
                     interpolation=cv2.INTER_LINEAR)
    # remember to scale intrinsic, hold depth
    intrinsic = [intrinsic[0] * scale, intrinsic[1] *
                 scale, intrinsic[2] * scale, intrinsic[3] * scale]
    # padding to input_size
    padding = [123.675, 116.28, 103.53]
    h, w = rgb.shape[:2]
    pad_h = input_size[0] - h
    pad_w = input_size[1] - w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half,
                             pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

    # normalize
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb = torch.div((rgb - mean), std)
    rgb = rgb[None, :, :, :].cuda()

    ###################### canonical camera space ######################
    # inference
    with torch.no_grad():
        pred_depth, confidence, output_dict = depth_model.inference({
                                                                    'input': rgb})

    # un pad
    pred_depth = pred_depth.squeeze()
    confidence = confidence.squeeze()
    pred_depth = pred_depth[pad_info[0]: pred_depth.shape[0] -
                            pad_info[1], pad_info[2]: pred_depth.shape[1] - pad_info[3]]
    confidence = confidence[pad_info[0]: confidence.shape[0] -
                            pad_info[1], pad_info[2]: confidence.shape[1] - pad_info[3]]

    # upsample to original size
    pred_depth = torch.nn.functional.interpolate(
        pred_depth[None, None, :, :], rgb_origin.shape[:2], mode='nearest').squeeze()
    confidence = torch.nn.functional.interpolate(
        confidence[None, None, :, :], rgb_origin.shape[:2], mode='nearest').squeeze()
    ###################### canonical camera space ######################

    # de-canonical transform
    # 1000.0 is the focal length of canonical camera
    canonical_to_real_scale = intrinsic[0] / 1000.0
    pred_depth = pred_depth * canonical_to_real_scale  # now the depth is metric
    pred_depth = torch.clamp(pred_depth, 0, 5)
    pred_depth = pred_depth.cpu().numpy()
    confidence = confidence.cpu().numpy()
    return pred_depth, confidence

def retrieve_obj_mesh(obj_category, target_size=1, obj_mesh_dir="data_and_weights/mesh/"):
    obj_mesh_files = glob(os.path.join(obj_mesh_dir, obj_category, "*", "mesh.obj"))
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

def preprocess_image_groundingdino(image):
    transform = GDinoT.Compose(
        [
            GDinoT.RandomResize([800], max_size=1333),
            GDinoT.ToTensor(),
            GDinoT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.fromarray(image.astype(np.uint8))
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed

def get_heatmap(values, cmap_name="turbo", invert=False):
    if invert:
        values = -values
    values = (values - values.min()) / (values.max() - values.min())
    colormaps = cm.get_cmap(cmap_name)
    rgb = colormaps(values)[..., :3]  # don't need alpha channel
    return rgb

def generate_heatmap_target_point(batch, model_pred, intrinsics=None):
    """
    Generate heatmap for the model prediction and groud truth mask

    Parameters:
    batch: dict
        batch of data
    model_pred: torch.tensor (default: None) [b, num_points, 1]
        model prediction

    Returns:
    img_pred_list: list
        list of PIL images of the model prediction
    img_gt_list: list
        list of PIL images of the ground truth mask
    file_path: list
        list of file path
    phrase: list
        list of phrase
    """
    if not intrinsics:
        intrinsics = np.array(
            [[591.0125, 0.0, 322.525], [0.0, 590.16775, 244.11084], [0.0, 0.0, 1.0]]
        )
        # intrinsics = np.array([[619.0125 ,   0.     , 326  ], [  0.     , 619, 239], [  0.     ,   0.     ,   1.     ]])
        # intrinsics = np.array([[607.0125 ,   0.     , 636.525  ], [  0.     , 607.16775, 367.11084], [  0.     ,   0.     ,   1.     ]])
    img_pred_list = []
    img_gt_list = []
    img_rgb_list = batch["image"].cpu()  # the image of the scene [b,c, h, w]

    feat = model_pred.sigmoid()
    min_feat = feat.min(dim=1, keepdim=True)[0]
    max_feat = feat.max(dim=1, keepdim=True)[0]
    normalized_pred_feat = (feat - min_feat) / (max_feat - min_feat + 1e-6)

    turbo_colormap = cm.get_cmap(
        "turbo", 256
    )  # get the color map for the prediction and ground truth

    # normalize the prediction and ground truth

    normalized_pred_feat_np = normalized_pred_feat.cpu().detach().numpy()

    # get the color map for the prediction and ground truth [b, num_points, 3]

    color_pred_maps = turbo_colormap(normalized_pred_feat_np)[
        :, :, :, :3
    ]  # [b, num_points, 3] ignore alpha
    color_pred_maps = torch.from_numpy(color_pred_maps).squeeze(2).cpu()

    projector = ProjectColorOntoImage()

    pcs = []
    color_pred_list = []
    for i in range(batch["fps_points_scene"].shape[0]):
        depth = batch["depth"][i].cpu().numpy()
        fps_points_scene = batch["fps_points_scene"][i].cpu().numpy()
        # fps_colors = batch['fps_colors_scene'][i].cpu().numpy()
        points_scene, _ = backproject(
            depth,
            intrinsics,
            np.logical_and(depth > 0, depth < 2),
            NOCS_convention=False,
        )
        pcs.append(points_scene)

        distance_pred = cdist(points_scene, fps_points_scene)
        nearest_pred_idx = np.argmin(distance_pred, axis=1)
        color_pred_map = color_pred_maps[i]
        color_pred_scene = color_pred_map[nearest_pred_idx]
        color_pred_list.append(color_pred_scene)

    # pcs = torch.tensor(pcs, dtype=torch.float32) # list to tensor
    output_pred_img_list = []

    for i in range(len(pcs)):
        output_pred_img = projector(
            image_grid=img_rgb_list[i],
            query_points=torch.tensor(pcs[i]),
            query_colors=color_pred_list[i],
            intrinsics=intrinsics,
        )
        output_pred_img_list.append(output_pred_img)

    # merge the image and heatmap of prediction
    for i, pred_img in enumerate(output_pred_img_list):
        color_image = T.ToPILImage()(img_rgb_list[i].cpu())
        pil_img = T.ToPILImage()(pred_img.squeeze(0).cpu())

        image_np = np.clip(pil_img, 0, 255)

        color_image_np = np.floor(color_image)
        color_image_np = np.clip(color_image_np, 0, 255)
        color_image_np = np.uint8(color_image_np)

        image_np = cv2.addWeighted(image_np, 0.4, color_image_np, 0.6, 0.0)
        pil_image = Image.fromarray(np.uint8(image_np))
        img_pred_list.append(pil_image)

    # merge the image and heatmap of ground truth

    return img_pred_list, img_gt_list, batch["file_path"], batch["phrase"]

def generate_heatmap_pc(batch, model_pred, intrinsics=None, interpolate=False):
    if intrinsics is None:
        intrinsics = np.array(
            [
                [607.09912 / 2, 0.0, 636.85083 / 2],
                [0.0, 607.05212 / 2, 367.35952 / 2],
                [0.0, 0.0, 1.0],
            ]
        )

    # intrinsics = np.array([[591.0125 ,   0.     , 322.525  ],[  0.     , 590.16775, 244.11084],[  0.     ,   0.     ,   1.     ]])
    # intrinsics = np.array([[619.0125 ,   0.     , 326  ], [  0.     , 619, 239], [  0.     ,   0.     ,   1.     ]])
    # intrinsics = np.array([[607.0125 ,   0.     , 636.525  ], [  0.     , 607.16775, 367.11084], [  0.     ,   0.     ,   1.     ]])
    feat = model_pred.sigmoid()
    min_feat = feat.min(dim=1, keepdim=True)[0]
    max_feat = feat.max(dim=1, keepdim=True)[0]
    normalized_pred_feat = (feat - min_feat) / (max_feat - min_feat + 1e-6)
    turbo_colormap = cm.get_cmap(
        "turbo", 256
    )  # get the color map for the prediction and ground truth

    # normalize the prediction and ground truth
    normalized_pred_feat_np = normalized_pred_feat.cpu().detach().numpy()

    # get the color map for the prediction and ground truth [b, num_points, 3]
    color_pred_maps = turbo_colormap(normalized_pred_feat_np)[
        :, :, :, :3
    ]  # [b, num_points, 3] ignore alpha
    color_pred_maps = torch.from_numpy(color_pred_maps).squeeze(2).cpu()

    pcs = []
    color_pred_list = []
    color_img_list = []
    for i in range(batch["fps_points_scene"].shape[0]):
        depth = batch["depth"][i].cpu().numpy()
        image_color = batch["image"][i].permute(1, 2, 0).cpu().numpy()
        fps_points_scene = batch["fps_points_scene"][i].cpu().numpy()
        points_scene, idx = backproject(
            depth,
            intrinsics,
            np.logical_and(depth > 0, depth < 2),
            NOCS_convention=False,
        )
        image_color = image_color[idx[0], idx[1], :]
        pcs.append(points_scene)

        if interpolate:
            distance_pred = cdist(points_scene, fps_points_scene)

            # find the nearest 5 points in the scene points
            nearest_pred_idx = np.argmin(distance_pred, axis=1)
            nearest_10_idx = np.argsort(distance_pred, axis=1)[:, :10]

            # nearest_pred_idx = np.argmin(distance_pred, axis=1)
            color_pred_map = color_pred_maps[i]
            # color_pred_scene = color_pred_map[nearest_pred_idx]
            color_pred_scene = color_pred_map[nearest_10_idx].mean(axis=1)
            pred_value_thershold = 0.3  # for visualization
            pred_value = normalized_pred_feat_np[0, nearest_10_idx, :].mean(axis=1)

        else:
            distance_pred = cdist(points_scene, fps_points_scene)
            nearest_pred_idx = np.argmin(distance_pred, axis=1)
            color_pred_map = color_pred_maps[i]
            color_pred_scene = color_pred_map[nearest_pred_idx]
            pred_value_thershold = 0.3

        color_pred_list.append(color_pred_scene)
        color_img_list.append(image_color / 255)

    for i, pc in enumerate(pcs):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd.colors = o3d.utility.Vector3dVector(
            color_pred_list[i].cpu().numpy() #* 0.3 + color_img_list[i]* 255 * 0.7
        )
        #o3d.io.write_point_cloud(f"test_front.ply", pcd)
        o3d.visualization.draw_geometries([pcd])

    return pcd

def generate_heatmap_feature_pc(batch, model_pred, intrinsics=None, interpolate=False):
    if intrinsics is None:
        intrinsics = np.array(
            [
                [607.09912 / 2, 0.0, 636.85083 / 2],
                [0.0, 607.05212 / 2, 367.35952 / 2],
                [0.0, 0.0, 1.0],
            ]
        )

    # intrinsics = np.array([[591.0125 ,   0.     , 322.525  ],[  0.     , 590.16775, 244.11084],[  0.     ,   0.     ,   1.     ]])
    # intrinsics = np.array([[619.0125 ,   0.     , 326  ], [  0.     , 619, 239], [  0.     ,   0.     ,   1.     ]])
    # intrinsics = np.array([[607.0125 ,   0.     , 636.525  ], [  0.     , 607.16775, 367.11084], [  0.     ,   0.     ,   1.     ]])

    normalized_pred_feat = model_pred  # [b, num_points, 3]


    # normalize the prediction and ground truth
    normalized_pred_feat_np = normalized_pred_feat

    # get the color map for the prediction and ground truth [b, num_points, 3]

    color_pred_maps = torch.from_numpy(normalized_pred_feat_np)

    pcs = []
    color_pred_list = []
    color_img_list = []
    for i in range(batch["fps_points_scene"].shape[0]):
        depth = batch["depth"][i].cpu().numpy()
        image_color = batch["image"][i].permute(1, 2, 0).cpu().numpy()
        fps_points_scene = batch["fps_points_scene"][i].cpu().numpy()
        points_scene, idx = backproject(
            depth,
            intrinsics,
            np.logical_and(depth > 0, depth < 2),
            NOCS_convention=False,
        )
        image_color = image_color[idx[0], idx[1], :]
        pcs.append(points_scene)

        if interpolate:
            distance_pred = cdist(points_scene, fps_points_scene)

            # find the nearest 5 points in the scene points
            nearest_pred_idx = np.argmin(distance_pred, axis=1)
            nearest_10_idx = np.argsort(distance_pred, axis=1)[:, :10]

            # nearest_pred_idx = np.argmin(distance_pred, axis=1)
            color_pred_map = color_pred_maps[i]
            # color_pred_scene = color_pred_map[nearest_pred_idx]
            color_pred_scene = color_pred_map[nearest_10_idx].mean(axis=1)
            pred_value_thershold = 0.3  # for visualization
            pred_value = normalized_pred_feat_np[0, nearest_10_idx, :].mean(axis=1)

        else:
            distance_pred = cdist(points_scene, fps_points_scene)
            nearest_pred_idx = np.argmin(distance_pred, axis=1)
            color_pred_map = color_pred_maps[i]
            color_pred_scene = color_pred_map[nearest_pred_idx]
            pred_value_thershold = 0.3

        color_pred_list.append(color_pred_scene)
        color_img_list.append(image_color)

    for i, pc in enumerate(pcs):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd.colors = o3d.utility.Vector3dVector(
            color_pred_list[i].cpu().numpy() * 1 + color_img_list[i] * 0
        )
        o3d.io.write_point_cloud(f"test_front.ply", pcd)
        #o3d.visualization.draw_geometries([pcd])

def is_red(color, tolerance=0.1):
    return color[0] > 1 - tolerance and color[1] < tolerance and color[2] < tolerance


def visualize_xy_pred_points(pred, batch, intrinsics=None):
    """
    visualize the predicted xy points on the scene points

    Parameters:
    points: torch.tensor [num_preds=8, 3]
        the predicted xy points
    batch

    Returns:
    None
    """
    depth = batch["depth"][0].cpu().numpy()
    image = batch["image"][0].permute(1, 2, 0).cpu().numpy()
    points = pred["pose_xyR_pred"]  # [1, N, 3]
    guide_cost = pred["guide_losses"]["affordance_loss"]  # [1, N]

    if intrinsics is None:
        intrinsics = np.array(
            [[619.0125, 0.0, 326.525], [0.0, 619.16775, 239.11084], [0.0, 0.0, 1.0]]
        )

    points_scene, idx = backproject(
        depth,
        intrinsics,
        np.logical_and(depth > 0, depth < 2000),
        NOCS_convention=False,
    )
    image_color = image[idx[0], idx[1], :] / 255

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_scene)

    colors = np.zeros((points_scene.shape[0], 3))


    points = points.cpu().numpy()
    guide_cost = guide_cost.cpu().numpy()

    points = points[0]
    guide_cost = guide_cost[0]


    distances = np.sqrt(
        ((points_scene[:, :2][:, None, :] - points[:, :2]) ** 2).sum(axis=2)
    )

    scenepts_to_anchor_dist = np.min(distances, axis=1)  # [num_points]
    scenepts_to_anchor_id = np.argmin(distances, axis=1)  # [num_points]
    topk_points_id = np.argsort(scenepts_to_anchor_dist, axis=0)[: points.shape[0]]
    tokk_points_id_corr_anchor = scenepts_to_anchor_id[topk_points_id]

    guide_cost = guide_cost[tokk_points_id_corr_anchor]
    guide_cost_color = get_heatmap(guide_cost[None])[0]

    colors[topk_points_id] = [1, 0, 0]
    pcd.colors = o3d.utility.Vector3dVector(colors * 0.3 + image_color * 0.7 * 255)


    points_for_place = points_scene[topk_points_id]

    points_for_place_goal = np.mean(points_for_place, axis=0)
    print("points_for_place_goal:", points_for_place_goal)

    vis = [pcd]
    # print("points_for_place_goal:", points_for_place_goal)
    for ii, pos in enumerate(points_for_place):
        pos_vis = o3d.geometry.TriangleMesh.create_sphere()
        pos_vis.compute_vertex_normals()
        pos_vis.scale(0.01, [0, 0, 0])
        pos_vis.translate(pos[:3])
        vis_color = guide_cost_color[ii]
        pos_vis.paint_uniform_color(vis_color)

        vis.append(pos_vis)
    o3d.visualization.draw(vis)

def apply_kmeans_to_affordance(points, affordance_values, n_clusters=3, 
                               percentile_threshold=70, dist_factor=1):
    """
    Filter out low affordance points using K-means clustering.
    
    Parameters:
    -----------
    points : torch.Tensor or np.ndarray
        Point cloud data of shape [N, 3]
    affordance_values : torch.Tensor or np.ndarray
        Affordance values of shape [N, 1] or [N]
    n_clusters : int
        Number of clusters for K-means
    percentile_threshold : float
        Percentile threshold for filtering low affordance points
        
    Returns:
    --------
    filtered_points : np.ndarray
        Points with high affordance values
    filtered_affordance : np.ndarray
        Corresponding affordance values
    """
    # Convert to numpy if tensors
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(affordance_values, torch.Tensor):
        affordance_values = affordance_values.cpu().numpy()
    
    # Reshape affordance values if needed
    if len(affordance_values.shape) == 2:
        affordance_values = affordance_values.squeeze(1)
    
    # Calculate threshold
    threshold = np.percentile(affordance_values, percentile_threshold)
    
    # Get high affordance points
    high_affordance_mask = affordance_values > threshold
    high_affordance_points = points[high_affordance_mask]
    high_affordance_values = affordance_values[high_affordance_mask]
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(high_affordance_points)
    
    # Get cluster centers and labels
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # Compute the pairwise distance between the points and the cluster centers
    pairwise_distance = np.linalg.norm(points[:, None, :] - cluster_centers[None, :, :], axis=2)
    # Find the nearest cluster center for each point
    nearest_cluster_center = np.argmin(pairwise_distance, axis=1)
    dist = np.min(pairwise_distance, axis=1)
    dist = dist ** dist_factor
    dist *= -1
    affordance_values_new = (dist - np.min(dist)) / (np.max(dist) - np.min(dist))
    assert len(affordance_values_new) == len(points)
    return affordance_values_new
    
def generate_affordance_direct(
        data_batch
):
    anchor_position = data_batch["anchor_position"]
    points_scene = data_batch["pc_position"]
    distance = torch.norm(points_scene - anchor_position, dim=-1)
    affordance_value = torch.clamp(1 - (distance / 0.2) **0.9, min=-10, max=10)
    affordance_value =  affordance_value.view(1, 2048, 1)

    return affordance_value 



def prepare_data_batch(rgb_image, 
                       depth_image,
                       intrinsics,
                       target_name, 
                       target_box,
                       direction_text,
                       to_tensor = True):
    
    def data_batch_to_tensor(data_batch):
        for k, v in data_batch.items():
            if not isinstance(v, np.ndarray):
                continue
            elif "image" in k:
                data_batch[k] = T.ToTensor()(v)
            else:
                data_batch[k] = torch.from_numpy(v).float()

    data_batch = {}
    rgb_image = rgb_image[:, :, [2, 1, 0]].copy() # BGR to RGB
    depth_image = depth_image.astype(np.float32) / 1000.0
    points_scene, points_scene_idx = backproject(
        depth_image,
        intrinsics,
        np.logical_and(depth_image > 0, depth_image < 2),
        NOCS_convention=False,
    )
    scene_pcd_colors = rgb_image[points_scene_idx[0], points_scene_idx[1]] / 255.0
    scene_pcd = visualize_points(points_scene, scene_pcd_colors)
    
    # fps the points_scene
    scene_pcd_tensor = torch.tensor(np.asarray(scene_pcd.points), dtype=torch.float32).unsqueeze(0)
    scene_pcd_colors = np.asarray(scene_pcd.colors)
    fps_indices_scene = pointnet2_utils.furthest_point_sample(
            scene_pcd_tensor.contiguous().cuda(), 2048
        )
    fps_indices_scene_np = fps_indices_scene.squeeze(0).cpu().numpy()
    fps_points_scene_from_original = points_scene[fps_indices_scene_np]
    fps_colors_scene_from_original = scene_pcd_colors[fps_indices_scene_np]
    
    # Acquire the location of the anchor object
    box_mask = np.zeros((rgb_image.shape[0], rgb_image.shape[1]))
    x1, y1, x2, y2 = target_box
    width, height = x2 - x1, y2 - y1
    scale = 1.0
    if "Left" in direction_text:
        x2 = int(x1 + width * 0.25)
    if "Right" in direction_text:
        x1 = int(x2 - width * 0.25)
    if "Front" in direction_text:
        y1 = int(y2 - height * 0.25)
        scale = 0.8
    if "Behind" in direction_text:
        y2 = int(y1 + height * 0.25)
    if "On" in direction_text:
        y1 = int(y1 + height * 0.4)
        x1 = int(x1 + width * 0.4)
        y2 = int(y2 - height * 0.4)
        x2 = int(x2 - width * 0.4)
    # cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # plt.imshow(rgb_image)
    # plt.show()
    box_mask[y1:y2, x1:x2] = 1
    points_anchor_scene, _ = backproject(
        depth_image,
        intrinsics,
        box_mask,
        NOCS_convention=False,
    )
    anchor_position = np.median(points_anchor_scene, axis=0) * scale
    data_batch["phrase"] =  ["n/a"]
    data_batch["file_path"] = ["n/a"]
    data_batch["mask"] =  ["n/a"]
    data_batch["reference_obj"] = [target_name]
    data_batch["direction_text"] = [direction_text]
    
    data_batch["anchor_position"] = anchor_position
    data_batch["image"] = rgb_image
    data_batch["depth"] = depth_image
    data_batch["fps_points_scene"] = fps_points_scene_from_original
    data_batch["fps_colors_scene"] = fps_colors_scene_from_original
    data_batch["pc_position"] = fps_points_scene_from_original
    if to_tensor:
        data_batch_to_tensor(data_batch)

    return data_batch
    
    

def detect_object(
    detection_model,
    image,
    text_prompt,
    use_chatgpt=False,
):
    TEXT_PROMPT = text_prompt
    BOX_TRESHOLD = 0.20 # 0.35
    TEXT_TRESHOLD = 0.20 # 0.25

    image_source, image_input = preprocess_image_groundingdino(image)
    boxes, logits, phrases = predict(
        model=detection_model,
        image=image_input,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
    )

    phrases = [f"id{id}" for id in range(len(phrases))]

    _, h, w = image_input.shape
    boxes_xyxy = boxes * torch.Tensor([w, h, w, h])
    boxes_xyxy = box_convert(boxes=boxes_xyxy, in_fmt="cxcywh", out_fmt="xyxy").numpy()


    if len(phrases) > 0:
        # print("orignal boxes cxcy:", ori_boxes, ori_boxes[0][0], ori_boxes[0][1])
        annotated_frame = annotate(
            image_source=image_source, boxes=boxes, logits=logits, phrases=phrases
        )
        write_path = ".tmp/annotated_detection_chatgpt.jpg"
        cv2.imwrite(write_path, annotated_frame)

        if use_chatgpt:
            id_selected_vlm= chatgpt_select_id(write_path, TEXT_PROMPT)
            selected_id = int(id_selected_vlm)
        else:
            selected_id = 0
    else: 
        print("No object detected")
        return None, None
            

    selected_box_xyxy = boxes_xyxy[selected_id].astype(np.int32)
    selected_phrase = phrases[selected_id]
    return selected_box_xyxy, selected_phrase


def detect_object_with_vlm(
    detection_model,
    image,
    use_chatgpt=True,
):
    """
    Detect object with VLM: GroudingDIno -> chatgpt select anchor obj_name, direction, bbox_id -> bbox
    """

    TEXT_PROMPT = "detergent, sink, kettle"
    #TEXT_PROMPT = 'mug, cup, keyboard, laptop, white cup' 
    # TEXT_PROMPT = "book, monitor, screen, laptop, display, mouse, keyboard, clock, remote, headphone, camera, printer, scanner"
    # TEXT_PROMPT = "plate, cookie" #, fork, spoon, knife, wine, napkin, box, paper, food"
   
   
    BOX_TRESHOLD = 0.25# 0.35
    TEXT_TRESHOLD = 0.25 # 0.25


    image_source, image_input = preprocess_image_groundingdino(image)
    boxes, logits, phrases = predict(
        model=detection_model,
        image=image_input,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
    )

    #phrases = [f"id{id}" for id in range(len(phrases))]

    _, h, w = image_input.shape
    boxes_xyxy = boxes * torch.Tensor([w, h, w, h])
    boxes_xyxy = box_convert(boxes=boxes_xyxy, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    
    # filter out the bboex whose area is larger than 80% of the whole image
    image_area = w * h
    boxes_area = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
    area_mask = boxes_area < image_area * 0.8
    boxes_xyxy = boxes_xyxy[area_mask]
    boxes = boxes[area_mask]
    logits = logits[area_mask]
    phrases = [f"id{id}" for id in range(boxes_xyxy.shape[0])]
    
    if len(phrases) > 0:
        # print("orignal boxes cxcy:", ori_boxes, ori_boxes[0][0], ori_boxes[0][1])
        annotated_frame = annotate(
            image_source=image_source, boxes=boxes, logits=logits, phrases=phrases
        )
        write_path = ".tmp/annotated_detection_chatgpt_direct.jpg"
        cv2.imwrite(write_path, annotated_frame[:, :, [2, 1, 0]])

        target_obj_list, direction_list, bbox_id_list = chatgpt_selected_plan(write_path)

        for i, bbox_id in enumerate(bbox_id_list):
            bbox_id_list[i] = int(bbox_id)
        # if use_chatgpt:
        #     id_selected_vlm= chatgpt_select_id(write_path, TEXT_PROMPT)
        #     selected_id = int(id_selected_vlm)
        # else:
        #     selected_id = 0
    
   

    selected_box_xyxy = [boxes_xyxy[id].astype(np.int32) for id in bbox_id_list]
    boxes_xyxy = [boxes_xyxy[id].astype(np.int32) for id in range(len(boxes_xyxy))] # all boxes


    # get the boxes of object with the "On" direction and delete it from the boxes_xyxy
    if "On" in direction_list:
        boxes_with_on = [boxes_xyxy[id].astype(np.int32) for id, direction in zip(bbox_id_list, direction_list) if direction == "On"]
        boxes_xyxy = [item for item in boxes_xyxy if not any(np.array_equal(item, box) for box in boxes_with_on)]


    
    #selected_box_xyxy = boxes_xyxy[selected_id].astype(np.int32)

    return target_obj_list, direction_list, selected_box_xyxy, boxes_xyxy
 
def full_pipeline_v2(
        model_detection,
        model_affordance,
        model_diffuser,
        rgb_image,
        depth_image,
        obj_mesh,
        intrinsics,
        target_names=[],
        direction_texts = [],
        use_vlm = False,
        fast_vlm_detection = True,
        use_kmeans = True,
        disable_rotation = False,
        visualize_affordance = False,
        visualize_diff = False,
        visualize_final_obj = False,
        rendering = False,
):
    #1 use chatgpt or directly provide target_name and direction_text
    assert (len(target_names) == len(direction_texts)> 0) or use_vlm, "Please provide target_name and direction_text"
    points_scene, scene_idx = backproject(
        depth_image / 1000.0,
        intrinsics,
        np.logical_and(depth_image / 1000.0 > 0, depth_image / 1000.0 < 2),
        NOCS_convention=False,
    )
    colors_scene = rgb_image[scene_idx[0], scene_idx[1]][..., [2,1,0]] / 255.0
    pcd_scene = visualize_points(points_scene, colors_scene) 
    

    #### 2 use_vlm 
    all_bboxes = None
    if use_vlm:
        if fast_vlm_detection:
        # option1: GroundingDINO -> chatgpt select anchor obj_name, direction, bbox_id -> bbox, else, provided target_name and direction_text -> GroudingDINO -> bbox
            target_names, direction_texts, selected_boxes, all_bboxes = detect_object_with_vlm(model_detection, rgb_image)
        # import pdb; pdb.set_trace()
        # option2: chatgpt -> target_name, direction_text, bbox_id_list -> GDino -> bbox, idx -> chatgpt -> id
        else:
            temp_rgb_path = ".tmp/temp_rgb.jpg"
            temp_depth_path = ".tmp/temp_depth.png"
            os.makedirs(".tmp", exist_ok=True)
            cv2.imwrite(temp_rgb_path, rgb_image.astype(np.uint8)) # BGR
            cv2.imwrite(temp_depth_path, depth_image.astype(np.uint16))
            # TODO: multi hypotheses
            # Save temporary image for VLM processing
            target_names, direction_texts = chatgpt_condition(
                        temp_rgb_path, "object_placement"
                    )
            print("====> Using VLM to parse the target object and direction...")
        # target_name = [target_name] 
        # direction_text = [direction_text]

    #3 use GroundingDINO to detect the target object
    pred_affordance_list = []
    for i in range(len(target_names)):
        if fast_vlm_detection:
            selected_box = selected_boxes[i]
            selected_phrase = target_names[i]
        else:
            selected_box, selected_phrase = detect_object(
                model_detection, rgb_image, target_names[i], use_chatgpt=use_vlm
            )
        


        # prepare the data batch
        data_batch = prepare_data_batch(rgb_image, depth_image, intrinsics, target_names[i], selected_box, direction_texts[i], to_tensor=True)
        for key, val in data_batch.items():
            if not isinstance(val, torch.Tensor):
                continue
            data_batch[key] = val.float().to("cuda")[None]
        
        # implement the "On" direction
        if direction_texts[i] == "On":
            affordance_pred = generate_affordance_direct(data_batch)
        
        else:
            with torch.no_grad():
                affordance_pred = model_affordance(batch=data_batch)["affordance"].squeeze(1)
        affordance_pred_sigmoid = affordance_pred.sigmoid().cpu().numpy()
        affordance_thershold = -np.inf
        fps_points_scene_from_original = data_batch["fps_points_scene"][0]

        fps_points_scene_affordance = fps_points_scene_from_original[
            affordance_pred_sigmoid[0][:, 0] > affordance_thershold
        ]
        fps_points_scene_affordance = fps_points_scene_affordance.cpu().numpy()
        min_bound_affordance = np.append(
            np.min(fps_points_scene_affordance, axis=0), -1
        )
        max_bound_affordance = np.append(
            np.max(fps_points_scene_affordance, axis=0), 1
        )
        # sample 512 points from fps_points_scene_affordance
        fps_points_scene_affordance = fps_points_scene_affordance[
            np.random.choice(
                fps_points_scene_affordance.shape[0], 512, replace=True
            )
        ]  # [512, 3]
        pred_affordance_list.append(affordance_pred) # make the affordance map smoother
        
    # merge the affordance prediction to the scene point cloud
    # merge the affordance prediction: use the GMM or the mean
    pred_affordance_merge = torch.cat(pred_affordance_list, dim=0)
    pred_affordance_merge_mean = pred_affordance_merge.mean(dim=0, keepdim=True)
    pred_affordance_merge = torch.cat([pred_affordance_merge_mean, pred_affordance_merge], dim=0)
    pred_affordance_merge, _ = (pred_affordance_merge).max(dim=0, keepdim=True)
    pred_affordance_fine = pred_affordance_merge.clone()
    data_batch['affordance_fine'] = pred_affordance_fine
    # pred_affordance_merge = pred_affordance_merge.mean(dim=0, keepdim=True)
    
    # normalize the affordance prediction
    if use_kmeans:
        # get the max affordance point and the sample points
        pred_affordance_np = pred_affordance_merge.cpu().numpy()[0, : ,0] # [B, N, 1]
        fps_points_scene = data_batch["fps_points_scene"].cpu().numpy()[0] # [N, 3]
        # Filter points
        pred_affordance_merge = apply_kmeans_to_affordance(
            fps_points_scene, 
            pred_affordance_np,
            n_clusters=len(target_names),  # Adjust based on how many distinct regions you want
            percentile_threshold=95,  # Adjust based on how strict you want the filtering
            dist_factor=0.5
        )
        pred_affordance_merge = torch.from_numpy(pred_affordance_merge).float().cuda()[None, :, None]
    
    pred_affordance_merge = (pred_affordance_merge - pred_affordance_merge.min()) / \
        (pred_affordance_merge.max() - pred_affordance_merge.min()) 
    pred_affordance_fine = (pred_affordance_fine - pred_affordance_fine.min() ) / \
        (pred_affordance_fine.max() - pred_affordance_fine.min())
    
    if visualize_affordance:
        generate_heatmap_pc(data_batch, pred_affordance_fine, intrinsics, interpolate=False) # visualize single case
        generate_heatmap_pc(data_batch, pred_affordance_merge, intrinsics, interpolate=False) # visualize single case

    # 7 normalize the prediction for the diffuser
    pred_affordance_merge = affordance_pred.sigmoid()
    pred_affordance_merge = (pred_affordance_merge - pred_affordance_merge.min()) / (pred_affordance_merge.max() - pred_affordance_merge.min())

    # 8 prepare the data for the diffuser, especially the object mesh
    data_batch['affordance'] = pred_affordance_merge.to("cuda")
    # obj_mesh = o3d.io.read_triangle_mesh(obj_to_place_path)
    # obj_mesh.compute_vertex_normals()
    obj_pc = obj_mesh.sample_points_uniformly(512)
    obj_pc = np.asarray(obj_pc.points)
    
    data_batch['object_pc_position'] = torch.tensor(obj_pc, dtype=torch.float32).unsqueeze(0).to("cuda")
    data_batch['gt_pose_xy_min_bound'] = torch.tensor(min_bound_affordance[...,:2], dtype=torch.float32).unsqueeze(0).to("cuda")
    data_batch['gt_pose_xy_max_bound'] = torch.tensor(max_bound_affordance[...,:2], dtype=torch.float32).unsqueeze(0).to("cuda")
    data_batch['gt_pose_xyR_min_bound'] = torch.tensor(np.delete(min_bound_affordance, obj=2, axis=0), dtype=torch.float32).unsqueeze(0).to("cuda")
    data_batch['gt_pose_xyR_max_bound'] = torch.tensor(np.delete(max_bound_affordance, obj=2, axis=0), dtype=torch.float32).unsqueeze(0).to("cuda")

    # Build the TSDF for collision avoidance guidance
    if all_bboxes is not None:
        obj_bbox_mask = np.zeros((rgb_image.shape[0], rgb_image.shape[1]))
        for bg_obj_bbox in all_bboxes:
            x1, y1, x2, y2 = bg_obj_bbox
            obj_bbox_mask[y1:y2, x1:x2] = 1
    else:
        obj_bbox_mask = np.ones((rgb_image.shape[0], rgb_image.shape[1]))
    T_plane, plane_model = get_tf_for_scene_rotation(points_scene)
    vol_bnds = np.zeros((3, 2))
    view_frust_pts = get_view_frustum(depth_image / 1000.0, intrinsics, np.eye(4))
    vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
    vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
    vol_bnds[:, 0] = vol_bnds[:, 0].min()
    vol_bnds[:, 1] = vol_bnds[:, 1].max()
    color_tsdf = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    tsdf = TSDFVolume(vol_bnds, voxel_dim=256, num_margin=20, unknown_free=False)
    tsdf.integrate(color_tsdf, depth_image * obj_bbox_mask / 1000.0, intrinsics, np.eye(4))

    # mesh = tsdf.get_mesh()
    # mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh])

    # if T_camera_plane is not None:
    #     T_plane[:3, :3] = T_camera_plane[:3, :3]   

    data_batch['vol_bnds'] = torch.tensor(vol_bnds, dtype=torch.float32).unsqueeze(0).to("cuda")
    data_batch['tsdf_vol'] = torch.tensor(tsdf._tsdf_vol, dtype=torch.float32).unsqueeze(0).to("cuda")
    data_batch["T_plane"] = torch.tensor(T_plane, dtype=torch.float32).unsqueeze(0).to("cuda")
    data_batch['intrinsics'] = torch.tensor(intrinsics, dtype=torch.float32).unsqueeze(0).to("cuda")
    data_batch['color_tsdf'] = torch.tensor(color_tsdf, dtype=torch.float32).unsqueeze(0).to("cuda")
    data_batch['intrinsics'] = torch.tensor(intrinsics, dtype=torch.float32).unsqueeze(0).to("cuda")
  
    # 9 predict the xyR 
    pred = model_diffuser(data_batch, num_samp=1, class_free_guide_w=-0.1, apply_guidance=True, guide_clean=True)
    if visualize_diff:
        visualize_xy_pred_points(pred, data_batch, intrinsics=intrinsics)

    # 10 select topk points
    topk = 10
    target_shape = pred['pose_xyR_pred'].shape[1] 


    guide_affordance_loss = pred["guide_losses"]["affordance_loss"].cpu().numpy().reshape(target_shape) # [BN, ]
    guide_collision_loss = pred["guide_losses"]["collision_loss"].cpu().numpy().reshape(target_shape) # [BN, ]
    guide_loss_total = pred["guide_losses"]["loss"].cpu().numpy().reshape(target_shape) # [BN, ]
        
    # guide_distance_error = pred["guide_losses"]["distance_error"].cpu().numpy().reshape(target_shape) # [BN, ]
    pred_points = pred['pose_xyR_pred'].cpu().numpy().reshape(target_shape, -1)
    # guide_loss_color = get_heatmap(guide_collision_loss[None])[0] # [N,]
    # min_colliion_loss = guide_collision_loss.min()
    # guide_affordance_loss[guide_collision_loss > min_colliion_loss] = np.inf
    # guide_loss_total = guide_affordance_loss + guide_collision_loss
    # Select the topk points with the lowest guide loss
    min_guide_loss_idx = np.argsort(guide_loss_total)[:topk]
    pred_points = pred_points[min_guide_loss_idx]
    guide_loss_total = guide_loss_total[min_guide_loss_idx] # [N,]
    guide_loss_color = get_heatmap(guide_loss_total[None])[0] # [N, 3]
    
    # print("min distance loss:", guide_distance_error[min_guide_loss_idx])
    print("min collision loss:", guide_collision_loss[min_guide_loss_idx])
    print("pred xyR:", pred_points) 
  
    pred_xyz_all, pred_r_all = [], []
    for i in range(len(pred_points)):
        pred_xy = pred_points[i,:2]
        pred_r = pred_points[i, 2]
        pred_r = pred_r * 180 / np.pi
        pred_z = (-plane_model[0] * pred_xy[0] - plane_model[1] * pred_xy[1] - plane_model[3]-0.01) / plane_model[2]
        pred_xyz = np.append(pred_xy, pred_z)
        if disable_rotation:
            pred_r = 0
        pred_xyz = pred_xyz  
        pred_xyz_all.append(pred_xyz)
        pred_r_all.append(pred_r)
        
    pred_xyz_all = np.array(pred_xyz_all) # [N, 3]
    pred_r_all = np.array(pred_r_all) # [N,]
    pred_cost = guide_loss_total # [N,]

    min_point_coord = np.min(points_scene, axis=0) * 1.2  # [3,]
    max_point_coord = np.max(points_scene, axis=0) * 0.8  # [3,]
    pred_xyz_all = np.clip(pred_xyz_all, min_point_coord, max_point_coord)
    
    if visualize_final_obj: 
        #11 add mesh obj to the scene
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        vis_o3d = [pcd_scene, coordinate_frame]

        for i in range(len(pred_points)):
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
    
    if use_vlm and not fast_vlm_detection:
        os.remove(temp_rgb_path)  # Clean up temporary file
        os.remove(temp_depth_path)  # Clean up temporary file
        
    return pred_xyz_all, pred_r_all, pred_cost



if __name__ == "__main__":
    # INTRINSICS = np.array([[591.0125 ,   0.     , 322.525  ],[  0.     , 590.16775, 244.11084],[  0.     ,   0.     ,   1.     ]])
    # INTRINSICS = np.array([[591.0125 ,   0.     , 636  ],[  0.     , 590.16775, 367],[  0.     ,   0.     ,   1.     ]])
    # intr = np.array([[591.0125 ,   0.     , 322.525  ],[  0.     , 590.16775, 244.11084],[  0.     ,   0.     ,   1.     ]])
    #INTRINSICS = np.array([[619.0125 ,   0.     , 326.525  ],[  0.     , 619.16775, 239.11084],[  0.     ,   0.     ,   1.     ]]) #realsense
    #INTRINSICS = np.array([[607.0125 ,   0.     , 636.525  ], [  0.     , 607.16775, 367.11084], [  0.     ,   0.     ,   1.     ]]) # kinect
    #INTRINSICS = np.array([[607.09912/2 , 0. , 636.85083/2 ], [0., 607.05212/2, 367.35952/2], [0.0, 0.0, 1.0]])
    INTRINSICS = np.array([[911.09,   0.  , 657.44],[  0.  , 910.68,  346.58],[  0.  ,   0.  ,   1.  ]]) # realsense
    # congiguration
    # scene_pcd_file_path = "dataset/scene_RGBD_mask_direction_mult/id10_1/clock_0001_normal/mask_Behind.ply"
    # blendproc dataset
    #rgb_image_file_path = "dataset/scene_RGBD_mask_v2_kinect_cfg/id110/eye_glasses_0003_black/with_obj/test_pbr/000000/rgb/000000.jpg"
    #depth_image_file_path = "dataset/scene_RGBD_mask_v2_kinect_cfg/id110/eye_glasses_0003_black/with_obj/test_pbr/000000/depth/000000.png"

    # kinect data
    # rgb_image_file_path = "dataset/kinect_dataset/color/000025.png"
    # depth_image_file_path = "dataset/kinect_dataset/depth/000025.png"

    # realsense data
    rgb_image_file_path = "data_and_weights/realworld_2103/color/000082.png"
    depth_image_file_path = "data_and_weights/realworld_2103/depth/000082.png"

    # data from robot camera
    #rgb_image_file_path = "dataset/data_from_robot/img/img_10.jpg"
    #depth_image_file_path = "dataset/data_from_robot/depth/depth_10.png"
    model_detection = load_model(
        "./thirdpart/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", 
        "./thirdpart/GroundingDINO/weights/groundingdino_swint_ogc.pth"
    )
    model_affordance_cls = registry.get_affordance_model("GeoL_net_v9")
    model_affordance = model_affordance_cls(
        input_shape=(3, 720, 1280),
        target_input_shape=(3, 128, 128),
        intrinsics=INTRINSICS,
    ).to("cuda")
    state_affordance_dict = torch.load("data_and_weights/ckpt_11.pth", map_location="cpu")
    model_affordance.load_state_dict(state_affordance_dict["ckpt_dict"])
    model_affordance.eval()

    # diffuser model
    guidance = CompositeGuidance()
    with open("config/baseline/diffusion.yaml", "r") as file:
        yaml_data = yaml.safe_load(file)
    config_diffusion = OmegaConf.create(yaml_data)
    model_diffuser_cls = PoseDiffusionModel
    model_diffuser = model_diffuser_cls(config_diffusion.model).to("cuda")
    state_diffusion_dict = torch.load("data_and_weights/ckpt_93.pth", map_location="cpu")
    model_diffuser.load_state_dict(state_diffusion_dict["ckpt_dict"])
    model_diffuser.nets["policy"].set_guidance(guidance)
    model_diffuser.eval()
    
    # Load the image and depth image, and object mesh
    rgb_image = cv2.imread(rgb_image_file_path)
    depth_image = cv2.imread(depth_image_file_path, -1)
    obj_mesh = retrieve_obj_mesh("cup", target_size=0.5)
    
    # DO the inference
    seed_everything(42)
    full_pipeline_v2(
        model_detection=model_detection,
        model_affordance=model_affordance,
        model_diffuser=model_diffuser,
        rgb_image=rgb_image,
        depth_image=depth_image,
        obj_mesh=obj_mesh,
        intrinsics=INTRINSICS,
        target_names=["white plate in the middle"],     #, "Monitor", "Monitor"],
        direction_texts=["On"],     #, "Left Front", "Right Front"],
        use_vlm=True,
        fast_vlm_detection=True,
        use_kmeans=True,
        visualize_affordance=False,
        visualize_diff=False,
        visualize_final_obj=True,
        rendering = False,
    )