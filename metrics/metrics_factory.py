
import numpy as np
import open3d as o3d
import torch
from GeoL_net.dataset_gen.RGBD2PC import backproject, visualize_points
from GeoL_diffuser.models.utils.fit_plane import *
import math

def position_metric(mask_prediction, batch):
    """
    calculate the distance between position_pred and position_gt
    """
    image = batch['image']
    depth = batch['depth']
    
    pc_positions = batch['fps_points_scene_from_original']
    pc_colors = batch['fps_colors_scene_from_original']

    # get the ground truth position
    max_indices = torch.argmax(pc_colors[:, :, 1], dim=1)
    max_positions = pc_positions[torch.arange(pc_positions.shape[0]), max_indices]

    # get the pred position, find the most highest n points
    batch_size, height, width = mask_prediction.shape
    n = 10
    prediction_flat = mask_prediction.view(batch_size, -1)
    topk_values, topk_indices = torch.topk(prediction_flat, n, dim=1) 
    row_indices = topk_indices // mask_prediction.shape[2]
    col_indices = topk_indices % mask_prediction.shape[2]

    batch_indices = torch.arange(batch_size).view(-1, 1).repeat(1, n).flatten()
    row_indices_flat = row_indices.flatten()
    col_indices_flat = col_indices.flatten()

    color_pred = torch.zeros((batch_size, height, width, 3), dtype=torch.uint8) 
    color_pred[batch_indices, row_indices_flat, col_indices_flat] = torch.tensor([0, 0, 255], dtype=torch.uint8)
    
    # get the point cloud from pred
    intrinsics = np.array([[591.0125 ,   0.     , 322.525  ],
                            [  0.     , 590.16775, 244.11084],
                            [  0.     ,   0.     ,   1.     ]])
    depth = depth.detach().cpu().numpy()
    color_pred = color_pred.detach().cpu().numpy()

    metrics_distance = []
    for i in range(batch_size):
        points_scene, _ = backproject(depth[i], intrinsics, depth[i]>0)
        pc_pred = visualize_points(points_scene, color_pred[i].reshape(-1, 3).astype(np.float64))
        points = np.asarray(pc_pred.points)
        colors = np.asarray(pc_pred.colors)
        blue_mask = (colors[:, 0] <= 0.1) & (colors[:, 1] <= 0.1) & (colors[:, 2] >= 0.9)

        blue_points = points[blue_mask]
        distances = np.linalg.norm(blue_points - max_positions[i].detach().cpu().numpy(), axis=1)
        metrics_distance.append(np.mean(distances))

    return metrics_distance 


def directional_sptail_metric(mask_prediction, batch):
    """
    calculate the directional spatial metric
    """
    anchor_points = batch['anchor_point']
    directions = batch['direction']
    image = batch['image']
    depth = batch['depth']
    pc_positions = batch['fps_points_scene_from_original']
    pc_colors = batch['fps_colors_scene_from_original']

    # get the ground truth position
    max_indices = torch.argmax(pc_colors[:, :, 1], dim=1)
    max_positions = pc_positions[torch.arange(pc_positions.shape[0]), max_indices]

    # get the pred position, find the most highest n points
    batch_size, height, width = mask_prediction.shape
    n = 10
    prediction_flat = mask_prediction.view(batch_size, -1)
    topk_values, topk_indices = torch.topk(prediction_flat, n, dim=1) 
    row_indices = topk_indices // mask_prediction.shape[2]
    col_indices = topk_indices % mask_prediction.shape[2]

    batch_indices = torch.arange(batch_size).view(-1, 1).repeat(1, n).flatten()
    row_indices_flat = row_indices.flatten()
    col_indices_flat = col_indices.flatten()

    color_pred = torch.zeros((batch_size, height, width, 3), dtype=torch.uint8) 
    color_pred[batch_indices, row_indices_flat, col_indices_flat] = torch.tensor([0, 0, 255], dtype=torch.uint8)
    
    # get the point cloud from pred
    intrinsics = np.array([[591.0125 ,   0.     , 322.525  ],
                            [  0.     , 590.16775, 244.11084],
                            [  0.     ,   0.     ,   1.     ]])
    depth = depth.detach().cpu().numpy()
    color_pred = color_pred.detach().cpu().numpy()

    metrics_direction = []
    for i in range(batch_size):
        points_scene, _ = backproject(depth[i], intrinsics, depth[i]>0)
        T_plane, plane_model = get_tf_for_scene_rotation(points_scene)
        pc_pred = visualize_points(points_scene, color_pred[i].reshape(-1, 3).astype(np.float64))
        points = np.asarray(pc_pred.points)
        colors = np.asarray(pc_pred.colors)
        blue_mask = (colors[:, 0] <= 0.1) & (colors[:, 1] <= 0.1) & (colors[:, 2] >= 0.9)

        blue_points = points[blue_mask] # predicted points

        # transform the predicted position and anchor point to the plane coordinate
        anchor_point = anchor_points[i].detach().cpu().numpy()
        anchor_point = np.dot(anchor_point, T_plane[:3, :3])
        blue_point = np.dot(blue_points, T_plane[:3, :3])

        # calculate the direction
        for k in range(len(blue_point)):
            direction_pred = determine_direction(anchor_point, blue_point[k])
            direction_gt = directions[i]
            if direction_pred == direction_gt:
                metrics_direction.append(1) # success
            else:
                metrics_direction.append(0) # fail
        success_rate = np.mean(metrics_direction)
    return metrics_direction



def receptacle_metric(mask_prediction, batch):
    """
    check if the prediction is on the receptacle (desk or table)
    """
    anchor_points = batch['anchor_point']
    directions = batch['direction']
    image = batch['image']
    depth = batch['depth']
    pc_positions = batch['fps_points_scene_from_original']
    pc_colors = batch['fps_colors_scene_from_original']

    # get the ground truth position
    max_indices = torch.argmax(pc_colors[:, :, 1], dim=1)
    max_positions = pc_positions[torch.arange(pc_positions.shape[0]), max_indices]

    # get the pred position, find the most highest n points
    batch_size, height, width = mask_prediction.shape
    n = 10
    prediction_flat = mask_prediction.view(batch_size, -1)
    topk_values, topk_indices = torch.topk(prediction_flat, n, dim=1) 
    row_indices = topk_indices // mask_prediction.shape[2]
    col_indices = topk_indices % mask_prediction.shape[2]

    batch_indices = torch.arange(batch_size).view(-1, 1).repeat(1, n).flatten()
    row_indices_flat = row_indices.flatten()
    col_indices_flat = col_indices.flatten()

    color_pred = torch.zeros((batch_size, height, width, 3), dtype=torch.uint8) 
    color_pred[batch_indices, row_indices_flat, col_indices_flat] = torch.tensor([0, 0, 255], dtype=torch.uint8)
    
    # get the point cloud from pred
    intrinsics = np.array([[591.0125 ,   0.     , 322.525  ],
                            [  0.     , 590.16775, 244.11084],
                            [  0.     ,   0.     ,   1.     ]])
    depth = depth.detach().cpu().numpy()
    color_pred = color_pred.detach().cpu().numpy()

    metrics_receptable = []
    for i in range(batch_size):
        points_scene, _ = backproject(depth[i], intrinsics, depth[i]>0)
        T_plane, plane_model = get_tf_for_scene_rotation(points_scene) # T_plane of the tilted desktop
        pc_pred = visualize_points(points_scene, color_pred[i].reshape(-1, 3).astype(np.float64))
        points = np.asarray(pc_pred.points)
        colors = np.asarray(pc_pred.colors)
        blue_mask = (colors[:, 0] <= 0.1) & (colors[:, 1] <= 0.1) & (colors[:, 2] >= 0.9)

        blue_points = points[blue_mask] # predicted points

        # transform the predicted position and anchor point to the plane coordinate
        anchor_point = anchor_points[i].detach().cpu().numpy()
        anchor_point = np.dot(anchor_point, T_plane[:3, :3])
        blue_point = np.dot(blue_points, T_plane[:3, :3])

        # get the plane model of the level desktop
        level_points_scene = np.dot(points_scene, T_plane[:3, :3])
        _, level_plane_model = get_tf_for_scene_rotation(level_points_scene)
        height_desktop = - level_plane_model[3] / level_plane_model[2]


        # calculate the direction
        for k in range(len(blue_point)):
            if abs(blue_point[k][2] - height_desktop) < 0.1:
                metrics_receptable.append(1)
            else:
                metrics_receptable.append(0)
        success_rate = np.mean(metrics_receptable)
    return metrics_receptable


def collision_metric(y_true, y_pred, object_pc, scene_pc):
    """
    calculate the collision metric between y_true and y_pred
    """
    pass

def determine_direction(p1, p2):
    x_diff = p2[0] - p1[0]
    y_diff = p2[1] - p1[1]
    
    angle = math.degrees(math.atan2(y_diff, x_diff))
    if angle < 0:
        angle += 360
    
    if 0 <= angle < 22.5 or 337.5 <= angle < 360:
        return "Left"
    elif 22.5 <= angle < 67.5:
        return "Left Behind"
    elif 67.5 <= angle < 112.5:
        return "Behind"
    elif 112.5 <= angle < 157.5:
        return "Right Behind"
    elif 157.5 <= angle < 202.5:
        return "Right"
    elif 202.5 <= angle < 247.5:
        return "Right Front"
    elif 247.5 <= angle < 292.5:
        return "Front"
    elif 292.5 <= angle < 337.5:
        return "Left Front"

if __name__ == "__main__":
    data = {}
    data['depth'] = torch.rand((8, 360, 640)).cuda()
    data['image'] = torch.rand((8, 3, 360, 640))
    data['fps_points_scene_from_original'] = torch.rand((8, 2048, 3)).cuda()
    data['fps_colors_scene_from_original'] = torch.rand((8, 2048, 3)).cuda()
    data['direction'] = ['Left', 'Left Behind', 'Behind', 'Right Behind', 'Right', 'Right Front', 'Front', 'Left Front']
    data['anchor_point'] = torch.rand((8, 3)).cuda()
    mask_prediction = torch.rand((8, 360, 640)).cuda()

    receptacle_metric(mask_prediction, data)
