import numpy as np
import open3d as o3d
import torch


def position_metric(position_gt, position_pred):
    """
    calculate the distance between position_pred and position_gt
    """
    pass

def receptacle_metric(position_gt, position_pred, receptacle):
    """
    check if the prediction is on the receptacle
    """
    pass


def collision_metric(y_true, y_pred, object_pc, scene_pc):
    """
    calculate the collision metric between y_true and y_pred
    """
    pass


