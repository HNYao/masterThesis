import torch
import cv2
import torch.nn as nn
import numpy as np
import open3d as o3d
import torch.nn.functional as F
import random
from GeoL_net.dataset_gen.RGBD2PC import backproject, visualize_points

class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T"""

    def __init__(self, eps=1e-7):
        super(Project3D, self).__init__()
        self.eps = eps

    def forward(self, points, K, T=None):
        """_summary_

        Parameters
        ----------
        points : torch.Tensor
            [B, H*W, 3]
        K : torch.Tensor / np.ndarray
            [3, 3]
        T : torch.Tensor
            [B, 4, 4]

        Returns
        -------
        torch.Tensor
            [B, 2, H*W]
        """
        if isinstance(K, np.ndarray):
            K = torch.from_numpy(K).float().to(points.device)
            if len(K.shape) == 2:
                K = K[None]
        if T is None:
            T = torch.eye(4).float().to(points.device)[None]
            T = torch.tensor([[1, 0, 0,0], [0, 1, 0,0], [0,0,1,0], [0,0,0,100]]).float().to(points.device)[None]
        points = points.float()
        batch_size = points.shape[0]
        points_cam = points @ T[:, :3, :3].transpose(-1, -2) + T[:, :3, 3].unsqueeze(
            -2
        )  # [B, N, 3] @ [B, 3, 3] + [B, 1, 3]

        pix_coords = points_cam @ K.transpose(
            -1, -2
        )  # [B, N, 3] @ [B, 3, 3] => [B, N, 3]
        pix_coords = pix_coords[..., :2] / (
            pix_coords[..., 2:3] + self.eps
        )  # [B, N, 2]
        pix_coords = pix_coords.permute(0, 2, 1)  # [B, 2, N]
        return pix_coords

class FeatureConcat(nn.Module):
    def __init__(
        self
    ):
        super().__init__()
        self.project_3d = Project3D()


        #self.proj = nn.Linear(proj_dim_in, self.embedding_dim, bias=True)

    def interpolate_image_grid_features(self, image_grid, query_points, intrinsics):
        """
        Parameters
        ----------
        image_grid : torch.Tensor
            with shape [B, C, H, W]
        query_points : torch.Tensor
            _with shape [B, N, 3]
        """
        batch_size, _, height, width = image_grid.shape
        query_grids = self.project_3d(query_points, intrinsics)  # [B, 2, N]
        #print("grid cord:", query_grids)
        query_grids[:, 0] = (query_grids[:, 0] / (width - 1)) * 2 - 1
        query_grids[:, 1] = (query_grids[:, 1] / (height - 1)) * 2 - 1
        query_grids = query_grids.permute(0, 2, 1)[:, :, None]  # [B, N, 1, 2]
        query_featurs = F.grid_sample(
            image_grid, query_grids, mode="bilinear", align_corners=True
        )  # [B, C, N, 1]
        query_featurs = query_featurs.squeeze(-1)
        return query_featurs

    def feature_visualization(self, features, points):
        """_summary_

        Parameters
        ----------
        features : torch.Tensor
            with shape [B, N, F]
        points : query points
            [B, N, 3]
        """
        assert len(features) == len(points)
        for i in range(len(features)):
            feature = features[i]
            feature_np = feature.numpy()/255
            point_position = points[i]
            point_position_np = point_position.numpy()
                    
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_position_np)
            pcd.colors = o3d.utility.Vector3dVector(feature_np)
            o3d.visualization.draw_geometries([pcd])

    def forward(
        self,
        color_feature,
        query_points,
        intrinsics,
        **kwargs,
    ):
        """_summary_

        Parameters
        ----------
        color_feature : torch.Tensor
            with shape [B, C, H, W]
        query_points : query points
            [B, N, 3]
        intrinsics : torch.Tensor or np.ndarray
            [3, 3]

        Returns
        -------
        torch.Tensor
            shape of [B, N, C*4]
        """
        features = []
        feat_2d = self.interpolate_image_grid_features(
            color_feature, query_points, intrinsics
        )
        features.append(feat_2d)  # [B, C, N]
        features = torch.cat(features, dim=1).permute(0, 2, 1)  # [B, N, C*3]
        #features = self.proj(features) # [B, N, C]
        return features

def test_FetureConcat():
    camera_instrinsics =np.array([[591.0125 ,   0.     , 322.525  ],
            [  0.     , 590.16775, 244.11084],
            [  0.     ,   0.     ,   1.     ]])

    cam_rotation_matrix = np.array([
        [1, 0, 0],
        [0,0.8,-0.6],
        [0,0.6,0.8]
    ])  


    image = cv2.imread("dataset/scene_RGBD_mask/id162_1/lamp_0004_orange/no_obj/test_pbr/000000/rgb/000000.jpg", cv2.IMREAD_COLOR)
    depth_image = cv2.imread("dataset/scene_RGBD_mask/id162_1/lamp_0004_orange/no_obj/test_pbr/000000/depth/000000.png", cv2.IMREAD_UNCHANGED)
    depth = np.array(depth_image)
    #print(image.shape)

    #points, _ = backproject(depth_image, camera_instrinsics, np.logical_and(depth > 0, depth > 0),)
    pc = o3d.io.read_point_cloud("dataset/scene_RGBD_mask/id162_1/lamp_0004_orange/mask.ply")
    points = np.array(pc.points)
    points = (np.linalg.inv(cam_rotation_matrix) @ points.T).T

    pc = visualize_points(points)
    image_np = image.astype(np.float32)
    image_tensor = torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0)  # [B, C, H, W]
    #print('image tensor:', image_tensor)
    #image_tensor = torch.cat((image_tensor, image_tensor), dim=1)
    #print("image tensor:", image_tensor.shape)

    pc_points = np.asarray(pc.points)
    num_points = pc_points.shape[0]
    pc_idx = np.random.choice(num_points, size=4096, replace=False)
    pc_points = pc_points[pc_idx]
    #pc_points[:] -= 10000
    pc_tensor = torch.tensor(pc_points, dtype=torch.float32).unsqueeze(0)  # [B, N, 3]

    #print("Point cloud tensor shape:", pc_tensor.shape)  # (N, 3)

    featureconcat = FeatureConcat()
    feature = featureconcat(image_tensor, pc_tensor, camera_instrinsics)
    #print(feature.shape)
    #print(feature)
    featureconcat.feature_visualization(feature, pc_tensor)

if __name__ == "__main__":
    test_FetureConcat()