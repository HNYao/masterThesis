import torch
import cv2
import torch.nn as nn
import numpy as np
import open3d as o3d
import torch.nn.functional as F
import random
from GeoL_net.dataset_gen.RGBD2PC import backproject, visualize_points
from PIL import Image
import torchvision.transforms as T

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
            #o3d.io.write_point_cloud("exps/test_pcd_align.ply", pcd)

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

class ProjectColorOntoImage(nn.Module):
    def __init__(self):
        super().__init__()
        self.project_3d = Project3D()

    def forward(self, image_grid, query_points, query_colors, intrinsics):
        """
        Parameters
        ----------
        image_grid : torch.Tensor
            Image grid with shape [1, C, H, W] where C is the number of channels (typically 3 for RGB)
        query_points : torch.Tensor
            3D points with shape [1, N, 3]
        query_colors : torch.Tensor
            Colors for each query point, shape [B, N, 3] (RGB values)
        intrinsics : torch.Tensor or np.ndarray
            Camera intrinsics, shape [3, 3]

        Returns
        -------
        torch.Tensor
            Image grid with query points' colors projected onto it, shape [B, C, H, W]
        """
        _, height, width = image_grid.shape
        batch_size = 1

        # Step 1: Project 3D query points to 2D pixel coordinates
        query_grids = self.project_3d(query_points, intrinsics)  # [B, 2, N]
    
        # # Step 2: Normalize pixel coordinates to the range [-1, 1] for grid_sample
        query_grids = query_grids.permute(0, 2, 1)[:, :, None]  # [B, N, 1, 2]

        # Step 4: Blend the query_colors into the image at the projected points using the interpolated positions
        image_grid_updated = image_grid.clone().unsqueeze(0)  # [B, C, H, W]
        query_grids = query_grids.squeeze(2)  # [B, N, 2]
        for b in range(batch_size):
            projected_pixels_x = (query_grids[b, :, 0] - 0.5).floor().long() # [N]
            projected_pixels_y = (query_grids[b, :, 1] - 0.5).floor().long() # [N]
            projected_pixels_x = projected_pixels_x.clamp(0, width - 1)
            projected_pixels_y = projected_pixels_y.clamp(0, height - 1)
            image_grid_updated[b, :, projected_pixels_y, projected_pixels_x] = query_colors.to(torch.float32).T


        return image_grid_updated

class ProjectColorOntoImage_v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.project_3d = Project3D()

    def forward(self, image_grid, query_points, query_colors, intrinsics):
        """
        Parameters
        ----------
        image_grid : torch.Tensor
            Image grid with shape [B, C, H, W] where C is the number of channels (typically 3 for RGB)
        query_points : torch.Tensor
            3D points with shape [B, N, 3]
        query_colors : torch.Tensor
            Colors for each query point, shape [B, N, 3] (RGB values)
        intrinsics : torch.Tensor or np.ndarray
            Camera intrinsics, shape [3, 3]

        Returns
        -------
        torch.Tensor
            Image grid with query points' colors projected onto it, shape [B, C, H, W]
        """
        batch_size, _, height, width = image_grid.shape

        # Step 1: Project 3D query points to 2D pixel coordinates
        query_grids = self.project_3d(query_points, intrinsics)  # [B, 2, N]
        query_grids = query_grids.permute(0, 2, 1)  # [B, N, 2]

        # Step 2: Create pixel grid for the image
        y_coords, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')  # [H, W]
        pixel_grid = torch.stack([x_coords, y_coords], dim=-1).float().to(query_grids.device)  # [H, W, 2]
        pixel_grid = pixel_grid.view(-1, 2)  # [H*W, 2]

        # Step 3: For each pixel, find the 5 nearest query points and average their colors
        image_grid_updated = image_grid.clone()
        for b in range(batch_size):
            query_grid = query_grids[b]  # [N, 2]
            pixel_grid_b = pixel_grid  # [H*W, 2]

            # Compute the L2 distance between each pixel and each query point
            dists = torch.cdist(pixel_grid_b, query_grid)  # [H*W, N]

            # Find the 5 nearest query points for each pixel
            top5_dists, top5_idx = torch.topk(dists, k=5, dim=1, largest=False)  # [H*W, 5]

            # Get the colors of the 5 nearest query points
            top5_colors = query_colors[b, top5_idx]  # [H*W, 5, 3]

            # Average the colors of the 5 nearest query points
            avg_colors = top5_colors.mean(dim=1)  # [H*W, 3]

            # Assign the average color to each pixel
            avg_colors = avg_colors.T.view(3, height, width)  # [3, H, W]

            # Update the image grid
            image_grid_updated[b] = avg_colors

        return image_grid_updated


class ProjectColorOntoImage_v3(nn.Module):
    def __init__(self):
        super().__init__()
        self.project_3d = Project3D()

    def forward(self, image_grid, query_points,  query_colors, intrinsics):
        """
        Parameters
        ----------
        image_grid : torch.Tensor
            Image grid with shape [B, C, H, W] where C is the number of channels (typically 3 for RGB)
        query_points : torch.Tensor
            3D points with shape [B, N, 3]
        query_colors : torch.Tensor
            Colors for each query point, shape [B, N, 3] (RGB values)
        intrinsics : torch.Tensor or np.ndarray
            Camera intrinsics, shape [3, 3]

        Returns
        -------
        torch.Tensor
            Image grid with query points' colors projected onto it, shape [B, C, H, W]
        """
        batch_size, _, height, width = image_grid.shape

        # Step 1: Project 3D query points to 2D pixel coordinates
        query_grids = self.project_3d(query_points, intrinsics)  # [B, 2, N]
        query_grids = query_grids.permute(0, 2, 1)  # [B, N, 2]

        # Step 2: Create pixel grid for the image
        y_coords, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')  # [H, W]
        pixel_grid = torch.stack([x_coords, y_coords], dim=-1).float().to(query_grids.device)  # [H, W, 2]
        pixel_grid = pixel_grid.view(-1, 2)  # [H*W, 2]

        image_grid_updated = image_grid.clone()

        return image_grid_updated

def test_FetureConcat():
    camera_instrinsics =np.array([[591.0125 ,   0.     , 322.525  ],
            [  0.     , 590.16775, 244.11084],
            [  0.     ,   0.     ,   1.     ]])

    cam_rotation_matrix = np.array([
        [1, 0, 0],
        [0,0.8,-0.6],
        [0,0.6,0.8]
    ])  


    image = cv2.imread("dataset/scene_RGBD_mask/id9_2/eye_glasses_0001_orangeblack/no_obj/test_pbr/000000/rgb/000000.jpg", cv2.IMREAD_COLOR)
    image = shift_image(image, 0, 50) # shift image
    #depth_image = cv2.imread("dataset/scene_RGBD_mask/id196_2/bottle_0002_glass/no_obj/test_pbr/000000/depth/000000.png", cv2.IMREAD_UNCHANGED)
    #depth = np.array(depth_image)
    #print(image.shape)

    #points, _ = backproject(depth_image, camera_instrinsics, np.logical_and(depth > 0, depth > 0),)
    pc = o3d.io.read_point_cloud("outputs/model_output/point_cloud/id9_2-eye_glasses_0001_orangeblack-25_heatmap.ply")
    # ATTENTION if the pc is from processed heatmap, then do not rotate it again
    #pc = o3d.io.read_point_cloud("dataset/scene_RGBD_mask/id9_2/eye_glasses_0001_orangeblack/mask_red.ply")
    points = np.array(pc.points)
    #points = (np.linalg.inv(cam_rotation_matrix) @ points.T).T

    #pc = visualize_points(points)
    image_np = image.astype(np.float32)
    image_tensor = torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0)  # [B, C, H, W]
    #print('image tensor:', image_tensor)
    #image_tensor = torch.cat((image_tensor, image_tensor), dim=1)
    #print("image tensor:", image_tensor.shape)

    pc_points = np.asarray(points)
    num_points = pc_points.shape[0]
    pc_idx = np.random.choice(num_points, size=2048, replace=False)
    pc_points = pc_points[pc_idx]
    #pc_points[:] -= 10000
    pc_tensor = torch.tensor(pc_points, dtype=torch.float32).unsqueeze(0)  # [B, N, 3]

    #print("Point cloud tensor shape:", pc_tensor.shape)  # (N, 3)

    featureconcat = FeatureConcat()
    feature = featureconcat(image_tensor, pc_tensor, camera_instrinsics)
    #print(feature.shape)
    #print(feature)
    featureconcat.feature_visualization(feature, pc_tensor)

def pcdheatmap2img(ply_path, img_path, is_mask=False):
    cam_rotation_matrix = np.array([
        [1, 0, 0],
        [0,0.8,-0.6],
        [0,0.6,0.8]
    ])
    intrinsics = np.array([[591.0125 ,   0.     , 322.525  ],
            [  0.     , 590.16775, 244.11084],
            [  0.     ,   0.     ,   1.     ]])
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    
    image_np = image.astype(np.float32)

    image_tensor = torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0)  # [B, C, H, W]

    pc = o3d.io.read_point_cloud("outputs/model_output/point_cloud/id9_2-eye_glasses_0001_orangeblack-195_heatmap.ply") # 
    
    pc_points = np.array(pc.points)

    if is_mask:
        pc_points = pc_points @ cam_rotation_matrix
    #pc_points = pc_points @ cam_rotation_matrix # if mask

    #pc_points = pc_points @ np.linalg.inv(cam_rotation_matrix)

    query_points = torch.tensor(pc_points, dtype=torch.float32).unsqueeze(0)  # [B, N, 3]

    pc_colors = np.array(pc.colors)
    query_colors =  torch.tensor(pc_colors, dtype=torch.float32).unsqueeze(0)  # [B, N, 3]
    sample_ids = np.random.choice(np.arange(query_colors.size(1)), 2048)

    query_colors = query_colors[:, sample_ids]
    query_points = query_points[:, sample_ids]

    projector = ProjectColorOntoImage_v2()
    output_image = projector(image_grid=image_tensor, 
                             query_points=query_points, 
                             query_colors = query_colors, 
                             intrinsics = intrinsics)
    #save_path = "exps/heatmap_backalign.png"

    for i, img in enumerate(output_image):
        if img.ndim == 3:
            
            color_image = T.ToPILImage()(image_tensor[i].cpu())
            pil_image = T.ToPILImage()(img.cpu())
            
            image_np = np.clip(pil_image, 0, 255)
            #color_image_np = np.array(color_image)
            color_image_np = np.floor(color_image)
            color_image_np = np.clip(color_image_np, 0, 255)
            color_image_np = np.uint8(color_image_np)
            # image_np = cv2.blur(image_np, (50, 50))
            
            image_np = cv2.addWeighted(image_np, 0.3, color_image_np, 0.7, 0.0)
            pil_image = Image.fromarray(np.uint8(image_np))
            #pil_image = Image.fromarray(np.float32(image_np))
            #pil_image.save(save_path)
            #print(f"img is saved {save_path}")
    return pil_image


def shift_image(img, shift_x, shift_y):
    h, w = img.shape[:2]
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(img, M, (w, h))



if __name__ == "__main__":
##### test_FetureConcat    
    test_FetureConcat()

##### test ProjectColorOntoImage_v2
'''
    cam_rotation_matrix = np.array([
        [1, 0, 0],
        [0,0.8,-0.6],
        [0,0.6,0.8]
    ])  
    intrinsics = np.array([[591.0125 ,   0.     , 322.525  ],
            [  0.     , 590.16775, 244.11084],
            [  0.     ,   0.     ,   1.     ]])
    image = cv2.imread("dataset/scene_RGBD_mask/id9_2/eye_glasses_0001_orangeblack/no_obj/test_pbr/000000/rgb/000000.jpg", cv2.IMREAD_COLOR)
    
    #image_np = image.astype(np.uint8)
    image_np = image.astype(np.float32)
    #image_np = np.floor(image_np)

    image_tensor = torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0)  # [B, C, H, W]

    #pc = o3d.io.read_point_cloud("outputs/model_output/point_cloud_overfit_0909/id151_1-cup_0003_green-89_heatmap.ply") # 直接读取model output的heatmap
    #pc = o3d.io.read_point_cloud("dataset/scene_RGBD_mask/id9_2/eye_glasses_0001_orangeblack/mask.ply") # 
    pc = o3d.io.read_point_cloud("outputs/model_output/point_cloud/id9_2-eye_glasses_0001_orangeblack-195_heatmap.ply") # 
    
    pc_points = np.array(pc.points)
    #pc_points = pc_points @ cam_rotation_matrix # if mask

    #pc_points = pc_points @ np.linalg.inv(cam_rotation_matrix)

    query_points = torch.tensor(pc_points, dtype=torch.float32).unsqueeze(0)  # [B, N, 3]

    pc_colors = np.array(pc.colors)
    query_colors =  torch.tensor(pc_colors, dtype=torch.float32).unsqueeze(0)  # [B, N, 3]
    sample_ids = np.random.choice(np.arange(query_colors.size(1)), 2048)

    query_colors = query_colors[:, sample_ids]
    query_points = query_points[:, sample_ids]

    projector = ProjectColorOntoImage_v2()
    output_image = projector(image_grid=image_tensor, 
                             query_points=query_points, 
                             query_colors = query_colors, 
                             intrinsics = intrinsics)
    save_path = "exps/heatmap_backalign.png"
    for i, img in enumerate(output_image):
        if img.ndim == 3:
            
            color_image = T.ToPILImage()(image_tensor[i].cpu())
            pil_image = T.ToPILImage()(img.cpu())
            
            image_np = np.clip(pil_image, 0, 255)
            #color_image_np = np.array(color_image)
            color_image_np = np.floor(color_image)
            color_image_np = np.clip(color_image_np, 0, 255)
            color_image_np = np.uint8(color_image_np)
            # image_np = cv2.blur(image_np, (50, 50))
            
            image_np = cv2.addWeighted(image_np, 0.3, color_image_np, 0.7, 0.0)
            pil_image = Image.fromarray(np.uint8(image_np))
            #pil_image = Image.fromarray(np.float32(image_np))
            pil_image.save(save_path)
            print(f"img is saved {save_path}")
        else:
            print("incorrect shape")
'''