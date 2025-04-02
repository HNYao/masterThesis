import torch
import torch.nn as nn
import numpy as np

class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud"""

    def __init__(self, height, width):
        super(BackprojectDepth, self).__init__()

        # self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing="xy")
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(
            torch.from_numpy(self.id_coords), requires_grad=False
        )
        self.pix_coords = torch.unsqueeze(
            torch.stack([self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0
        )

    def forward(self, depth, K):
        if isinstance(K, np.ndarray):
            assert K.shape == (3, 3)
            K = torch.from_numpy(K).float().to(depth.device)[None]

        batch_size = depth.shape[0]
        ones = torch.ones(batch_size, 1, self.height * self.width).to(depth.device)
        inv_K = torch.inverse(K).to(depth.device)  # [B, 3, 3]

        pix_coords = self.pix_coords.clone().to(depth.device)
        pix_coords = pix_coords.repeat(batch_size, 1, 1)
        pix_coords = torch.cat([pix_coords, ones], 1)  # [B, 3, H*W]

        cam_points = torch.matmul(inv_K, pix_coords)  # [B, 3, 3] @ [B, 3, H*W]
        cam_points = (
            depth.view(batch_size, 1, -1) * cam_points
        )  # [B, 1, H*W] * [B, 3, H*W]
        return cam_points


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