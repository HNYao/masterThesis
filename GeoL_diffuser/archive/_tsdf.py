# Copyright (c) 2018 Andy Zeng

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch
import time
import numpy as np
import open3d as o3d
from skimage import measure
from GeoL_diffuser.models.helpers import TSDFVolume, get_view_frustum
import time
from GeoL_net.dataset_gen.RGBD2PC import backproject, visualize_points
if __name__ == "__main__":
    import cv2
    import torch

    # Load RGB-D image
    depth = cv2.imread("dataset/scene_RGBD_mask_v2/id494_1/bottle_0001_plastic/with_obj/test_pbr/000000/depth/000000.png", cv2.IMREAD_UNCHANGED)
    depth = depth.astype(np.float32) /1000.0
    color = cv2.imread(
        "dataset/scene_RGBD_mask_v2/id494_1/bottle_0001_plastic/with_obj/test_pbr/000000/rgb/000000.jpg", cv2.IMREAD_COLOR
    )
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


    cam_intr = np.array([[591.0125 ,   0.     , 322.525  ],
                [  0.     , 590.16775, 244.11084],
                [  0.     ,   0.     ,   1.     ]])

    cam_pose = np.eye(4)

    view_frust_pts = get_view_frustum(depth, cam_intr, cam_pose)  # [3,]
    vol_bnds = np.zeros((3, 2))
    vol_bnds[:, 0] = np.minimum(
        vol_bnds[:, 0], np.amin(view_frust_pts, axis=1)
    )  # [3,], [3, ]
    vol_bnds[:, 1] = np.maximum(
        vol_bnds[:, 1], np.amax(view_frust_pts, axis=1)
    )  # [3,], [3,]
    

    
    points = backproject(depth, cam_intr, np.ones_like(depth))
    points_scene, scene_ids = backproject(
        depth,
        cam_intr,
        depth < 2,
        # np.logical_and(hand_mask == 0, depth > 0),
        NOCS_convention=False,
    )
    colors_scene = color.copy()[scene_ids[0], scene_ids[1]] / 255

    # visualizer = o3d.visualization.Visualizer()
    # visualizer.create_window(height=1080, width=1920)
    # visualizer.add_geometry(pcd_scene)
    # load_viewpoint(visualizer, "configs/vis_config.json")
    # visualizer.run()
    # visualizer.destroy_window()
    # render = render_offscreen([pcd_scene], "configs/vis_config.json")
    
    vol_bnds[:, 0] = vol_bnds[:, 0].min()  # pad bounds
    vol_bnds[:, 1] = vol_bnds[:, 1].max()
    print("Volume bounds: ", vol_bnds.max(), vol_bnds.min())
    # print("bbox: ", np.array(bbox.get_max_bound()).max(), np.array(bbox.get_min_bound()).min())
    # Create TSDF volume and integrate observation
    import time
    query_points_scene = points_scene + np.random.randn(*points_scene.shape) * 0.0

    points_scene = (points_scene - vol_bnds.T[0:1]) / (
            vol_bnds.T[1:2] - vol_bnds.T[0:1]
        )
    points_scene = points_scene * 2 - 1

    query_points_scene = (query_points_scene - vol_bnds.T[0:1]) / (
            vol_bnds.T[1:2] - vol_bnds.T[0:1]
        )
    query_points_scene = query_points_scene * 2 - 1

    pcd_scene = visualize_points(points_scene, colors_scene)
    bbox = pcd_scene.get_oriented_bounding_box()
    o3d.visualization.draw_geometries([pcd_scene])
    for _ in range(1):
        time_before = time.time()
        tsdf_vol = TSDFVolume(vol_bnds, voxel_dim=256, verbose=False, enable_color=True, use_gpu=False)
        tsdf_vol.integrate(color, depth, cam_intr, cam_pose)
        # print("Time to integrate: ", time.time() - time_before)
    
    tsdf = tsdf_vol._tsdf_vol
    tsdf = torch.tensor(tsdf, dtype=torch.float32).cuda()[None, None] # [B, C, D, H, W]
    query_points = torch.tensor(query_points_scene, dtype=torch.float32).cuda()[None, None, None, ...]  # [B, 1, 1, N, 3]
    query_points = query_points[..., [2, 1, 0]] #[B, 1, 1, N, 3]
    query_tsdf = nn.functional.grid_sample(tsdf, query_points, align_corners=True)  # [B, C, 1, 1, N]
    query_tsdf = query_tsdf[:, :, 0, 0, :].squeeze()  # [B, C, N] => [N]
    query_tsdf = query_tsdf.cpu().numpy()

    import pdb; pdb.set_trace()

    # Get mesh from TSDF volume
    mesh = tsdf_vol.get_mesh()
    # o3d.visualization.draw([mesh, bbox])
    o3d.visualization.draw([pcd_scene, mesh, bbox])

    

    # # Test the trlinear interpolation
    # vertices = np.array(mesh.vertices)  # [N, 3]
    # query_points = torch.tensor(vertices, dtype=torch.float32).cuda()[
    #     None, ...
    # ]  # [B, N, 3]
    # vol_bnds_pt = (
    #     torch.tensor(vol_bnds, dtype=torch.float32).cuda()[None, ...].permute(0, 2, 1)
    # )  # [B, 3, 2] => [B, 2, 3]
    # query_points_norm = (query_points - vol_bnds_pt[:, 0:1]) / (
    #     vol_bnds_pt[:, 1:2] - vol_bnds_pt[:, 0:1]
    # ) * 2 - 1  # [B, N, 3] - [B, 1, 3]

    # color_grid_pt = torch.tensor(color_grid, dtype=torch.float32).cuda()[
    #     None, ...
    # ]  # [B, C, H, W, D]
    # query_points_norm = query_points_norm[..., [2, 1, 0]]
    # sample_grid = query_points_norm[:, :, None, None]  # [B, N, 1, 1, 3]
    # query_colors_pt = torch.nn.functional.grid_sample(
    #     color_grid_pt, sample_grid, align_corners=True
    # )  # [B, C, N, 1, 1]
    # query_colors_pt = query_colors_pt.squeeze(-1).squeeze(-1)  # [B, C, N]
    # query_colors_np = query_colors_pt.permute(0, 2, 1)[0].cpu().numpy()  # [N, C]

    # mesh.paint_uniform_color([0.5, 0.5, 0.5])
    # o3d.visualization.draw([mesh])
    # mesh.vertex_colors = o3d.utility.Vector3dVector(query_colors_np / 255)
    # o3d.visualization.draw([mesh])

    # Test the bilinear interpolation
    # from models.layers_2d import Project3D, BackprojectDepth
    # project_3d = Project3D()
    # backproject = BackprojectDepth(256, 456)
    # depth_pt = torch.tensor(depth, dtype=torch.float32).cuda()[None, None, ...]
    # color_pt = torch.tensor(color, dtype=torch.float32).cuda()[None, ...] / 255
    # color_pt = color_pt.permute(0, 3, 1, 2)
    # batch_size, _, height, width = color_pt.shape

    # points_pt = backproject(depth_pt, cam_intr)
    # print("points_pt shape: ", points_pt.shape)
    # points_np = points_pt.squeeze().cpu().numpy().T
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points_np)
    # o3d.visualization.draw_geometries([pcd])

    # points_pt = points_pt.view(batch_size, 3, -1).permute(0, 2, 1)
    # query_grids = project_3d(points_pt, cam_intr)
    # query_grids[:, 0] = (query_grids[:, 0] / (width - 1)) * 2 - 1
    # query_grids[:, 1] = (query_grids[:, 1] / (height - 1)) * 2 - 1
    # query_grids = query_grids.view(batch_size, 2, height, width).permute(0, 2, 3, 1)
    # color_grid_pt = torch.nn.functional.grid_sample(color_pt, query_grids, align_corners=True)
    # color_grid_np = color_grid_pt.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # plt.imshow(color_grid_np)
    # plt.show()
