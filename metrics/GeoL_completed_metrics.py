import torch
import numpy as np
import random
import os

from PIL import Image
from torch.utils.data import DataLoader, Subset, random_split
from omegaconf import DictConfig, OmegaConf
import cv2

from thirdpart.seeingunseen.seeing_unseen.core.base import BaseTrainer
from thirdpart.seeingunseen.seeing_unseen.core.registry import registry
import torch.multiprocessing as mp
from GeoL_net.models.GeoL import GeoL_net_v9
from GeoL_diffuser.algos.pose_algos import PoseDiffusionModel

from GeoL_diffuser.models.guidance import *
from metrics.dataset_factory import BlendprocDesktopDataset, BlendprocDesktopDataset_incompleted, BlendprocDesktopDataset_incompleted_sparse, BlendprocDesktopDataset_incompleted_mult_cond    
from Geo_comb.pred_one_case_completed import pred_one_case_dataset, generate_heatmap_pc
from metrics.utils import *
from Geo_comb.pred_one_case_completed import rgb_obj_dect, rgb_obj_dect_no_vlm
import yaml
from sklearn.cluster import KMeans

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

def seed_everything(seed):
    random.seed(seed) #为python设置随机种子
    np.random.seed(seed)  #为numpy设置随机种子
    torch.manual_seed(seed)   #为CPU设置随机种子
    torch.cuda.manual_seed(seed)   #为当前GPU设置随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed_all(seed)   #为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.use_deterministic_algorithms(True, warn_only=True)





def GeoL_completed_metrics(
        dataset=BlendprocDesktopDataset(),
        sample_datasize=10,
        pretrained_affordance=None,
        pretrained_diffusion=None,
        process_metric_func=None, 
        use_kmeans=False,
):
    # initialize the dataset
    num_sample = sample_datasize
    indices = torch.arange(num_sample)
    subset = Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)
    
    # initialize the model
    INTRINSICS = np.array([[607.09912/2 , 0. , 636.85083/2 ], [0., 607.05212/2, 367.35952/2], [0.0, 0.0, 1.0]])
    model_affordance = GeoL_net_v9(
        input_shape=(3, 480, 640),
        target_input_shape=(3, 128, 128),
        intrinsics=INTRINSICS
    )

    with open("config/baseline/diffusion.yaml", "r") as file:
        yaml_data = yaml.safe_load(file)
    config_diffusion = OmegaConf.create(yaml_data)
    model_diffusion = PoseDiffusionModel(config_diffusion.model).to("cuda")

    state_affordance_dict = torch.load(pretrained_affordance, map_location="cpu")
    state_diffusion_dict = torch.load(pretrained_diffusion, map_location="cpu")

    model_affordance.load_state_dict(state_affordance_dict["ckpt_dict"])
    model_diffusion.load_state_dict(state_diffusion_dict["ckpt_dict"])
    #guidance = AffordanceGuidance_v2()
    #guidance = NonCollisionGuidance_v2()
    guidance = CompositeGuidance()
    model_diffusion.nets['policy'].set_guidance(guidance)

    model_affordance.eval()
    model_affordance.cuda()
    model_diffusion.eval()
    model_diffusion.cuda()


    is_success_result = []
    is_direction_result = []
    non_collision_result = []
    test_sum = 0
    for data_batch in dataloader:

        rgb_img_path = data_batch["image_without_obj_path"][0]
        rgb_img_with_obj_path = data_batch["image_with_obj_path"][0]
        rgb_image = Image.open(rgb_img_path).convert("RGB")
        rgb_image = np.array(rgb_image).astype(float)
        rgb_image = np.transpose(rgb_image, (2, 0, 1))  # Change to (C, H, W) format for PyTorch
        object_to_place = data_batch["object_name"][0]
        direction = data_batch["direction"][0]
        anchor_obj_name = data_batch["anchor_obj_name"][0]
        depth_file_path = data_batch["depth_without_obj_path"][0]
        mask_with_obj_path = data_batch["mask_with_obj_path"][0]
        mask_without_obj_path = data_batch["mask_without_obj_path"][0]
        hdf5_path = data_batch["hdf5_path"][0]
        depth_with_obj_path = data_batch["depth_with_obj_path"][0]
        depth_without_obj_path = data_batch["depth_without_obj_path"][0]
        vol_bnds = data_batch["vol_bnds"][0]
        tsdf_vol = data_batch["tsdf_vol"][0]
        T_plane = data_batch["T_plane"][0]


        print(f"place {object_to_place} to the {direction} of {anchor_obj_name}")
        print(mask_with_obj_path)

        # find anchor object
        annotated_frame = rgb_obj_dect_no_vlm(
            image_path=rgb_img_path,
            text_prompt=anchor_obj_name,
            out_dir="outputs"
        )

        color_no_obj = np.array(annotated_frame)
        depth = cv2.imread(depth_file_path, cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32) / 1000.0

        points_no_obj_scene, scene_no_obj_idx = backproject(
            depth,
            INTRINSICS,
            np.logical_and(depth > 0, depth < 2)
        )
        colors_no_obj_scene = color_no_obj[scene_no_obj_idx[0], scene_no_obj_idx[1]]
        pcd_no_obj_scene = visualize_points(points_no_obj_scene, colors_no_obj_scene)

        dataset_one_case = pred_one_case_dataset(
            pcd_no_obj_scene,
            rgb_image_file_path=rgb_img_with_obj_path,
            target_name=anchor_obj_name,
            direction_text=direction,
            depth_img_path=depth_with_obj_path, # use depth_with_obj_path to make crowded scene
        )
        dataloader_one_case = DataLoader(dataset_one_case, batch_size=1, shuffle=False)

        for i, batch in enumerate(dataloader_one_case):
            for key, val in batch.items():
                if type(val) == list:
                    continue
                batch[key] = val.float().to("cuda")

            with torch.no_grad():
                affordance_pred = model_affordance(batch=batch)["affordance"]
                #generate_heatmap_pc(batch, affordance_pred, intrinsics=INTRINSICS, interpolate=False)
                affordance_pred_sigmoid = affordance_pred.sigmoid().cpu().numpy()
                affordance_thershold = -100
                fps_points_scene_from_original = batch["fps_points_scene"][0]
                #fps_points_scene_from_original = batch["fps_points_scene"][0]

                fps_points_scene_affordance = fps_points_scene_from_original[
                    affordance_pred_sigmoid[0][:, 0] > affordance_thershold
                ]
                fps_points_scene_affordance = fps_points_scene_affordance.cpu().numpy()
                min_bound_affordance = np.append(
                    np.min(fps_points_scene_affordance, axis=0), -180
                )
                max_bound_affordance = np.append(
                    np.max(fps_points_scene_affordance, axis=0), 180
                )
                # sample 512 points from fps_points_scene_affordance
                fps_points_scene_affordance = fps_points_scene_affordance[
                    np.random.choice(
                        fps_points_scene_affordance.shape[0], 512, replace=True
                    )
                ]  # [512, 3]

                batch['affordance_fine'] = affordance_pred.clone()
                batch['affordance'] = affordance_pred.clone()

                # use kmeans
                if use_kmeans:
                    pred_affordance_np = affordance_pred.cpu().numpy()[0, : ,0]
                    fps_points_scene = data_batch["fps_points_scene"].cpu().numpy()[0]
                    pred_affordance_kmeans = apply_kmeans_to_affordance(
                        fps_points_scene, 
                        pred_affordance_np,
                        n_clusters=1,  # Adjust based on how many distinct regions you want
                        percentile_threshold=95,  # Adjust based on how strict you want the filtering
                        dist_factor=0.5
                    )
                    pred_affordance_kmeans = torch.from_numpy(pred_affordance_kmeans).float().cuda()[None, :, None]
                    pred_affordance_kmeans = (pred_affordance_kmeans - pred_affordance_kmeans.min()) / \
                        (pred_affordance_kmeans.max() - pred_affordance_kmeans.min()) 
                    pred_affordance_fine = (affordance_pred - affordance_pred.min() ) / \
                        (affordance_pred.max() - affordance_pred.min())
                #batch['affordance_fine'] = affordance_pred.clone()
                    batch['affordance'] = pred_affordance_kmeans.to('cuda')
                to_save = {
                    "pc_position": fps_points_scene_from_original.cpu().numpy(),
                    "affordance": affordance_pred_sigmoid,
                    "pc_position_xy_affordance": fps_points_scene_affordance,
                    "min_bound_affordance": min_bound_affordance,
                    "max_bound_affordance": max_bound_affordance,
                    "affordance_pred": affordance_pred_sigmoid,
                }
                #affordance_pred = affordance_pred_sigmoid
                fps_points_scene_affordance = fps_points_scene_from_original.cpu().numpy()
                
                # update dataset

                batch["object_name"] = [object_to_place]  
                obj_mesh = trimesh.load(data_batch["obj_mesh_path"][0])
                
                #obj_scale = [0.4554532861750685, 0.4554532861750685, 0.4554532861750685]
                #obj_mesh.apply_scale(obj_scale)
                #obj_pc = obj_mesh.sample(512)  
                obj_pc = data_batch["obj_points"][0].to("cuda")     

                batch["object_pc_position"] = (
                    torch.tensor(obj_pc, dtype=torch.float32).unsqueeze(0).to("cuda")
                )
                batch["pc_position_xy_affordance"] = (
                    torch.tensor(fps_points_scene_affordance[:, :2], dtype=torch.float32)
                    .unsqueeze(0)
                    .to("cuda")
                )
                batch['gt_pose_xyz_min_bound'] = (torch.tensor(min_bound_affordance, dtype=torch.float32).unsqueeze(0).to("cuda"))
                batch['gt_pose_xyz_max_bound'] = (torch.tensor(max_bound_affordance, dtype=torch.float32).unsqueeze(0).to("cuda"))
                batch['gt_pose_xyR_min_bound'] = torch.tensor(np.delete(min_bound_affordance, obj=2, axis=0), dtype=torch.float32).unsqueeze(0).to("cuda")
                batch['gt_pose_xyR_max_bound'] = torch.tensor(np.delete(max_bound_affordance, obj=2, axis=0), dtype=torch.float32).unsqueeze(0).to("cuda")
                depth_file_path = data_batch["depth_without_obj_path"][0]
                depth = cv2.imread(depth_file_path, cv2.IMREAD_UNCHANGED)
                depth = depth.astype(np.float32) / 1000.0
                depth[depth > 2] = 0
                scene_depth_cloud, _ = backproject(depth, INTRINSICS, np.logical_and(depth > 0, depth < 2))
                scene_depth_cloud = visualize_points(scene_depth_cloud)
                scene_depth_cloud_points = np.asarray(scene_depth_cloud.points)
                max_xyz_bound = np.max(scene_depth_cloud_points, axis=0)
                min_xyz_bound = np.min(scene_depth_cloud_points, axis=0)
                min_xyz_bound[2] = 1.0
                max_xyz_bound[2] = 1.0
                

                batch["gt_pose_xy_min_bound"] = (
                    torch.tensor(
                        min_xyz_bound[...,:2], dtype=torch.float32
                    )
                    .unsqueeze(0)
                    .to("cuda")
                )
                batch["gt_pose_xy_max_bound"] = (
                    torch.tensor(
                        max_xyz_bound[...,:2], dtype=torch.float32
                    )
                    .unsqueeze(0)
                    .to("cuda")
                )
                batch["gt_pose_4d_min_bound"] = (
                    torch.tensor(min_bound_affordance, dtype=torch.float32)
                    .unsqueeze(0)
                    .to("cuda")
                )
                batch["gt_pose_4d_max_bound"] = (
                    torch.tensor(max_bound_affordance, dtype=torch.float32)
                    .unsqueeze(0)
                    .to("cuda")
                )
                batch['affordance_non_cond'] = torch.randn_like(batch['affordance']).to("cuda")
                batch['gt_pose_xyz_for_non_cond'] = torch.randn((1,80,3)).to("cuda")
                batch['vol_bnds'] = vol_bnds.unsqueeze(0).float().to("cuda")
                batch['tsdf_vol'] = tsdf_vol.unsqueeze(0).to("cuda")
                batch['T_plane'] = T_plane.unsqueeze(0).float().to("cuda")
                batch['color_tsdf'] = data_batch["color_tsdf"].to("cuda")
                batch['intrinsics'] = data_batch["intrinsics"].to("cuda")


                #generate_heatmap_pc(batch, affordance_pred, intrinsics=INTRINSICS, interpolate=False)

                pred = model_diffusion(batch, num_samp=1, class_free_guide_w=0, apply_guidance=True, guide_clean=True)  
                # for key, val in pred['guide_losses'].items():
                #     print(f"{key} mean: {val.mean()}")
                #     print(f"{key} max: {val.max()}")
                #     print(f"{key} min: {val.min()}")
                

                #visualize_xy_pred_points(pred, batch, intrinsics=INTRINSICS)

                #pred['pose_xy_pred'] = pred['pose_xy_pred'] 
                #pred['pose_xy_pred'] = pred['pose_xy_pred'][:, min_guide_loss_idx, :][None]
                #pred['guide_losses']['loss'] = pred['guide_losses']['loss'][:, min_guide_loss_idx][None]

                # transfer 2D pred_points to 3D
                depth = batch["depth"][0].cpu().numpy()
                points_scene, idx = backproject(
                    depth,
                    INTRINSICS,
                    np.logical_and(depth > 0, depth < 2),
                    NOCS_convention=False,
                )
                T_plane, plane_model = get_tf_for_scene_rotation(points_scene)
                points_for_place = pred["pose_xyR_pred"][..., :2] # option2: use the plane_model to get the points 
                points_for_place = points_for_place[0].cpu().numpy()
                points_for_place_z = (-plane_model[3] - plane_model[0] * points_for_place[..., 0] - plane_model[1] * points_for_place[..., 1]) / plane_model[2]
                points_for_place = np.concatenate([points_for_place, points_for_place_z[:, None]], axis=1)
                pred_points = points_for_place 


                # only get the topk point with the lowest cost
                topk = 5
                # guide_affordance_loss = pred["guide_losses"]["loss"].cpu().numpy()
                # distance_error = pred["guide_losses"]["distance_error"].cpu().numpy()
                # min_guide_loss_idx = np.argsort(guide_affordance_loss)[0][:topk]
                # pred_points = pred_points[min_guide_loss_idx]

                guide_affordance_loss = pred["guide_losses"]["affordance_loss"].cpu().numpy()
                guide_collision_loss = pred["guide_losses"]["collision_loss"].cpu().numpy()
                guide_distance_error = pred["guide_losses"]["distance_error"].cpu().numpy()
                min_colliion_loss = guide_collision_loss.min()
                guide_composite_loss = guide_affordance_loss
                guide_composite_loss[guide_collision_loss > min_colliion_loss] = np.inf
                min_guide_loss_idx = np.argsort(guide_composite_loss)[0][:topk]
                pred_points = pred_points[min_guide_loss_idx]


                print("min distance loss:", guide_distance_error[0][min_guide_loss_idx])
                print("min collision loss:", guide_collision_loss[0][min_guide_loss_idx])


                
                
                #visualize_xy_pred_points(pred, batch, intrinsics=INTRINSICS)
                is_success, is_direction, non_collision = process_metric_func(
                    affordance_pred,
                    pred_points,
                    mask_with_obj_path, 
                    mask_without_obj_path, 
                    depth_with_obj_path, 
                    depth_without_obj_path, 
                    hdf5_path, 
                    direction,
                    obj_mesh_path = data_batch["obj_mesh_path"][0],
                    scene_mesh_path = data_batch["scene_mesh_path"][0],
                    pred_points = pred_points,
                )

                is_success_result += is_success
                is_direction_result += is_direction
                non_collision_result += non_collision
                test_sum += 1

                print("test_sum:", test_sum) 
                print("success:", is_success)
                print("direction:", is_direction)

                print("non_collision:", non_collision)
                print("current success rate:", sum(is_success_result) / len(is_success_result)*100, "%")
                print("current direction rate:", sum(is_direction_result) / len(is_direction_result)*100, "%")
                print("current non_collision rate:", sum(non_collision_result) / len(non_collision_result)*100, "%")
             

            break
    print("Final success rate:", sum(is_success_result) / len(is_success_result)*100, "%")
    print("Final direction rate:", sum(is_direction_result) / len(is_direction_result)*100, "%")
    print("Final non_collision rate:", sum(non_collision_result) / len(non_collision_result)*100, "%")

        # cv2.imshow("top_k", updated_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        

        # print(top_k_idx)
        # print(top_k_value)

        # map_2d = model_pred.squeeze(0).cpu().detach().numpy()
        # map_2d = (255 * (map_2d - map_2d.min())/ (map_2d.max() - map_2d.min())).astype(np.uint8)
        # color_map = cv2.applyColorMap(map_2d, cv2.COLORMAP_JET)
        # cv2.imshow("color_map", color_map)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
def GeoL_completed_metrics_mult_cond(
        dataset=BlendprocDesktopDataset_incompleted_mult_cond,
        sample_datasize=10,
        pretrained_affordance=None,
        pretrained_diffusion=None,
        process_metric_func=None, 
):
    #1 initialize the dataset
    num_sample = sample_datasize
    indices = torch.arange(num_sample)
    subset = Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=1, shuffle=True, num_workers=0)
    
    #2 initialize the model
    INTRINSICS = np.array([[607.09912/2 , 0. , 636.85083/2 ], [0., 607.05212/2, 367.35952/2], [0.0, 0.0, 1.0]])
    model_affordance = GeoL_net_v9(
        input_shape=(3, 480, 640),
        target_input_shape=(3, 128, 128),
        intrinsics=INTRINSICS
    )
    with open("config/baseline/diffusion.yaml", "r") as file:
        yaml_data = yaml.safe_load(file)
    config_diffusion = OmegaConf.create(yaml_data)
    model_diffusion = PoseDiffusionModel(config_diffusion.model).to("cuda")
    state_affordance_dict = torch.load(pretrained_affordance, map_location="cpu")
    state_diffusion_dict = torch.load(pretrained_diffusion, map_location="cpu")
    model_affordance.load_state_dict(state_affordance_dict["ckpt_dict"])
    model_diffusion.load_state_dict(state_diffusion_dict["ckpt_dict"])
    guidance = CompositeGuidance()
    model_diffusion.nets['policy'].set_guidance(guidance)
    model_affordance.eval()
    model_affordance.cuda()
    model_diffusion.eval()
    model_diffusion.cuda()

    #3 initialize the metrics
    is_success_result = []
    is_direction_result = []
    non_collision_result = []

    # loop over the dataset
    test_sum = 0
    for data_batch in dataloader:
        
        #4 get the data
        rgb_img_path = data_batch["image_without_obj_path"][0]
        rgb_img_with_obj_path = data_batch["image_with_obj_path"][0]
        rgb_image = Image.open(rgb_img_path).convert("RGB")
        rgb_image = np.array(rgb_image).astype(float)
        rgb_image = np.transpose(rgb_image, (2, 0, 1))  # Change to (C, H, W) format for PyTorch
        object_to_place = data_batch["object_name"][0]
        direction_list = data_batch["direction"]
        anchor_obj_name_list = data_batch["anchor_obj_name"]
        anchor_obj_position_list = data_batch["anchor_obj_positions"][0] # only for rw metric
        depth_file_path = data_batch["depth_without_obj_path"][0]
        mask_with_obj_path = data_batch["mask_with_obj_path"][0]
        mask_without_obj_path = data_batch["mask_without_obj_path"][0]
        hdf5_path = data_batch["hdf5_path"][0]
        depth_with_obj_path = data_batch["depth_with_obj_path"][0]
        depth_without_obj_path = data_batch["depth_without_obj_path"][0]
        vol_bnds = data_batch["vol_bnds"][0]
        tsdf_vol = data_batch["tsdf_vol"][0]
        T_plane = data_batch["T_plane"][0]
        
        print(mask_with_obj_path)

        affordance_pred_list = []
        for i, (direction, anchor_obj_name) in enumerate(zip(direction_list, anchor_obj_name_list)):
            direction = direction[0]
            anchor_obj_name = anchor_obj_name[0]
            print(f"place {object_to_place} to the {direction} of {anchor_obj_name}")
            #5  find anchor object: use grounding dino or use the anchor position directly
            annotated_frame = rgb_obj_dect_no_vlm(
                image_path=rgb_img_path,
                text_prompt=anchor_obj_name,
                out_dir="outputs"
            )
            color_no_obj = np.array(annotated_frame)
            depth = cv2.imread(depth_file_path, cv2.IMREAD_UNCHANGED)
            depth = depth.astype(np.float32) / 1000.0

            points_no_obj_scene, scene_no_obj_idx = backproject(
                depth,
                INTRINSICS,
                np.logical_and(depth > 0, depth < 2)
            )
            colors_no_obj_scene = color_no_obj[scene_no_obj_idx[0], scene_no_obj_idx[1]]
            pcd_no_obj_scene = visualize_points(points_no_obj_scene, colors_no_obj_scene)

            dataset_one_case = pred_one_case_dataset(
                pcd_no_obj_scene,
                rgb_image_file_path=rgb_img_path,
                target_name=anchor_obj_name,
                direction_text=direction,
                depth_img_path=depth_without_obj_path, # use depth_with_obj_path to make crowded scene
            )
            dataloader_one_case = DataLoader(dataset_one_case, batch_size=1, shuffle=False)

            for i, batch in enumerate(dataloader_one_case):

                for k in range(anchor_obj_position_list.shape[0]):
                    #batch["anchor_position"] = anchor_obj_position_list[k].unsqueeze(0) # use the anchor position directly instead of grounding dino

                    for key, val in batch.items():
                        if type(val) == list:
                            continue
                        batch[key] = val.float().to("cuda")

                    with torch.no_grad():
                        affordance_pred = model_affordance(batch=batch)["affordance"]
                        #generate_heatmap_pc(batch, affordance_pred, intrinsics=INTRINSICS, interpolate=False)
                        affordance_pred_list.append(affordance_pred)
                
                break
                
            
        #7 merge the affordance prediction
        pred_affordance_merge = torch.cat(affordance_pred_list, dim=0)
        pred_affordance_merge = (pred_affordance_merge ).mean(dim=0, keepdim=True)
        # normalize the affordance prediction
        pred_affordance_merge = (pred_affordance_merge - pred_affordance_merge.min()) / (pred_affordance_merge.max() - pred_affordance_merge.min()) 
        #generate_heatmap_pc(batch, pred_affordance_merge, INTRINSICS, interpolate=False) # visualize single case

        #8 prepare the data for diffusion model
        affordance_pred_sigmoid = pred_affordance_merge.cpu().numpy()
        affordance_thershold = -100
        fps_points_scene_from_original = batch["fps_points_scene"][0]
        #fps_points_scene_from_original = batch["fps_points_scene"][0]

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

        batch['affordance'] = pred_affordance_merge.to('cuda')
        batch['gt_pose_xy_min_bound'] = torch.tensor(min_bound_affordance[...,:2], dtype=torch.float32).unsqueeze(0).to("cuda")
        batch['gt_pose_xy_max_bound'] = torch.tensor(max_bound_affordance[...,:2], dtype=torch.float32).unsqueeze(0).to("cuda")
        batch['gt_pose_xyR_min_bound'] = torch.tensor(np.delete(min_bound_affordance, obj=2, axis=0), dtype=torch.float32).unsqueeze(0).to("cuda")
        batch['gt_pose_xyR_max_bound'] = torch.tensor(np.delete(max_bound_affordance, obj=2, axis=0), dtype=torch.float32).unsqueeze(0).to("cuda")
        batch['object_pc_position'] = torch.tensor(data_batch["obj_points"], dtype=torch.float32).to("cuda")
        batch['tsdf_vol'] = torch.tensor(data_batch["tsdf_vol"], dtype=torch.float32).to("cuda")
        batch['T_plane'] = torch.tensor(data_batch["T_plane"], dtype=torch.float32).to("cuda")
        batch['color_tsdf'] = torch.tensor(data_batch["color_tsdf"], dtype=torch.float32).to("cuda")
        batch['intrinsics'] = torch.tensor(data_batch["intrinsics"], dtype=torch.float32).to("cuda")
        batch['vol_bnds'] = torch.tensor(data_batch["vol_bnds"], dtype=torch.float32).to("cuda")
        #9 predict the xyR
        pred = model_diffusion(batch, num_samp=1, class_free_guide_w=-0.2, apply_guidance=True, guide_clean=True)
        #visualize_xy_pred_points(pred, batch, intrinsics=INTRINSICS) # visualize the singel case
        
        #10 transfer 2D pred_points to 3D
        depth = batch["depth"][0].cpu().numpy()
        points_scene, idx = backproject(
            depth,
            INTRINSICS,
            np.logical_and(depth > 0, depth < 2),
            NOCS_convention=False,
        )
        T_plane, plane_model = get_tf_for_scene_rotation(points_scene)
        points_for_place = pred["pose_xyR_pred"][..., :2] # option2: use the plane_model to get the points 
        points_for_place = points_for_place[0].cpu().numpy()
        points_for_place_z = (-plane_model[3] - plane_model[0] * points_for_place[..., 0] - plane_model[1] * points_for_place[..., 1]) / plane_model[2]
        points_for_place = np.concatenate([points_for_place, points_for_place_z[:, None]], axis=1)
        pred_points = points_for_place

        #11 get the topk point with the lowest cost
        topk = 5
        guide_affordance_loss = pred["guide_losses"]["affordance_loss"].cpu().numpy()
        guide_collision_loss = pred["guide_losses"]["collision_loss"].cpu().numpy()
        guide_distance_error = pred["guide_losses"]["distance_error"].cpu().numpy()
        min_colliion_loss = guide_collision_loss.min()
        guide_composite_loss = guide_affordance_loss
        guide_composite_loss[guide_collision_loss > min_colliion_loss] = np.inf
        min_guide_loss_idx = np.argsort(guide_composite_loss)[0][:topk]
        pred_points = pred_points[min_guide_loss_idx]
        print("min distance loss:", guide_distance_error[0][min_guide_loss_idx])
        print("min collision loss:", guide_collision_loss[0][min_guide_loss_idx])

        #12 process the merics
        is_success, is_direction, non_collision = process_metric_func(
                affordance_pred,
                pred_points,
                mask_with_obj_path, 
                mask_without_obj_path, 
                depth_with_obj_path, 
                depth_without_obj_path, 
                hdf5_path, 
                direction,
                obj_mesh_path = data_batch["obj_mesh_path"][0],
                scene_mesh_path = data_batch["scene_mesh_path"][0],
                pred_points = pred_points,
                anchor_position_list = anchor_obj_position_list,
                direction_list = direction_list,
            )          

        is_success_result += is_success
        is_direction_result += is_direction
        non_collision_result += non_collision
        test_sum += 1

        print("test_sum:", test_sum) 
        print("success:", is_success)
        print("direction:", is_direction)
        print("non_collision:", non_collision)
        print("current success rate:", sum(is_success_result) / len(is_success_result)*100, "%")
        print("current direction rate:", sum(is_direction_result) / len(is_direction_result)*100, "%")
        print("current non_collision rate:", sum(non_collision_result) / len(non_collision_result)*100, "%")

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
    points = pred["pose_xyR_pred"][...,:2]  # [1, N*H, 3] descaled
    #points = batch['gt_pose_xyz_for_non_cond'] #NOTE: for debug
    guide_cost = pred["guide_losses"]["loss"]  # [1, N*H]
    #guide_cost = pred["guide_losses"]["collision_loss"]  # [1, N*H]
    #guide_cost = torch.zeros((1, 800))

    if intrinsics is None:
        intrinsics = np.array(
            [[619.0125, 0.0, 326.525], [0.0, 619.16775, 239.11084], [0.0, 0.0, 1.0]]
        )

    points_scene, idx = backproject(
        depth,
        intrinsics,
        np.logical_and(depth > 0, depth < 2),
        NOCS_convention=False,
    )
    T_plane, plane_model = get_tf_for_scene_rotation(points_scene)
    image_color = image[idx[0], idx[1], :]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_scene)

    points = points.cpu().numpy()
    guide_cost = guide_cost.cpu().numpy()

    points = points[0] # [N*H, 3]
    guide_cost = guide_cost[0] # [N*H]

    distances = np.sqrt(
        ((points_scene[:, :2][:, None, :] - points[:, :2]) ** 2).sum(axis=2)
    ) # x, y distance   
    scenepts_to_anchor_dist = np.min(distances, axis=1)  # [num_points]
    scenepts_to_anchor_id = np.argmin(distances, axis=1)  # [num_points]
    topk_points_id = np.argsort(scenepts_to_anchor_dist, axis=0)[: points.shape[0]]
    tokk_points_id_corr_anchor = scenepts_to_anchor_id[topk_points_id]

    #guide_cost = guide_cost[tokk_points_id_corr_anchor] # NOTE: uncomment if use the option 1 visualization
    guide_cost_color = get_heatmap(guide_cost[None], invert=False)[0]

    pcd.colors = o3d.utility.Vector3dVector(image_color)


    #points_for_place = points_scene[topk_points_id] # option1: use the topk nearest points to visualize
    
    points_for_place = points # option2: use the plane_model to get the points 
    points_for_place_z = (-plane_model[3] - plane_model[0] * points_for_place[..., 0] - plane_model[1] * points_for_place[..., 1]) / plane_model[2]
    points_for_place = np.concatenate([points_for_place, points_for_place_z[:, None]], axis=1)

    points_for_place_goal = np.mean(points_for_place, axis=0)
    #print("points_for_place_goal:", points_for_place_goal)

    # get the topk affordance points avg position and visualize
    affordance = batch["affordance"] # [B, 2048, 1]
    affordance = torch.max(affordance, dim=-1)[0] # [B, 2048] # NOTE: test
    position = batch["pc_position"] # [B, 2048, 3]
    affordance = affordance.squeeze(-1) # [B, 2048]
    k = 10
    topk_affordance, topk_idx = torch.topk(affordance, k, dim=-1)
    topk_positions = torch.gather(position, dim=1, index=topk_idx.unsqueeze(-1).expand(-1, -1, position.size(-1)))
    avg_topk_positions = topk_positions.mean(dim=1)  # (B, 3)
    avg_topk_positions = avg_topk_positions[0].cpu().numpy()
    avg_topk_mean_sphere = create_sphere_at_points(avg_topk_positions, radius=0.02, color=[1, 0, 0])
    second_sphere = create_sphere_at_points([0.061, -0.273, 1.455], radius=0.02, color=[0, 1, 0])

    #vis = [pcd, avg_topk_mean_sphere]
    vis = [pcd]
    #o3d.visualization.draw(vis)

    # print("points_for_place_goal:", points_for_place_goal)
    for ii, pos in enumerate(points_for_place):
        pos_vis = o3d.geometry.TriangleMesh.create_sphere()
        #pos_vis.compute_vertex_normals()
        pos_vis.scale(0.01, [0, 0, 0])
        pos_vis.translate(pos[:3])
        vis_color = guide_cost_color[ii]
        #vis_color = [1, 0, 0]
        pos_vis.paint_uniform_color(vis_color)
        vis.append(pos_vis)
    o3d.visualization.draw(vis)


def create_sphere_at_points(center, radius=0.05, color=[1, 0, 0]):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color(color)
    sphere.translate(center)
    return sphere


def get_heatmap(values, cmap_name="turbo", invert=False):
    if invert:
        values = -values
    values = (values - values.min()) / (values.max() - values.min())
    colormaps = cm.get_cmap(cmap_name)
    rgb = colormaps(values)[..., :3]  # don't need alpha channel
    return rgb



        
if __name__ == "__main__":
    seed_everything(42)
    dataset = BlendprocDesktopDataset_incompleted_sparse()
    #dataset = BlendprocDesktopDataset_incompleted_mult_cond()
    #dataset = BlendprocDesktopDataset_incompleted()
    print(len(dataset))
    GeoL_completed_metrics(
        dataset=BlendprocDesktopDataset_incompleted_sparse(),
        sample_datasize=len(dataset),
        pretrained_affordance="outputs/checkpoints/GeoL_v9_20K_meter_retrain_lr_1e-4_0213/ckpt_11.pth",
        pretrained_diffusion="outputs/checkpoints/GeoL_diffuser_v0_1K_scene_xyr/ckpt_31.pth",
        process_metric_func=process_success_metrics_GeoL_completed,
    )

    # GeoL_completed_metrics_mult_cond(
    #     dataset=BlendprocDesktopDataset_incompleted_sparse(),
    #     sample_datasize=len(dataset),
    #     pretrained_affordance="outputs/checkpoints/GeoL_v9_20K_meter_retrain_lr_1e-4_0213/ckpt_11.pth",
    #     pretrained_diffusion="outputs/checkpoints/GeoL_diffuser_v0_1K_scene_xyr/ckpt_31.pth",
    #     process_metric_func=process_success_metrics_GeoL_completed,
    # )

    #"outputs/checkpoints/GeoL_v9_10K_meter_retrain/ckpt_29.pth",
    #"outputs/checkpoints/GeoL_v9_20K_meter_retrain_lr_1e-4_0213/ckpt_11.pth"
    #"outputs/checkpoints/GeoL_diffuser_v0__topk_1K/ckpt_176.pth"