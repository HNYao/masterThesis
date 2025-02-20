import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Subset, random_split
from omegaconf import DictConfig
import cv2

from thirdpart.seeingunseen.seeing_unseen.core.base import BaseTrainer
from thirdpart.seeingunseen.seeing_unseen.core.registry import registry
import torch.multiprocessing as mp
from GeoL_net.models.GeoL import GeoL_net_v9
from metrics.dataset_factory import BlendprocDesktopDataset, BlendprocDesktopDataset_incompleted, BlendprocDesktopDataset_incompleted_sparse
from Geo_comb.pred_one_case_completed import pred_one_case_dataset, generate_heatmap_pc
from metrics.utils import *
from Geo_comb.pred_one_case_completed import rgb_obj_dect


def GeoL_metrics(
        dataset=BlendprocDesktopDataset(),
        sample_datasize=10,
        pretrained=None,
        process_metric_func=None, 
):
    # initialize the dataset
    num_sample = sample_datasize
    indices = torch.arange(num_sample)
    subset = Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)
    
    # initialize the model
    INTRINSICS = np.array([[607.09912/2 , 0. , 636.85083/2 ], [0., 607.05212/2, 367.35952/2], [0.0, 0.0, 1.0]])
    model = GeoL_net_v9(
        input_shape=(3, 480, 640),
        target_input_shape=(3, 128, 128),
        intrinsics=INTRINSICS
    )

    state_affordance_dict = torch.load(
        pretrained, map_location="cpu"
    )
    model.load_state_dict(state_affordance_dict["ckpt_dict"])

    model.eval()
    model.cuda()

    is_success_result = []
    is_direction_result = []
    is_in_box_result = []
    non_collision_result = []
    test_sum = 0
    for data_batch in dataloader:

        rgb_img_path = data_batch["image_without_obj_path"][0]
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


        print("anchor_obj_name:", anchor_obj_name)
        print("object_to_place:", object_to_place)
        print("direction:", direction)
        print("rgb_img_path:", rgb_img_path)

        # find anchor object
        annotated_frame = rgb_obj_dect(
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
            depth_img_path=depth_file_path,
        )
        dataloader_one_case = DataLoader(dataset_one_case, batch_size=1, shuffle=False)
        for i, batch in enumerate(dataloader_one_case):
            for key, val in batch.items():
                if type(val) == list:
                    continue
                batch[key] = val.float().to("cuda")

            with torch.no_grad():
                affordance_pred = model(batch=batch)["affordance"]
                affordance_value = affordance_pred[0].cpu().detach().numpy()
                pcd_affordance_map = generate_heatmap_pc(batch, affordance_pred, intrinsics=INTRINSICS, interpolate=False)

                coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                vis = [coordinate_frame, pcd_affordance_map]
                #o3d.visualization.draw_geometries(vis)
                #generate_heatmap_pc(batch, affordance_pred, intrinsics=INTRINSICS, interpolate=False)

                fps_points_pcd = batch["fps_points_scene"][0].cpu().numpy()
                fps_pcd = o3d.geometry.PointCloud()
                fps_pcd.points = o3d.utility.Vector3dVector(fps_points_pcd)
                is_success, is_direction, is_in_box, non_collision = process_metric_func(
                    fps_pcd,
                    affordance_value, 
                    mask_with_obj_path, 
                    mask_without_obj_path, 
                    depth_with_obj_path, 
                    depth_without_obj_path, 
                    hdf5_path, 
                    direction,
                    obj_mesh_path = data_batch["obj_mesh_path"][0],
                    scene_mesh_path = data_batch["scene_mesh_path"][0],
                    )
                is_success_result += is_success
                is_direction_result += is_direction
                is_in_box_result += is_in_box
                non_collision_result += non_collision
                test_sum+=1
                print('test_sum:', test_sum)
                print("current success rate:", sum(is_success_result) / len(is_success_result) * 100, "%")
                print("current direction rate:", sum(is_direction_result) / len(is_direction_result) * 100, "%")
                print("current in box rate:", sum(is_in_box_result) / len(is_in_box_result) * 100 , "%")
                print("current non collision rate:", sum(non_collision_result) / len(non_collision_result)*100, "%")
        
            break

    print("final success rate:", sum(is_success_result) / len(is_success_result)*100, "%")
    print("final direction rate:", sum(is_direction_result) / len(is_direction_result)*100, "%")
    print("final in box rate:", sum(is_in_box_result) / len(is_in_box_result)*100, "%")
    print("final non collision rate:", sum(non_collision_result) / len(non_collision_result)*100, "%")

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

        
if __name__ == "__main__":
    
    dataset = BlendprocDesktopDataset()
    #dataset = BlendprocDesktopDataset_incompleted()
    dataset = BlendprocDesktopDataset_incompleted_sparse()

    print(len(dataset))

    GeoL_metrics(
        dataset=dataset,
        sample_datasize=len(dataset),
        pretrained="outputs/checkpoints/GeoL_v9_20K_meter_retrain_lr_1e-4_0213/ckpt_11.pth",
        process_metric_func=process_success_metrics_GeoL,
    )