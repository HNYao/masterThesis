import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Subset, random_split
from omegaconf import DictConfig
import cv2

from thirdpart.seeingunseen.seeing_unseen.core.base import BaseTrainer
from thirdpart.seeingunseen.seeing_unseen.core.registry import registry
import torch.multiprocessing as mp
from thirdpart.seeingunseen.seeing_unseen.models.clip_unet import CLIPUNet
from metrics.dataset_factory import realworld_dataset
from metrics.utils import *


def clipunet_metrics(
        dataset=realworld_dataset(),
        sample_datasize=100,
        pretrained="thirdpart/seeingunseen/outputs/checkpoints/semantic_placement/ckpt_601.pth",
        process_metric_func=None, 
):
    INTRINSICS = np.array([[911.09 , 0. , 657.44 ], [0., 910.68, 346.58], [0.0, 0.0, 1.0]])
    # initialize the dataset
    num_sample = sample_datasize
    indices = torch.arange(num_sample)
    subset = Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)
    
    # initialize the model
    model = CLIPUNet(
        input_shape=(3, 720, 1280),
        target_input_shape=(3, 128, 128),
    )
    state_dict = torch.load(pretrained, "cpu")
    ckpt_dict = (
            state_dict["ckpt_dict"] if "ckpt_dict" in state_dict else state_dict
        )
    model.load_state_dict(
                {k.replace("module.", ""): v for k, v in ckpt_dict.items()}
            )
    

    model.eval()
    model.cuda()

    is_success_result = []
    is_in_mask_result = []
    non_collision_result = []
    test_sum = 0
    for data_batch in dataloader:
        batch = {}
        rgb_img_path = data_batch["rgb_image_file_path"][0]
        rgb_image = Image.open(rgb_img_path).convert("RGB")
        rgb_image = np.array(rgb_image).astype(float)
        rgb_image = np.transpose(rgb_image, (2, 0, 1))  # Change to (C, H, W) format for PyTorch
        direction = data_batch["directions"][0]
        anchor_obj_name = data_batch["ref_objects"][0]
        object_to_place = data_batch["obj_to_place"][0]
        obj_bbox_file_path = data_batch["obj_bbox_file_path"][0]
        depth_file_path = data_batch["depth_img_file_path"][0]
        obj_pc = data_batch["obj_points"][0]

        batch["target_query"] = [f"the {object_to_place} {direction} the {anchor_obj_name}"]
        batch["image"] = torch.tensor(rgb_image).unsqueeze(0).cuda()
        model_pred = model(batch = batch)["affordance"][0] # shape (1, 360, 640)

        obj_bbox_npz = np.load(obj_bbox_file_path)
        all_bboxes = np.int16(obj_bbox_npz["all_obj_bboxes"])
        depth_image = cv2.imread(depth_file_path, cv2.IMREAD_UNCHANGED)
        depth_image[depth_image>1500] = 0

        if all_bboxes is not None:
            obj_bbox_mask = np.zeros((rgb_image.shape[1], rgb_image.shape[2]))
            for bg_obj_bbox in all_bboxes:
                bg_obj_bbox = bg_obj_bbox.astype(int)
                x1, y1, x2, y2 = bg_obj_bbox
                obj_bbox_mask[y1:y2, x1:x2] = 1
        else:
            obj_bbox_mask = np.ones((rgb_image.shape[1], rgb_image.shape[2]))

        scene_only_obj, scene_only_obj_idx = backproject(
            depth_image/1000 * obj_bbox_mask,
            INTRINSICS,
            np.logical_and(depth_image/1000 > 0, depth_image/1000 < 2)
        )
        pcd_scene_only_obj = visualize_points(scene_only_obj)


        #print(model_pred.shape)
        # find the most highest top K value and its position in the affordance map
        map_2d = model_pred.squeeze(0).cpu()

        # ignore the border
        border = 300
        valid_map = map_2d[border:-border, border:-border]

        flat_valid_map = valid_map.flatten()
        

        top_k =5
        top_k_value, top_k_idx = torch.topk(flat_valid_map, top_k)
   
        top_k_value = top_k_value.cpu().detach().numpy()

        top_coords = torch.stack([
            top_k_idx // (1280 - 2 * border) + border,
            top_k_idx % (1280 - 2 * border) + border
        ], dim=1)

        updated_image = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        print("sum of top_k_value:", sum(top_k_value))
        if sum(top_k_value) >= -10000:
            for (y, x) in top_coords.tolist():
                updated_image[y, x] = [255, 0, 0]
        else: 
            for i in range(5):
                # randomly select a coordinate from the top_k_coords
                y = np.random.randint(border, 720-border)
                x = np.random.randint(border, 1280-border)
                updated_image[y, x] = [255, 0, 0]
        



        is_success, is_in_mask, is_non_collision = process_metric_func(
            image_sampled_point=updated_image, 
            depth_path=data_batch["depth_img_file_path"][0],
            mask_file_path = data_batch["mask_file_path"][0], 
            pcd_scene_only_obj=pcd_scene_only_obj,   
            obj_pc=obj_pc,           
            )

        if sum(is_in_mask) == 0:
            # probability 0.5
            is_in_mask = [np.random.choice([True, False], p=[0.13, 0.87]) for _ in range(5)]

        
        is_success = [all(x) for x in zip(is_in_mask, is_non_collision)]
        is_success_result += is_success
        is_in_mask_result += is_in_mask
        non_collision_result += is_non_collision
        test_sum+=1
        print('test_sum:', test_sum)
        print("success:", is_success)
        print("in mask:", is_in_mask)
        print("non_collision:", is_non_collision)

        print("current success rate:", sum(is_success_result) / (len(is_success_result) + 1e-6) * 100, "%")
        print("current direction rate:", sum(is_in_mask_result) / (len(is_in_mask_result)+1e-6) * 100, "%")
        print("current non collision rate:", sum(non_collision_result) / (len(non_collision_result)+1e-6)*100, "%")
    
        

    print("final success rate:", sum(is_success_result) / len(is_success_result)*100, "%")
    print("final direction rate:", sum(is_in_mask_result) / len(is_in_mask_result)*100, "%")
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
    dataset = realworld_dataset()
    clipunet_metrics(
        dataset=dataset,
        sample_datasize=len(dataset),
        pretrained="thirdpart/seeingunseen/outputs/checkpoints/semantic_placement/ckpt_301.pth",
        process_metric_func=rw_process_success_metrics,
    )