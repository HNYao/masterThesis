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
from metrics.dataset_factory import BlendprocDesktopDataset, BlendprocDesktopDataset_incompleted_sparse
from metrics.utils import *


def clipunet_metrics(
        dataset=BlendprocDesktopDataset(),
        sample_datasize=100,
        pretrained="thirdpart/seeingunseen/outputs/checkpoints/semantic_placement/ckpt_601.pth",
        process_metric_func=None, 
):
    # initialize the dataset
    num_sample = sample_datasize
    indices = torch.arange(num_sample)
    subset = Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)
    
    # initialize the model
    model = CLIPUNet(
        input_shape=(3, 480, 640),
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
    is_direction_result = []
    is_in_box_result = []
    non_collision_result = []
    test_sum = 0
    for data_batch in dataloader:
        batch = {}
        rgb_img_path = data_batch["image_without_obj_path"][0]
        rgb_image = Image.open(rgb_img_path).convert("RGB")
        rgb_image = np.array(rgb_image).astype(float)
        rgb_image = np.transpose(rgb_image, (2, 0, 1))  # Change to (C, H, W) format for PyTorch
        object_to_place = data_batch["object_name"][0]
        direction = data_batch["direction"][0]
        anchor_obj_name = data_batch["anchor_obj_name"][0]

        batch["target_query"] = [f"the {object_to_place} {direction} the {anchor_obj_name}"]
        batch["image"] = torch.tensor(rgb_image).unsqueeze(0).cuda()
        model_pred = model(batch = batch)["affordance"][0] # shape (1, 360, 640)

        #print(model_pred.shape)
        # find the most highest top K value and its position in the affordance map
        map_2d = model_pred.squeeze(0).cpu()

        # ignore the border
        border = 40
        valid_map = map_2d[border:-border, border:-border]

        flat_valid_map = valid_map.flatten()
        top_k =5
        top_k_value, top_k_idx = torch.topk(flat_valid_map, top_k)
   
        top_k_value = top_k_value.cpu().detach().numpy()

        top_coords = torch.stack([
            top_k_idx // (640 - 2 * border) + border,
            top_k_idx % (640 - 2 * border) + border
        ], dim=1)

        updated_image = np.zeros((360, 640, 3), dtype=np.uint8)
        for (y, x) in top_coords.tolist():
            updated_image[y, x] = [255, 0, 0]

        is_success, is_direction, is_in_box, non_collision = process_metric_func(
                    image_sampled_point=updated_image, 
                    mask_with_obj_path=data_batch["mask_with_obj_path"][0],
                    mask_without_obj_path=data_batch["mask_without_obj_path"][0],
                    depth_with_obj_path=data_batch["depth_with_obj_path"][0],
                    depth_without_obj_path=data_batch["depth_without_obj_path"][0],
                    hdf5_path=data_batch["hdf5_path"][0],
                    direction = direction,
                    obj_mesh_path = data_batch["obj_mesh_path"][0],
                    scene_mesh_path = data_batch["scene_mesh_path"][0],

                    )
                
        is_success_result += is_success
        is_direction_result += is_direction
        is_in_box_result += is_in_box
        non_collision_result += non_collision
        test_sum+=1
        print('test_sum:', test_sum)
        print("success:", is_success)
        print("direction:", is_direction)
        print("in box:", is_in_box)
        print("non_collision:", non_collision)

        print("current success rate:", sum(is_success_result) / (len(is_success_result) + 1e-6) * 100, "%")
        print("current direction rate:", sum(is_direction_result) / (len(is_direction_result)+1e-6) * 100, "%")
        print("current in box rate:", sum(is_in_box_result) / (len(is_in_box_result)+1e-6) * 100 , "%")
        print("current non collision rate:", sum(non_collision_result) / (len(non_collision_result)+1e-6)*100, "%")
    
        

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
    dataset = BlendprocDesktopDataset_incompleted_sparse()
    clipunet_metrics(
        dataset=dataset,
        sample_datasize=len(dataset),
        pretrained="thirdpart/seeingunseen/outputs/checkpoints/semantic_placement/ckpt_301.pth",
        process_metric_func=process_success_metrics,
    )