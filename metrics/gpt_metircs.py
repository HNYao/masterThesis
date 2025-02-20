import torch
from torch.utils.data import DataLoader, Subset, random_split
from metrics.dataset_factory import BlendprocDesktopDataset, BlendprocDesktopDataset_incompleted, BlendprocDesktopDataset_incompleted_sparse
from metrics.utils import *
from GeoL_net.gpt.gpt import *
def gpt_metrics(
        dataset=BlendprocDesktopDataset(),
        sample_datasize=10,
        gpt_version = "gpt-4o-mini",
        process_metric_func = process_direction_metrics
):
    
    num_sample = sample_datasize
    indices = torch.arange(num_sample)
    subset = Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)

    # intialize the model



    is_success_result = []
    is_direction_result = []
    is_in_box_result = []
    non_collision_result = []
    test_sum = 0
    for data_batch in dataloader:

        img_path = data_batch["image_without_obj_path"][0]
        image = Image.open(img_path)
        object_to_place = data_batch["object_name"][0]
        direction = data_batch["direction"][0]
        anchor_obj_name = data_batch["anchor_obj_name"][0]

        bbox = chatgpt_object_placement_bbox(
            gpt_version=gpt_version,
            image_path=img_path,
            prompts_obj_place=object_to_place,
            prompts_direction=direction,
            prompts_anchor_obj=anchor_obj_name,
        )

        if bbox is None:
            print("Failed to find position")
            is_success = [False, False, False, False, False]
            is_direction = [False, False, False, False, False]
            is_in_box = [False, False, False, False, False]
            non_collision = [False, False, False, False, False]
            test_sum+=1
        
        else:

            image = np.array(image)
            updated_img = sample_points_in_bbox(image, bbox, n=5)

            is_success, is_direction, is_in_box, non_collision = process_metric_func(
                image_sampled_point=updated_img, 
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
        print("current success rate:", sum(is_success_result) / len(is_success_result) * 100, "%")
        print("current direction rate:", sum(is_direction_result) / len(is_direction_result) * 100, "%")
        print("current in box rate:", sum(is_in_box_result) / len(is_in_box_result) * 100 , "%")
        print("current non collision rate:", sum(non_collision_result) / len(non_collision_result)*100, "%")

    
    print("final success rate:", sum(is_success_result) / len(is_success_result)*100, "%")
    print("final direction rate:", sum(is_direction_result) / len(is_direction_result)*100, "%")
    print("final in box rate:", sum(is_in_box_result) / len(is_in_box_result)*100, "%")
    print("final non collision rate:", sum(non_collision_result) / len(non_collision_result)*100, "%")


if __name__ == "__main__":
    dataset = BlendprocDesktopDataset_incompleted_sparse()
    total_resualt = gpt_metrics(
        dataset, 
        sample_datasize=len(dataset), 
        gpt_version="gpt-4o", 
        process_metric_func=process_success_metrics)