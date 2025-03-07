'''
    predict one case
'''

from GeoL_net.core.registry import registry
import open3d as o3d
import torch
from PIL import Image
from pointnet2_ops import pointnet2_utils
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import numpy as np
import torch
import json
import os
from GeoL_net.dataset_gen.RGBD2PC import backproject, visualize_points
from GeoL_net.models.modules import ProjectColorOntoImage_v3, ProjectColorOntoImage
from scipy.spatial.distance import cdist
from matplotlib import cm
import torchvision.transforms as T
from GeoL_net.gpt.gpt import chatgpt_condition



def generate_heatmap_target_point(batch, model_pred):
    """
    Generate heatmap for the model prediction and groud truth mask

    Parameters:
    batch: dict
        batch of data
    model_pred: torch.tensor (default: None) [b, num_points, 1]
        model prediction
    
    Returns:
    img_pred_list: list
        list of PIL images of the model prediction
    img_gt_list: list
        list of PIL images of the ground truth mask
    file_path: list
        list of file path
    phrase: list
        list of phrase
    """
    intrinsics = np.array([[591.0125 ,   0.     , 322.525  ],[  0.     , 590.16775, 244.11084],[  0.     ,   0.     ,   1.     ]])
    #intrinsics = np.array([[619.0125 ,   0.     , 326  ], [  0.     , 619, 239], [  0.     ,   0.     ,   1.     ]])
    #intrinsics = np.array([[607.0125 ,   0.     , 636.525  ], [  0.     , 607.16775, 367.11084], [  0.     ,   0.     ,   1.     ]])
    img_pred_list = []
    img_gt_list = []
    img_rgb_list = batch["image"].cpu() # the image of the scene [b,c, h, w]

    
    feat = model_pred.sigmoid()
    min_feat = feat.min(dim=1, keepdim=True)[0]
    max_feat = feat.max(dim=1, keepdim=True)[0]
    normalized_pred_feat = (feat - min_feat) / (max_feat - min_feat + 1e-6)

    turbo_colormap = cm.get_cmap('turbo', 256) # get the color map for the prediction and ground truth

    # normalize the prediction and ground truth
   
    normalized_pred_feat_np = normalized_pred_feat.cpu().detach().numpy()

    # get the color map for the prediction and ground truth [b, num_points, 3]
    
    
    color_pred_maps = turbo_colormap(normalized_pred_feat_np)[:, :, :, :3] # [b, num_points, 3] ignore alpha
    color_pred_maps = torch.from_numpy(color_pred_maps).squeeze(2).cpu()
    
    projector = ProjectColorOntoImage()


    pcs = []
    color_pred_list = []
    for i in range(batch['fps_points_scene'].shape[0]):
        depth = batch['depth'][i].cpu().numpy()
        fps_points_scene = batch['fps_points_scene'][i].cpu().numpy()
        #fps_colors = batch['fps_colors_scene'][i].cpu().numpy()
        points_scene, _ = backproject(depth, intrinsics, np.logical_and(depth > 0, depth > 0), NOCS_convention=False)
        pcs.append(points_scene)

        distance_pred= cdist(points_scene, fps_points_scene)
        nearest_pred_idx = np.argmin(distance_pred, axis=1)
        color_pred_map = color_pred_maps[i]
        color_pred_scene = color_pred_map[nearest_pred_idx]
        color_pred_list.append(color_pred_scene)
        



    #pcs = torch.tensor(pcs, dtype=torch.float32) # list to tensor
    output_pred_img_list = []

    for i in range(len(pcs)):
        output_pred_img = projector(image_grid = img_rgb_list[i],
                            query_points = torch.tensor(pcs[i]),
                            query_colors = color_pred_list[i],
                            intrinsics = intrinsics)
        output_pred_img_list.append(output_pred_img)

    
    # merge the image and heatmap of prediction
    for i, pred_img in enumerate(output_pred_img_list):
        color_image = T.ToPILImage()(img_rgb_list[i].cpu())
        pil_img = T.ToPILImage()(pred_img.squeeze(0).cpu())

        image_np = np.clip(pil_img, 0, 255)

        color_image_np = np.floor(color_image)
        color_image_np = np.clip(color_image_np, 0, 255)
        color_image_np = np.uint8(color_image_np)

        image_np = cv2.addWeighted(image_np, 0.4, color_image_np, 0.6, 0.0)
        pil_image = Image.fromarray(np.uint8(image_np))
        img_pred_list.append(pil_image)

    # merge the image and heatmap of ground truth


    return img_pred_list, img_gt_list, batch['file_path'], batch['phrase']


def generate_heatmap_pc(batch, model_pred, intrinsics=None):
    if intrinsics is None:
        intrinsics =  np.array([[607.09912/2 ,   0.     , 636.85083/2  ],
                [  0.     , 607.05212/2, 367.35952/2],
                [  0.     ,   0.     ,   1.     ]])

    #intrinsics = np.array([[591.0125 ,   0.     , 322.525  ],[  0.     , 590.16775, 244.11084],[  0.     ,   0.     ,   1.     ]])
    #intrinsics = np.array([[619.0125 ,   0.     , 326  ], [  0.     , 619, 239], [  0.     ,   0.     ,   1.     ]])
    #intrinsics = np.array([[607.0125 ,   0.     , 636.525  ], [  0.     , 607.16775, 367.11084], [  0.     ,   0.     ,   1.     ]])
    img_pred_list = []
    img_gt_list = []
    img_rgb_list = batch["image"].cpu() # the image of the scene [b,c, h, w]

    
    feat = model_pred.sigmoid()
    min_feat = feat.min(dim=1, keepdim=True)[0]
    max_feat = feat.max(dim=1, keepdim=True)[0]
    normalized_pred_feat = (feat - min_feat) / (max_feat - min_feat + 1e-6)

    turbo_colormap = cm.get_cmap('turbo', 256) # get the color map for the prediction and ground truth

    # normalize the prediction and ground truth
   
    normalized_pred_feat_np = normalized_pred_feat.cpu().detach().numpy()

    # get the color map for the prediction and ground truth [b, num_points, 3]
    
    
    color_pred_maps = turbo_colormap(normalized_pred_feat_np)[:, :, :, :3] # [b, num_points, 3] ignore alpha
    color_pred_maps = torch.from_numpy(color_pred_maps).squeeze(2).cpu()
    
    pcs = []
    color_pred_list = []
    for i in range(batch['fps_points_scene'].shape[0]):
        depth = batch['depth'][i].cpu().numpy()
        fps_points_scene = batch['fps_points_scene'][i].cpu().numpy()
        #fps_colors = batch['fps_colors_scene'][i].cpu().numpy()
        points_scene, _ = backproject(depth, intrinsics, np.logical_and(depth > 0, depth < 1500), NOCS_convention=False)
        pcs.append(points_scene)

        distance_pred= cdist(points_scene, fps_points_scene)
        nearest_pred_idx = np.argmin(distance_pred, axis=1)
        color_pred_map = color_pred_maps[i]
        color_pred_scene = color_pred_map[nearest_pred_idx]
        color_pred_list.append(color_pred_scene)
        

    for i, pc in enumerate(pcs):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd.colors = o3d.utility.Vector3dVector(color_pred_list[i].cpu().numpy())
        o3d.visualization.draw_geometries([pcd])



        



def is_red(color, tolerance=0.1):
    return (color[0] > 1-tolerance and color[1] < tolerance and color[2] < tolerance)

class pred_one_case_dataset(Dataset):
    def __init__(self, scene_pcd, rgb_image_file_path, target_name, direction_text, depth_img_path):
        self.rgb_image_file_path = rgb_image_file_path
        self.target_name = target_name
        self.direction_text = direction_text


        self.scene_pcd = scene_pcd
        self.scene_pcd_points = np.asarray(self.scene_pcd.points)
        #self.scene_pcd_points = (np.linalg.inv(cam_rotation_matrix) @ self.scene_pcd_points.T).T
        self.scene_pcd_tensor = torch.tensor(self.scene_pcd_points, dtype=torch.float32).unsqueeze(0)



        # 从旋转后的points中提取出red的点，计算绿色点的位置的均值
        self.scene_pcd_colors = np.asarray(self.scene_pcd.colors)
        green_mask = np.apply_along_axis(is_red, 1, self.scene_pcd_colors)
        green_points = self.scene_pcd_points[green_mask]
        self.green_pcd_center = np.mean(green_points, axis=0)


        self.scene_pcd_tensor = self.scene_pcd_tensor.to("cuda")

        fps_indices_scene = pointnet2_utils.furthest_point_sample(self.scene_pcd_tensor.contiguous(), 2048)
        fps_indices_scene_np = fps_indices_scene.squeeze(0).cpu().numpy()
        fps_points_scene_from_original = self.scene_pcd_points[fps_indices_scene_np]
        self.fps_points_scene_from_original = fps_points_scene_from_original
        self.fps_colors_scene_from_original = self.scene_pcd_colors[fps_indices_scene_np]

        rgb_image = Image.open(rgb_image_file_path).convert("RGB")
        #if rgb_image.height != 480 or rgb_image.width != 640:
        #    rgb_image = rgb_image.resize((640, 480))
        rgb_image = np.asarray(rgb_image).astype(float)
        rgb_image = np.transpose(rgb_image, (2, 0, 1))

        self.rgb_image = rgb_image
        self.depth = np.array(cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)).astype(float)
    
    def __len__(self):
        return 2
    
    def __getitem__(self, index):
        sample = {
            "fps_points_scene": self.fps_points_scene_from_original,
            "fps_colors_scene": self.fps_colors_scene_from_original,
            "phrase": "no phrase in testing mode",
            "image": self.rgb_image,
            "mask": "no mask in testing mode",
            "file_path": "dataset/scene_RGBD_mask/id000/book_0000_fake", #fake
            "reference_obj": self.target_name,
            "direction_text": self.direction_text,
            "anchor_position": self.green_pcd_center,
            "depth": self.depth
        }
        return sample


def rgb_obj_dect(image_path, text_prompt, out_dir=None, model_path = "GroundingDINO/weights/groundingdino_swint_ogc.pth"):

    model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", model_path)
    IMAGE_PATH = image_path
    TEXT_PROMPT = text_prompt
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    image_source, image = load_image(IMAGE_PATH)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    h, w, _ = image_source.shape
    ori_boxes = boxes * torch.Tensor([w, h, w, h])
    ori_boxes = torch.round(ori_boxes)

    center_x = int(ori_boxes[0][0].item())
    center_y = int(ori_boxes[0][1].item())
    if out_dir is not None:
        #print("orignal boxes cxcy:", ori_boxes, ori_boxes[0][0], ori_boxes[0][1])
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        annotated_frame[:] = 0
        cv2.circle(annotated_frame, (center_x, center_y), 5, (255,0 ,0), -1)
        cv2.imwrite(out_dir, annotated_frame)
    
    return annotated_frame

if __name__ == "__main__":

    #congiguration
    #scene_pcd_file_path = "dataset/scene_RGBD_mask_direction_mult/id10_1/clock_0001_normal/mask_Behind.ply"
    # blendproc dataset
    #rgb_image_file_path = "dataset/test/scene_RGBD_mask_direction_mult/id117_2/cup_0001_red/img.jpg"
    #depth_image_file_path = "dataset/test/scene_RGBD_mask_direction_mult/id117_2/cup_0001_red/000000.png"

    # kinect data
    rgb_image_file_path = "dataset/kinect_dataset/color/000025.png"
    depth_image_file_path = "dataset/kinect_dataset/depth/000025.png"

    # realsense data
    rgb_image_file_path = "dataset/realsense/color/000054.png"
    depth_image_file_path = "dataset/realsense/depth/000054.png"

    use_chatgpt = True
    if use_chatgpt:
        target_name, direction_text = chatgpt_condition(rgb_image_file_path, "object_placement")
        print("====> Predicting Affordance...")
    else:
        target_name = "the Monitor"
        direction_text = "Front"

    # use GroundingDINO to detect the target object
    annotated_frame = rgb_obj_dect(rgb_image_file_path, target_name, "exps/pred_one/RGB_ref.jpg")
    color_no_obj = np.array(annotated_frame) / 255


    depth = cv2.imread(depth_image_file_path, cv2.IMREAD_UNCHANGED)
    depth = depth.astype(np.float32)

    #intr = np.array([[591.0125 ,   0.     , 322.525  ],[  0.     , 590.16775, 244.11084],[  0.     ,   0.     ,   1.     ]])
    intr = np.array([[619.0125 ,   0.     , 326.525  ],[  0.     , 619.16775, 239.11084],[  0.     ,   0.     ,   1.     ]])

    #intr = np.array([[607.0125 ,   0.     , 636.525  ], [  0.     , 607.16775, 367.11084], [  0.     ,   0.     ,   1.     ]]) # kinect
    points_no_obj_scene, scene_no_obj_idx = backproject(
        depth,
        intr,
        np.logical_and(depth > 0, depth < 6500),
        NOCS_convention=False,
    )
    colors_no_obj_scene = color_no_obj[scene_no_obj_idx[0], scene_no_obj_idx[1]]
    
    pcd_no_obj_scene = visualize_points(points_no_obj_scene, colors_no_obj_scene)

    scaling_factors =  np.array([2,2,5])
    # Create a scaling matrix
    scaling_matrix = np.eye(4)  # 4x4 identity matrix
    scaling_matrix[0, 0] = scaling_factors[0]  # Scale x
    scaling_matrix[1, 1] = scaling_factors[1]  # Scale y
    scaling_matrix[2, 2] = scaling_factors[2]  # Scale z

    # load the model
    model_cls=  registry.get_affordance_model("GeoL_net_v9")
    model = model_cls(input_shape=(3,720,1280),target_input_shape=(3,128,128)).to("cuda")

    # load the checkpoint
    state_dict = torch.load("outputs/checkpoints/GeoL_v9/ckpt_211.pth", map_location="cpu")
    model.load_state_dict(state_dict["ckpt_dict"])

    # create the dataset
    dataset = pred_one_case_dataset(pcd_no_obj_scene, rgb_image_file_path, target_name, direction_text, depth_image_file_path)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model.eval()
    for i ,batch in enumerate(data_loader):
        for key, val in batch.items():
            if type(val) == list:
                continue
            batch[key] = val.float().to("cuda")
        with torch.no_grad():
            result = model(batch=batch)["affordance"].squeeze(1)
                
            generate_heatmap_pc(batch, result, intrinsics=intr)
            break


    # visualize or save the result
