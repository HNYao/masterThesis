"""
    predict one case
    1. GeoL_net predicts affordance heatmap
    2. GeoL_diffuser predicts 4d pose
"""

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
from GeoL_diffuser.models.guidance import *
from GeoL_diffuser.models.utils.fit_plane import *
from scipy.spatial.distance import cdist
from matplotlib import cm
import torchvision.transforms as T
from GeoL_net.gpt.gpt import chatgpt_condition, chatgpt_select_id
from GeoL_diffuser.algos.pose_algos import PoseDiffusionModel
import yaml
from omegaconf import OmegaConf
import trimesh
from sklearn.manifold import TSNE
from Geo_comb.groundingdino_chatgpt import *


def get_heatmap(values, cmap_name="turbo", invert=False):
    if invert:
        values = -values
    values = (values - values.min()) / (values.max() - values.min())
    colormaps = cm.get_cmap(cmap_name)
    rgb = colormaps(values)[..., :3]  # don't need alpha channel
    return rgb

def generate_heatmap_target_point(batch, model_pred, intrinsics=None):
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
    if not intrinsics:
        intrinsics = np.array(
            [[591.0125, 0.0, 322.525], [0.0, 590.16775, 244.11084], [0.0, 0.0, 1.0]]
        )
        # intrinsics = np.array([[619.0125 ,   0.     , 326  ], [  0.     , 619, 239], [  0.     ,   0.     ,   1.     ]])
        # intrinsics = np.array([[607.0125 ,   0.     , 636.525  ], [  0.     , 607.16775, 367.11084], [  0.     ,   0.     ,   1.     ]])
    img_pred_list = []
    img_gt_list = []
    img_rgb_list = batch["image"].cpu()  # the image of the scene [b,c, h, w]

    feat = model_pred.sigmoid()
    min_feat = feat.min(dim=1, keepdim=True)[0]
    max_feat = feat.max(dim=1, keepdim=True)[0]
    normalized_pred_feat = (feat - min_feat) / (max_feat - min_feat + 1e-6)

    turbo_colormap = cm.get_cmap(
        "turbo", 256
    )  # get the color map for the prediction and ground truth

    # normalize the prediction and ground truth

    normalized_pred_feat_np = normalized_pred_feat.cpu().detach().numpy()

    # get the color map for the prediction and ground truth [b, num_points, 3]

    color_pred_maps = turbo_colormap(normalized_pred_feat_np)[
        :, :, :, :3
    ]  # [b, num_points, 3] ignore alpha
    color_pred_maps = torch.from_numpy(color_pred_maps).squeeze(2).cpu()

    projector = ProjectColorOntoImage()

    pcs = []
    color_pred_list = []
    for i in range(batch["fps_points_scene"].shape[0]):
        depth = batch["depth"][i].cpu().numpy()
        fps_points_scene = batch["fps_points_scene"][i].cpu().numpy()
        # fps_colors = batch['fps_colors_scene'][i].cpu().numpy()
        points_scene, _ = backproject(
            depth,
            intrinsics,
            np.logical_and(depth > 0, depth < 2),
            NOCS_convention=False,
        )
        pcs.append(points_scene)

        distance_pred = cdist(points_scene, fps_points_scene)
        nearest_pred_idx = np.argmin(distance_pred, axis=1)
        color_pred_map = color_pred_maps[i]
        color_pred_scene = color_pred_map[nearest_pred_idx]
        color_pred_list.append(color_pred_scene)

    # pcs = torch.tensor(pcs, dtype=torch.float32) # list to tensor
    output_pred_img_list = []

    for i in range(len(pcs)):
        output_pred_img = projector(
            image_grid=img_rgb_list[i],
            query_points=torch.tensor(pcs[i]),
            query_colors=color_pred_list[i],
            intrinsics=intrinsics,
        )
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

    return img_pred_list, img_gt_list, batch["file_path"], batch["phrase"]


def generate_heatmap_pc(batch, model_pred, intrinsics=None, interpolate=False):
    if intrinsics is None:
        intrinsics = np.array(
            [
                [607.09912 / 2, 0.0, 636.85083 / 2],
                [0.0, 607.05212 / 2, 367.35952 / 2],
                [0.0, 0.0, 1.0],
            ]
        )

    # intrinsics = np.array([[591.0125 ,   0.     , 322.525  ],[  0.     , 590.16775, 244.11084],[  0.     ,   0.     ,   1.     ]])
    # intrinsics = np.array([[619.0125 ,   0.     , 326  ], [  0.     , 619, 239], [  0.     ,   0.     ,   1.     ]])
    # intrinsics = np.array([[607.0125 ,   0.     , 636.525  ], [  0.     , 607.16775, 367.11084], [  0.     ,   0.     ,   1.     ]])
    feat = model_pred.sigmoid()
    min_feat = feat.min(dim=1, keepdim=True)[0]
    max_feat = feat.max(dim=1, keepdim=True)[0]
    normalized_pred_feat = (feat - min_feat) / (max_feat - min_feat + 1e-6)
    turbo_colormap = cm.get_cmap(
        "turbo", 256
    )  # get the color map for the prediction and ground truth

    # normalize the prediction and ground truth
    normalized_pred_feat_np = normalized_pred_feat.cpu().detach().numpy()

    # get the color map for the prediction and ground truth [b, num_points, 3]
    color_pred_maps = turbo_colormap(normalized_pred_feat_np)[
        :, :, :, :3
    ]  # [b, num_points, 3] ignore alpha
    color_pred_maps = torch.from_numpy(color_pred_maps).squeeze(2).cpu()

    pcs = []
    color_pred_list = []
    color_img_list = []
    for i in range(batch["fps_points_scene"].shape[0]):
        depth = batch["depth"][i].cpu().numpy()
        image_color = batch["image"][i].permute(1, 2, 0).cpu().numpy()
        fps_points_scene = batch["fps_points_scene"][i].cpu().numpy()
        points_scene, idx = backproject(
            depth,
            intrinsics,
            np.logical_and(depth > 0, depth < 2),
            NOCS_convention=False,
        )
        image_color = image_color[idx[0], idx[1], :]
        pcs.append(points_scene)

        if interpolate:
            distance_pred = cdist(points_scene, fps_points_scene)

            # find the nearest 5 points in the scene points
            nearest_pred_idx = np.argmin(distance_pred, axis=1)
            nearest_10_idx = np.argsort(distance_pred, axis=1)[:, :10]

            # nearest_pred_idx = np.argmin(distance_pred, axis=1)
            color_pred_map = color_pred_maps[i]
            # color_pred_scene = color_pred_map[nearest_pred_idx]
            color_pred_scene = color_pred_map[nearest_10_idx].mean(axis=1)
            pred_value_thershold = 0.3  # for visualization
            pred_value = normalized_pred_feat_np[0, nearest_10_idx, :].mean(axis=1)

        else:
            distance_pred = cdist(points_scene, fps_points_scene)
            nearest_pred_idx = np.argmin(distance_pred, axis=1)
            color_pred_map = color_pred_maps[i]
            color_pred_scene = color_pred_map[nearest_pred_idx]
            pred_value_thershold = 0.3

        color_pred_list.append(color_pred_scene)
        color_img_list.append(image_color / 255)

    for i, pc in enumerate(pcs):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd.colors = o3d.utility.Vector3dVector(
            color_pred_list[i].cpu().numpy() * 0.3 + color_img_list[i]* 255 * 0.7
        )
        #o3d.io.write_point_cloud(f"test_front.ply", pcd)
        o3d.visualization.draw_geometries([pcd])

    return pcd

def generate_heatmap_feature_pc(batch, model_pred, intrinsics=None, interpolate=False):
    if intrinsics is None:
        intrinsics = np.array(
            [
                [607.09912 / 2, 0.0, 636.85083 / 2],
                [0.0, 607.05212 / 2, 367.35952 / 2],
                [0.0, 0.0, 1.0],
            ]
        )

    # intrinsics = np.array([[591.0125 ,   0.     , 322.525  ],[  0.     , 590.16775, 244.11084],[  0.     ,   0.     ,   1.     ]])
    # intrinsics = np.array([[619.0125 ,   0.     , 326  ], [  0.     , 619, 239], [  0.     ,   0.     ,   1.     ]])
    # intrinsics = np.array([[607.0125 ,   0.     , 636.525  ], [  0.     , 607.16775, 367.11084], [  0.     ,   0.     ,   1.     ]])

    normalized_pred_feat = model_pred  # [b, num_points, 3]


    # normalize the prediction and ground truth
    normalized_pred_feat_np = normalized_pred_feat

    # get the color map for the prediction and ground truth [b, num_points, 3]

    color_pred_maps = torch.from_numpy(normalized_pred_feat_np)

    pcs = []
    color_pred_list = []
    color_img_list = []
    for i in range(batch["fps_points_scene"].shape[0]):
        depth = batch["depth"][i].cpu().numpy()
        image_color = batch["image"][i].permute(1, 2, 0).cpu().numpy()
        fps_points_scene = batch["fps_points_scene"][i].cpu().numpy()
        points_scene, idx = backproject(
            depth,
            intrinsics,
            np.logical_and(depth > 0, depth < 2),
            NOCS_convention=False,
        )
        image_color = image_color[idx[0], idx[1], :]
        pcs.append(points_scene)

        if interpolate:
            distance_pred = cdist(points_scene, fps_points_scene)

            # find the nearest 5 points in the scene points
            nearest_pred_idx = np.argmin(distance_pred, axis=1)
            nearest_10_idx = np.argsort(distance_pred, axis=1)[:, :10]

            # nearest_pred_idx = np.argmin(distance_pred, axis=1)
            color_pred_map = color_pred_maps[i]
            # color_pred_scene = color_pred_map[nearest_pred_idx]
            color_pred_scene = color_pred_map[nearest_10_idx].mean(axis=1)
            pred_value_thershold = 0.3  # for visualization
            pred_value = normalized_pred_feat_np[0, nearest_10_idx, :].mean(axis=1)

        else:
            distance_pred = cdist(points_scene, fps_points_scene)
            nearest_pred_idx = np.argmin(distance_pred, axis=1)
            color_pred_map = color_pred_maps[i]
            color_pred_scene = color_pred_map[nearest_pred_idx]
            pred_value_thershold = 0.3

        color_pred_list.append(color_pred_scene)
        color_img_list.append(image_color)

    for i, pc in enumerate(pcs):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd.colors = o3d.utility.Vector3dVector(
            color_pred_list[i].cpu().numpy() * 1 + color_img_list[i] * 0
        )
        o3d.io.write_point_cloud(f"test_front.ply", pcd)
        #o3d.visualization.draw_geometries([pcd])

def is_red(color, tolerance=0.1):
    return color[0] > 1 - tolerance and color[1] < tolerance and color[2] < tolerance


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
    points = pred["pose_xyR_pred"]  # [1, N, 3]
    guide_cost = pred["guide_losses"]["affordance_loss"]  # [1, N]

    if intrinsics is None:
        intrinsics = np.array(
            [[619.0125, 0.0, 326.525], [0.0, 619.16775, 239.11084], [0.0, 0.0, 1.0]]
        )

    points_scene, idx = backproject(
        depth,
        intrinsics,
        np.logical_and(depth > 0, depth < 2000),
        NOCS_convention=False,
    )
    image_color = image[idx[0], idx[1], :] / 255

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_scene)

    colors = np.zeros((points_scene.shape[0], 3))


    points = points.cpu().numpy()
    guide_cost = guide_cost.cpu().numpy()

    points = points[0]
    guide_cost = guide_cost[0]


    distances = np.sqrt(
        ((points_scene[:, :2][:, None, :] - points[:, :2]) ** 2).sum(axis=2)
    )

    scenepts_to_anchor_dist = np.min(distances, axis=1)  # [num_points]
    scenepts_to_anchor_id = np.argmin(distances, axis=1)  # [num_points]
    topk_points_id = np.argsort(scenepts_to_anchor_dist, axis=0)[: points.shape[0]]
    tokk_points_id_corr_anchor = scenepts_to_anchor_id[topk_points_id]

    guide_cost = guide_cost[tokk_points_id_corr_anchor]
    guide_cost_color = get_heatmap(guide_cost[None])[0]

    colors[topk_points_id] = [1, 0, 0]
    pcd.colors = o3d.utility.Vector3dVector(colors * 0.3 + image_color * 0.7 * 255)


    points_for_place = points_scene[topk_points_id]

    points_for_place_goal = np.mean(points_for_place, axis=0)
    print("points_for_place_goal:", points_for_place_goal)

    vis = [pcd]
    # print("points_for_place_goal:", points_for_place_goal)
    for ii, pos in enumerate(points_for_place):
        pos_vis = o3d.geometry.TriangleMesh.create_sphere()
        pos_vis.compute_vertex_normals()
        pos_vis.scale(0.01, [0, 0, 0])
        pos_vis.translate(pos[:3])
        vis_color = guide_cost_color[ii]
        pos_vis.paint_uniform_color(vis_color)

        vis.append(pos_vis)
    o3d.visualization.draw(vis)


class pred_one_case_dataset(Dataset):
    def __init__(
        self,
        scene_pcd,
        rgb_image_file_path,
        target_name,
        direction_text,
        depth_img_path,
    ):
        self.rgb_image_file_path = rgb_image_file_path
        self.target_name = target_name
        self.direction_text = direction_text
        self.scene_pcd = scene_pcd
        self.scene_pcd_points = np.asarray(self.scene_pcd.points)
        self.scene_pcd_tensor = torch.tensor(
            self.scene_pcd_points, dtype=torch.float32
        ).unsqueeze(0)

        self.scene_pcd_colors = np.asarray(self.scene_pcd.colors)
        green_mask = np.apply_along_axis(is_red, 1, self.scene_pcd_colors)
        green_points = self.scene_pcd_points[green_mask]
        self.green_pcd_center = np.mean(green_points, axis=0)
        self.scene_pcd_tensor = self.scene_pcd_tensor.to("cuda")

        fps_indices_scene = pointnet2_utils.furthest_point_sample(
            self.scene_pcd_tensor.contiguous(), 2048
        )
        fps_indices_scene_np = fps_indices_scene.squeeze(0).cpu().numpy()
        fps_points_scene_from_original = self.scene_pcd_points[fps_indices_scene_np]
        self.fps_points_scene_from_original = fps_points_scene_from_original
        self.fps_colors_scene_from_original = self.scene_pcd_colors[
            fps_indices_scene_np
        ]

        rgb_image = Image.open(rgb_image_file_path).convert("RGB")
        rgb_image = np.asarray(rgb_image).astype(float) / 255.0
        rgb_image = np.transpose(rgb_image, (2, 0, 1))

        self.rgb_image = rgb_image
        self.depth = np.array(cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)).astype(
            float
        ) / 1000.0

    def __len__(self):
        return 2

    def __getitem__(self, index):
        sample = {
            "fps_points_scene": self.fps_points_scene_from_original,
            "fps_colors_scene": self.fps_colors_scene_from_original,
            "pc_position": self.fps_points_scene_from_original,
            "phrase": "no phrase in testing mode",
            "image": self.rgb_image,
            "mask": "no mask in testing mode",
            "file_path": "dataset/scene_RGBD_mask/id000/book_0000_fake",  # fake
            "reference_obj": self.target_name,
            "direction_text": self.direction_text,
            "anchor_position": self.green_pcd_center,
            "depth": self.depth,
        }
        return sample


def rgb_obj_dect(
    image_path,
    text_prompt,
    out_dir=None,
    model_path="GroundingDINO/weights/groundingdino_swint_ogc.pth",
    use_chatgpt=False,
):

    model = load_model(
        "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", model_path
    )
    IMAGE_PATH = image_path
    TEXT_PROMPT = text_prompt
    BOX_TRESHOLD = 0.25 # 0.35
    TEXT_TRESHOLD = 0.25 # 0.25

    image_source, image = load_image(IMAGE_PATH)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
    )

    phrases = [f"{text_prompt} id{id}" for id in range(len(phrases))]

    h, w, _ = image_source.shape
    ori_boxes = boxes * torch.Tensor([w, h, w, h])
    ori_boxes = torch.round(ori_boxes)

    center_x = int(ori_boxes[0][0].item())
    center_y = int(ori_boxes[0][1].item())
    if out_dir is not None:
        # print("orignal boxes cxcy:", ori_boxes, ori_boxes[0][0], ori_boxes[0][1])
        annotated_frame = annotate(
            image_source=image_source, boxes=boxes, logits=logits, phrases=phrases
        )
        write_path = "Geo_comb/annotated.jpg"
        cv2.imwrite(write_path, annotated_frame)


        if len(phrases) > 0:
            if use_chatgpt:
                id_list, direction = chatgpt_select_id(write_path)
                id = int(id_list[0])
                center_x = int(ori_boxes[id][0].item())
                center_y = int(ori_boxes[id][1].item())
            else:
                center_x = int(ori_boxes[0][0].item())
                center_y = int(ori_boxes[0][1].item())
        annotated_frame[:] = 0
        cv2.circle(annotated_frame, (center_x, center_y), 5, (255, 0, 0), -1)
        cv2.imwrite("Geo_comb/annotated_circle.jpg", annotated_frame)
        # cv2.imshow("annotated_frame", annotated_frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    return annotated_frame



def full_pipeline(
        rgb_image_file_path,
        depth_image_file_path,
        obj_to_place_path,
        obj_target_size,
        intrinsics,
        target_name=[],
        direction_text = [],
        use_vlm = False,
        use_gmm = False,
        visualize_affordance = False,
        visualize_diff = False,
        visualize_final_obj = False,
        rendering = False,
        guidance = AffordanceGuidance_v2(),
):
    #1 use chatgpt or directly provide target_name and direction_text
    assert (len(target_name) == len(direction_text)> 0) or use_vlm, "Please provide target_name and direction_text"

    if use_vlm:
        target_name, direction_text = chatgpt_condition(
                    rgb_image_file_path, "object_placement"
                )
        print("====> Using VLM to sparse...")
        target_name = [target_name] 
        direction_text = [direction_text]


    #2 load the mapper and diffuser model
    # mapper model
    model_affordance_cls = registry.get_affordance_model("GeoL_net_v9")
    model_affordance = model_affordance_cls(
        input_shape=(3, 720, 1280),
        target_input_shape=(3, 128, 128),
        intrinsics=INTRINSICS,
    ).to("cuda")
    state_affordance_dict = torch.load("outputs/checkpoints/GeoL_v9_20K_meter_retrain_lr_1e-4_0213/ckpt_11.pth", map_location="cpu")
    model_affordance.load_state_dict(state_affordance_dict["ckpt_dict"])
    
    # diffuser model
    with open("config/baseline/diffusion.yaml", "r") as file:
        yaml_data = yaml.safe_load(file)
    config_diffusion = OmegaConf.create(yaml_data)
    model_diffuser_cls = PoseDiffusionModel
    model_diffuser = model_diffuser_cls(config_diffusion.model).to("cuda")
    state_diffusion_dict = torch.load("outputs/checkpoints/GeoL_diffuser_v0_1K_xyr/ckpt_93.pth", map_location="cpu")
    model_diffuser.load_state_dict(state_diffusion_dict["ckpt_dict"])
    model_diffuser.nets["policy"].set_guidance(guidance)

    #3 use GroundingDINO to detect the target object
    pred_affordance_list = []
    for i in range(len(target_name)):
        # annotated_frame = rgb_obj_dect(
        #     rgb_image_file_path, target_name[i], "exps/pred_one/RGB_ref.jpg", use_chatgpt=False
        # )
        annotated_frame = rgb_obj_dect_use_gpt_select(
            rgb_image_file_path, target_name[i], "exps/pred_one/RGB_ref.jpg")
        color_no_obj = np.array(annotated_frame)
        depth = cv2.imread(depth_image_file_path, cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32) / 1000.0
        points_no_obj_scene, scene_no_obj_idx = backproject(
            depth,
            intrinsics,
            np.logical_and(depth > 0, depth < 2),
            NOCS_convention=False,
        )
        colors_no_obj_scene = color_no_obj[scene_no_obj_idx[0], scene_no_obj_idx[1]]
        pcd_no_obj_scene = visualize_points(points_no_obj_scene, colors_no_obj_scene)  



        # 4 dataset to the mapper
        dataset_mapper = pred_one_case_dataset(
            pcd_no_obj_scene,
            rgb_image_file_path,
            target_name[i],
            direction_text[i],
            depth_image_file_path,
        )
        data_loader_mapper = DataLoader(dataset_mapper, batch_size=1, shuffle=False)

        #5 mapper model prediction and visualization

        model_affordance.eval()
        for i, batch in enumerate(data_loader_mapper):
            for key, val in batch.items():
                if type(val) == list:
                    continue
                batch[key] = val.float().to("cuda")
            with torch.no_grad():
                affordance_pred = model_affordance(batch=batch)["affordance"].squeeze(1)
                affordance_pred_sigmoid = affordance_pred.sigmoid().cpu().numpy()
                # normalize the affordance prediction

                # add the affordance prediction to the batch
                affordance_thershold = 0.00
                fps_points_scene_from_original = batch["fps_points_scene"][0]

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
                break 

        pred_affordance_list.append(affordance_pred) # make the affordance map smoother
        #generate_heatmap_pc(batch, affordance_pred, intrinsics, interpolate=False) # visualize single case
            
        
        # merge the affordance prediction to the scene point cloud
    # merge the affordance prediction: use the GMM or the mean
    pred_affordance_merge = torch.cat(pred_affordance_list, dim=0)
    pred_affordance_merge = (pred_affordance_merge ).mean(dim=0, keepdim=True)
    # normalize the affordance prediction
    pred_affordance_merge = (pred_affordance_merge - pred_affordance_merge.min()) / (pred_affordance_merge.max() - pred_affordance_merge.min()) 
    generate_heatmap_pc(batch, pred_affordance_merge, intrinsics, interpolate=False) # visualize single case

    # 7 normalize the prediction for the diffuser
    pred_affordance_merge = affordance_pred.sigmoid()
    pred_affordance_merge = (pred_affordance_merge - pred_affordance_merge.min()) / (pred_affordance_merge.max() - pred_affordance_merge.min())

    # 8 prepare the data for the diffuser, especially the object mesh
    batch['affordance'] = pred_affordance_merge.to("cuda")
    # obj_mesh = trimesh.load_mesh(obj_to_place_path)
    # obj_pc = obj_mesh.sample(512)
    obj_mesh = o3d.io.read_triangle_mesh(obj_to_place_path)
    obj_mesh.compute_vertex_normals()

    current_size = obj_mesh.get_axis_aligned_bounding_box().get_extent()
    obj_scale = np.array([obj_target_size[0]/ current_size[0], obj_target_size[1]/ current_size[1], obj_target_size[2]/ current_size[2]])
    obj_scale_matrix = np.array([
        [obj_scale[0], 0, 0, 0],
        [0, obj_scale[1], 0, 0],
        [0, 0, obj_scale[2], 0],
        [0,0,0,1]
    ])
    obj_mesh.transform(obj_scale_matrix)
    obj_pc = obj_mesh.sample_points_uniformly(512)
    obj_pc = np.asarray(obj_pc.points)

    #obj_mesh.rotate(obj_rotation_matrix, center=[0, 0, 0])  # rotate obj mesh
    #obj_mesh.rotate(obj_inverse_matrix, center=[0, 0, 0])  # rotate obj mesh
    #obj_mesh.rotate(obj_inverse_matrix, center=[0, 0, 0])  # rotate obj mesh
    #obj_mesh.rotate(T_plane[:3, :3], center=[0, 0, 0])  # rotate obj mesh
    batch['object_pc_position'] = torch.tensor(obj_pc, dtype=torch.float32).unsqueeze(0).to("cuda")
    batch['gt_pose_xy_min_bound'] = torch.tensor(min_bound_affordance[...,:2], dtype=torch.float32).unsqueeze(0).to("cuda")
    batch['gt_pose_xy_max_bound'] = torch.tensor(max_bound_affordance[...,:2], dtype=torch.float32).unsqueeze(0).to("cuda")
    batch['gt_pose_xyR_min_bound'] = torch.tensor(np.delete(min_bound_affordance, obj=2, axis=0), dtype=torch.float32).unsqueeze(0).to("cuda")
    batch['gt_pose_xyR_max_bound'] = torch.tensor(np.delete(max_bound_affordance, obj=2, axis=0), dtype=torch.float32).unsqueeze(0).to("cuda")

    vol_bnds = np.zeros((3, 2))
    view_frust_pts = get_view_frustum(depth, intrinsics, np.eye(4))
    vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
    vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
    vol_bnds[:, 0] = vol_bnds[:, 0].min()
    vol_bnds[:, 1] = vol_bnds[:, 1].max()

    color_tsdf = cv2.cvtColor(color_no_obj, cv2.COLOR_BGR2RGB)
    tsdf = TSDFVolume(vol_bnds, voxel_dim=256, num_margin=5)
    tsdf.integrate(color_tsdf, depth, intrinsics, np.eye(4))
    T_plane, plane_model = get_tf_for_scene_rotation(points_no_obj_scene)
    batch['vol_bnds'] = torch.tensor(vol_bnds, dtype=torch.float32).unsqueeze(0).to("cuda")
    batch['tsdf_vol'] = torch.tensor(tsdf._tsdf_vol, dtype=torch.float32).unsqueeze(0).to("cuda")
    batch["T_plane"] = torch.tensor(T_plane, dtype=torch.float32).unsqueeze(0).to("cuda")
    batch['intrinsics'] = torch.tensor(intrinsics, dtype=torch.float32).unsqueeze(0).to("cuda")
    batch['color_tsdf'] = torch.tensor(color_tsdf, dtype=torch.float32).unsqueeze(0).to("cuda")
    batch['intrinsics'] = torch.tensor(intrinsics, dtype=torch.float32).unsqueeze(0).to("cuda")
  
    # 9 predict the xyR 
    pred = model_diffuser(batch, num_samp=1, class_free_guide_w=-0.1, apply_guidance=False, guide_clean=True)

    # 10 select topk points
    topk = 1

    guide_affordance_loss = pred["guide_losses"]["affordance_loss"].cpu().numpy()
    guide_collision_loss = pred["guide_losses"]["collision_loss"].cpu().numpy()
    guide_distance_error = pred["guide_losses"]["distance_error"].cpu().numpy()
    min_colliion_loss = guide_collision_loss.min()
    guide_composite_loss = guide_affordance_loss
    guide_composite_loss[guide_collision_loss > min_colliion_loss] = np.inf
    min_guide_loss_idx = np.argsort(guide_composite_loss)[0][:topk]
    pred_points = pred['pose_xyR_pred'][0][min_guide_loss_idx]
    print("min distance loss:", guide_distance_error[0][min_guide_loss_idx])
    print("min collision loss:", guide_collision_loss[0][min_guide_loss_idx])
    print("pred xyR:", pred_points) 

    
    if visualize_diff:
        visualize_xy_pred_points(pred, batch, intrinsics=INTRINSICS)
    

    #11 add mesh obj to the scene
    obj_mesh = o3d.io.read_triangle_mesh(obj_to_place_path)
    obj_mesh.compute_vertex_normals()
    current_size = obj_mesh.get_axis_aligned_bounding_box().get_extent()
    obj_scale = np.array([obj_target_size[0]/ current_size[0], obj_target_size[1]/ current_size[1], obj_target_size[2]/ current_size[2]])
    obj_scale_matrix = np.array([
        [obj_scale[0], 0, 0, 0],
        [0, obj_scale[1], 0, 0],
        [0, 0, obj_scale[2], 0],
        [0,0,0,1]
    ])
    obj_mesh.transform(obj_scale_matrix)

    obj_rotation = 0
    obj_rotation_matrix = np.array(
        [
            [1, 0, 0],
            [0, 0.6, 0.8],
            [0, -0.8, 0.6],
        ]
    )
    obj_inverse_matrix = np.array(
        [
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ]
    )
    obj_pcd = obj_mesh.sample_points_uniformly(number_of_points=10000)
    #obj_mesh.rotate(obj_rotation_matrix, center=[0, 0, 0])  # rotate obj mesh
    #obj_mesh.rotate(obj_inverse_matrix, center=[0, 0, 0])  # rotate obj mesh
    obj_mesh.rotate(obj_inverse_matrix, center=[0, 0, 0])  # rotate obj mesh
    obj_mesh.rotate(T_plane[:3, :3], center=[0, 0, 0])  # rotate obj mesh

    #obj_mesh.translate([0, -1, 1])
    #obj_pcd.rotate(obj_rotation_matrix, center=[0, 0, 0])  # rotate obj mesh
    #obj_pcd.translate([0, -1, 1])
    obj_max_bound = obj_pcd.get_max_bound()
    obj_min_bound = obj_pcd.get_min_bound()
    obj_bottom_center = (obj_max_bound + obj_min_bound) / 2
    obj_bottom_center[2] = obj_max_bound[2]  # attention: the z axis is reversed
    #obj_pcd.translate(
        #pred_points[0].cpu().numpy() - obj_bottom_center
    #)  # move obj mesh to target point


    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    rgb_image = cv2.imread(rgb_image_file_path, -1)
    color_scene = np.array(rgb_image)/255.0
    depth = cv2.imread(depth_image_file_path, cv2.IMREAD_UNCHANGED)
    depth = depth.astype(np.float32) / 1000.0
    points_scene, scene_idx = backproject(
        depth,
        intrinsics,
        np.logical_and(depth > 0, depth < 2),
        NOCS_convention=False,
    )
    colors_scene = color_scene[scene_idx[0], scene_idx[1]]
    pcd_scene = visualize_points(points_scene, colors_scene) 

    # use the plane model to determine the z
    pred_xy = pred_points[0][...,:2].cpu().numpy()
    pred_z = (-plane_model[0] * pred_xy[0] - plane_model[1] * pred_xy[1] - plane_model[3]-0.01) / plane_model[2]
    pred_xyz = np.append(pred_xy, pred_z)

    obj_mesh.translate(pred_xyz - obj_bottom_center)  # move obj
    # create sphere
    sphere_target = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    sphere_target.compute_vertex_normals()
    sphere_target.paint_uniform_color([0.1, 0.1, 0.7])
    sphere_target.translate(pred_xyz)

    sphere_obj = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    sphere_obj.compute_vertex_normals()
    sphere_obj.paint_uniform_color([0.1, 0.7, 0.1])
    sphere_obj.translate(obj_bottom_center)
    o3d.visualization.draw_geometries([obj_mesh, pcd_scene, coordinate_frame, sphere_target, sphere_obj])




if __name__ == "__main__":
    # INTRINSICS = np.array([[591.0125 ,   0.     , 322.525  ],[  0.     , 590.16775, 244.11084],[  0.     ,   0.     ,   1.     ]])
    # INTRINSICS = np.array([[591.0125 ,   0.     , 636  ],[  0.     , 590.16775, 367],[  0.     ,   0.     ,   1.     ]])
    # intr = np.array([[591.0125 ,   0.     , 322.525  ],[  0.     , 590.16775, 244.11084],[  0.     ,   0.     ,   1.     ]])
    #INTRINSICS = np.array([[619.0125 ,   0.     , 326.525  ],[  0.     , 619.16775, 239.11084],[  0.     ,   0.     ,   1.     ]]) #realsense
    #INTRINSICS = np.array([[607.0125 ,   0.     , 636.525  ], [  0.     , 607.16775, 367.11084], [  0.     ,   0.     ,   1.     ]]) # kinect
    #INTRINSICS = np.array([[607.09912/2 , 0. , 636.85083/2 ], [0., 607.05212/2, 367.35952/2], [0.0, 0.0, 1.0]])
    INTRINSICS = np.array([[911.09,   0.  , 657.44],[  0.  , 910.68,  346.58],[  0.  ,   0.  ,   1.  ]]) # realsense
    # congiguration
    # scene_pcd_file_path = "dataset/scene_RGBD_mask_direction_mult/id10_1/clock_0001_normal/mask_Behind.ply"
    # blendproc dataset
    #rgb_image_file_path = "dataset/scene_RGBD_mask_v2_kinect_cfg/id110/eye_glasses_0003_black/with_obj/test_pbr/000000/rgb/000000.jpg"
    #depth_image_file_path = "dataset/scene_RGBD_mask_v2_kinect_cfg/id110/eye_glasses_0003_black/with_obj/test_pbr/000000/depth/000000.png"

    # kinect data
    # rgb_image_file_path = "dataset/kinect_dataset/color/000025.png"
    # depth_image_file_path = "dataset/kinect_dataset/depth/000025.png"

    # realsense data
    rgb_image_file_path = "dataset/realworld_2103/color/000002.png"
    depth_image_file_path = "dataset/realworld_2103/depth/000002.png"

    # data from robot camera
    #rgb_image_file_path = "dataset/data_from_robot/img/img_10.jpg"
    #depth_image_file_path = "dataset/data_from_robot/depth/depth_10.png"


    full_pipeline(
        rgb_image_file_path=rgb_image_file_path,
        depth_image_file_path=depth_image_file_path,
        obj_to_place_path="dataset/obj/mesh/keyboard/keyboard_0001_black/mesh.obj",
        obj_target_size = [0.8, 0.2, 0.01], # H W D
        intrinsics=INTRINSICS,
        target_name=["purple screen"],
        direction_text=["Front"],
        use_vlm=False,
        use_gmm=False,
        visualize_affordance=False,
        visualize_diff=True,
        visualize_final_obj=False,
        rendering = False,
        guidance=CompositeGuidance(),
    )