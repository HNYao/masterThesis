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
from scipy.spatial.distance import cdist
from matplotlib import cm
import torchvision.transforms as T
from GeoL_net.gpt.gpt import chatgpt_condition
from GeoL_diffuser.algos.pose_algos import PoseDiffusionModel
from GeoL_diffuser.models.guidance import OneGoalGuidance, AffordanceGuidance
import yaml
from omegaconf import OmegaConf
import trimesh

# INTRINSICS = np.array([[591.0125 ,   0.     , 322.525  ],[  0.     , 590.16775, 244.11084],[  0.     ,   0.     ,   1.     ]])
# INTRINSICS = np.array([[591.0125 ,   0.     , 636  ],[  0.     , 590.16775, 367],[  0.     ,   0.     ,   1.     ]])
# intr = np.array([[591.0125 ,   0.     , 322.525  ],[  0.     , 590.16775, 244.11084],[  0.     ,   0.     ,   1.     ]])
# INTRINSICS = np.array([[619.0125 ,   0.     , 326.525  ],[  0.     , 619.16775, 239.11084],[  0.     ,   0.     ,   1.     ]]) #realsense
# INTRINSICS = np.array([[607.0125 ,   0.     , 636.525  ], [  0.     , 607.16775, 367.11084], [  0.     ,   0.     ,   1.     ]]) # kinect
INTRINSICS = np.array([[607.09912/2 , 0. , 636.85083/2 ], [0., 607.05212/2, 367.35952/2], [0.0, 0.0, 1.0]])


def get_heatmap(values, cmap_name="turbo", invert=False):
    if invert:
        values = -values
    values = (values - values.min()) / (values.max() - values.min())
    colormaps = cm.get_cmap(cmap_name)
    rgb = colormaps(values)[..., :3]  # don't need alpha channel
    return rgb

def create_sphere_at_points(center, radius=0.05, color=[1, 0, 0]):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color(color)
    sphere.translate(center)
    return sphere

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
            np.logical_and(depth > 0, depth > 0),
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
            np.logical_and(depth > 0, depth < 2500),
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
            color_pred_list[i].cpu().numpy() * 0.3 + color_img_list[i] * 0.7
        )
        # o3d.io.write_point_cloud(f"test_front.ply", pcd)
        o3d.visualization.draw_geometries([pcd])


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
    depth = batch["depth"][0].cpu().numpy() / 1000.0
    image = batch["image"][0].permute(1, 2, 0).cpu().numpy()
    points = pred["pose_xyz_pred"]  # [1, N*H, 3] descaled
    guide_cost = pred["guide_losses"]["affordance_loss"]  # [1, N*H]
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
    image_color = image[idx[0], idx[1], :] / 255

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_scene)

    #colors = np.zeros((points_scene.shape[0], 3))

    points = points.cpu().numpy()
    guide_cost = guide_cost.cpu().numpy()

    points = points[0] # [N*H, 3]
    guide_cost = guide_cost[0] # [N*H]
    # guide_cost_color = cm.get_cmap("turbo")(guide_cost[None])[0, :, :3]

    distances = np.sqrt(
        ((points_scene[:, :2][:, None, :] - points[:, :2]) ** 2).sum(axis=2)
    ) # x, y distance   

    # is_near_pose = np.any(distances < distance_thershold, axis=1)
    scenepts_to_anchor_dist = np.min(distances, axis=1)  # [num_points]
    scenepts_to_anchor_id = np.argmin(distances, axis=1)  # [num_points]
    topk_points_id = np.argsort(scenepts_to_anchor_dist, axis=0)[: points.shape[0]]
    tokk_points_id_corr_anchor = scenepts_to_anchor_id[topk_points_id]

    guide_cost = guide_cost[tokk_points_id_corr_anchor]
    guide_cost_color = get_heatmap(guide_cost[None], invert=False)[0]
    #guide_cost_color = np.random.rand(800, 3)

    pcd.colors = o3d.utility.Vector3dVector(image_color)


    points_for_place = points_scene[topk_points_id]

    points_for_place_goal = np.mean(points_for_place, axis=0)
    print("points_for_place_goal:", points_for_place_goal)

    # get the topk affordance points avg position and visualize
    affordance = batch["affordance"] # [B, 2048, 1]
    position = batch["pc_position"] # [B, 2048, 3]
    affordance = affordance.squeeze(-1) # [B, 2048]
    k = 10
    topk_affordance, topk_idx = torch.topk(affordance, k, dim=-1)
    topk_positions = torch.gather(position, dim=1, index=topk_idx.unsqueeze(-1).expand(-1, -1, position.size(-1)))
    avg_topk_positions = topk_positions.mean(dim=1)  # (B, 3)
    avg_topk_positions = avg_topk_positions[0].cpu().numpy()
    avg_topk_mean_sphere = create_sphere_at_points(avg_topk_positions, radius=0.05, color=[1, 0, 0])
    

    vis = [pcd, avg_topk_mean_sphere]

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
    #o3d.io.write_point_cloud("outputs/model_output/test_diffusion/w=0.ply", pcd)


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
        rgb_image = np.asarray(rgb_image).astype(float)
        rgb_image = np.transpose(rgb_image, (2, 0, 1))

        self.rgb_image = rgb_image
        self.depth = np.array(cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)).astype(
            float
        )

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
):

    model = load_model(
        "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", model_path
    )
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
        text_threshold=TEXT_TRESHOLD,
    )

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
        annotated_frame[:] = 0
        cv2.circle(annotated_frame, (center_x, center_y), 5, (255, 0, 0), -1)
        cv2.imwrite(out_dir, annotated_frame)

    return annotated_frame


if __name__ == "__main__":

    # congiguration
    # scene_pcd_file_path = "dataset/scene_RGBD_mask_direction_mult/id10_1/clock_0001_normal/mask_Behind.ply"
    # blendproc dataset
    rgb_image_file_path = "dataset/scene_gen/scene_RGBD_mask_data_aug_test/id108_id96_0_0/bowl_0001_wooden/with_obj/test_pbr/000000/rgb/000000.jpg"
    depth_image_file_path = "dataset/scene_gen/scene_RGBD_mask_data_aug_test/id108_id96_0_0/bowl_0001_wooden/with_obj/test_pbr/000000/depth_noise/000000.png"

    # kinect data
    # rgb_image_file_path = "dataset/kinect_dataset/color/000025.png"
    # depth_image_file_path = "dataset/kinect_dataset/depth/000025.png"

    # realsense data
    # rgb_image_file_path = "dataset/realsense/color/000098.png"
    # depth_image_file_path = "dataset/realsense/depth/000098.png"

    use_chatgpt = False
    if use_chatgpt:
        target_name, direction_text = chatgpt_condition(
            rgb_image_file_path, "object_placement"
        )
        print("====> Predicting Affordance...")
    else:
        target_name = "the blue book"
        direction_text = "Right"

    # use GroundingDINO to detect the target object
    annotated_frame = cv2.imread(rgb_image_file_path)
    color_no_obj = np.array(annotated_frame)
    depth = cv2.imread(depth_image_file_path, cv2.IMREAD_UNCHANGED)
    depth = depth.astype(np.float32) / 1000

    intr = INTRINSICS
    points_no_obj_scene, scene_no_obj_idx = backproject(
        depth,
        intr,
        np.logical_and(depth > 0, depth < 2),
        NOCS_convention=False,
    )
    colors_no_obj_scene = color_no_obj[scene_no_obj_idx[0], scene_no_obj_idx[1]]
    pcd_no_obj_scene = visualize_points(points_no_obj_scene, colors_no_obj_scene)

    with open("config/baseline/diffusion.yaml", "r") as file:
        yaml_data = yaml.safe_load(file)

    config_diffusion = OmegaConf.create(yaml_data)

    model_diffuser_cls = PoseDiffusionModel
    model_diffuser = model_diffuser_cls(config_diffusion.model).to("cuda")
    #NOTE: wait the training to finish
    state_diffusion_dict = torch.load(
        "outputs/checkpoints/GeoL_diffuser_rand_afford/ckpt_5.pth", map_location="cpu"
    )
    model_diffuser.load_state_dict(state_diffusion_dict["ckpt_dict"])
    guidance = AffordanceGuidance()
    model_diffuser.nets["policy"].set_guidance(guidance)

    # create the dataset
    dataset = pred_one_case_dataset(
        pcd_no_obj_scene,
        rgb_image_file_path,
        target_name,
        direction_text,
        depth_image_file_path,
    )
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for i, batch in enumerate(data_loader):
        for key, val in batch.items():
            if type(val) == list:
                continue
            batch[key] = val.float().to("cuda")
        afford_pred_dict = np.load("Geo_comb/afford_pred.npz", allow_pickle=True)
        with torch.no_grad():
            affordance_pred = torch.tensor(afford_pred_dict["affordance"]).to(
                "cuda"
            )
            fps_points_scene_affordance = afford_pred_dict[
                "pc_position"
            ]
            min_bound_affordance = afford_pred_dict["min_bound_affordance"]
            max_bound_affordance = afford_pred_dict["max_bound_affordance"]
            pc_position_xy_affordance = afford_pred_dict["pc_position_xy_affordance"]

            # update dataset
            batch["affordance"] = affordance_pred
            batch["object_name"] = ["the green bottle"]
            rgb_image = Image.open(rgb_image_file_path).convert("RGB")
            rgb_image = np.asarray(rgb_image).astype(float)
            rgb_image = np.transpose(rgb_image, (2, 0, 1))
            rgb_image = rgb_image / 255
            assert rgb_image.max() <= 1.0 and rgb_image.min() >= 0.0

            depth = cv2.imread(depth_image_file_path, cv2.IMREAD_UNCHANGED)
            depth = depth.astype(np.float32) / 1000
            depth[depth > 2] = 0
            scene_depth_cloud, _ = backproject(depth, INTRINSICS, np.logical_and(depth > 0, depth < 2))
            scene_depth_cloud = visualize_points(scene_depth_cloud)
            scene_depth_cloud_points = np.asarray(scene_depth_cloud.points)
            max_xyz_bound = np.max(scene_depth_cloud_points, axis=0)
            min_xyz_bound = np.min(scene_depth_cloud_points, axis=0)
            min_xyz_bound[2] = 1.0
            max_xyz_bound[2] = 1.0

            obj_mesh = trimesh.load(
                "dataset/obj/mesh/bottle/bottle_0003_green/mesh.obj"
            )
            obj_scale = [0.4554532861750685, 0.4554532861750685, 0.4554532861750685]
            obj_mesh.apply_scale(obj_scale)
            obj_pc = obj_mesh.sample(512)

            batch["object_pc_position"] = (
                torch.tensor(obj_pc, dtype=torch.float32).unsqueeze(0).to("cuda")
            )
            batch["pc_position_xy_affordance"] = (
                torch.tensor(pc_position_xy_affordance[:, :2], dtype=torch.float32)
                .unsqueeze(0)
                .to("cuda")
            )
            batch["gt_pose_xyz_min_bound"] = (
                torch.tensor(
                    min_xyz_bound, dtype=torch.float32
                )
                .unsqueeze(0)
                .to("cuda")
            )
            batch["gt_pose_xyz_max_bound"] = (
                torch.tensor(
                    max_xyz_bound, dtype=torch.float32
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

            # pred pose
            pred = model_diffuser(batch, num_samp=10, class_free_guide_w=-0.1, apply_guidance=True, guide_clean=True)
            print("pred:", pred)
            visualize_xy_pred_points(pred, batch, intrinsics=INTRINSICS)

            break
