import torch
import torch.nn as nn
from matplotlib import cm
import cv2
import torchvision
import pytorch3d

#print("torch:", torch.__version__)  # 2.0.0
#print(torch.version.cuda)  # 11.7 cudatoolkit 11.8.0 pytorch-cuda 11.7
#print("torchvision:", torchvision.__version__)  # 0.15.0
#print("pytorch3d:", pytorch3d.__version__)  # 0.7.5
import open3d as o3d

#print("open3d:", o3d.__version__)  # 0.18.0
import matplotlib.pyplot as plt
from GeoL_net.models.clip_unet import CLIPUNet
from GeoL_net.models.geo_net import GeoAffordModule, FusionPointLayer
import torch.nn.functional as F
from GeoL_net.models.modules import FeatureConcat, Project3D, ProjectColorOntoImage_v2
from GeoL_net.core.registry import registry
import numpy as np
from PIL import Image
from clip.model import build_model, tokenize, load_clip
import torchvision.transforms as T
from GeoL_net.models.attention_module import SelfAttentionBlock, CrossAttentionLayer
from GeoL_net.trainer.losses import BinaryCELoss
from GeoL_net.models.cdm import ContactMLP

# TODO: GeoL_net_v2 use the almost same structure as GeoL_net, but with 2 encoders for object name and direction


@registry.register_affordance_model(name="GeoL_net")
class GeoL_net(nn.Module):
    def __init__(self, input_shape, target_input_shape, intrinsics=None):
        super().__init__()
        self.geoafford_module = GeoAffordModule(feat_dim=16)  # 0.3kw

        self.clipunet_module = CLIPUNet(
            input_shape=input_shape,
            target_input_shape=target_input_shape,
            output_dim=16,
        )  # 6kw parameters

        self.concate = FeatureConcat()
        self.device = "cuda"  # cpu for dataset
        self.lang_fusion_type = "mult"  # hard code from CLIPlingunet
        self._load_clip()

        self.fusion_point_moudule = FusionPointLayer()  # 0.3kw

        if intrinsics is None:
            self.intrinsics = np.array(
                [[591.0125, 0.0, 322.525], [0.0, 590.16775, 244.11084], [0.0, 0.0, 1.0]]
            )
        else:
            self.intrinsics = intrinsics

    def _load_clip(self):
        model, _ = load_clip("RN50", device=self.device)
        self.clip_rn50 = build_model(model.state_dict()).to(self.device)  # 10kw frozen
        del model
        # Frozen clip
        for param in self.clip_rn50.parameters():
            param.requires_grad = False

    def encode_text(self, x):
        with torch.no_grad():
            tokens = tokenize(x).to(self.device)
            text_feat, text_emb = self.clip_rn50.encode_text_with_embeddings(tokens)

        text_mask = torch.where(tokens == 0, tokens, 1)  # [1, max_token_len]
        return text_feat, text_emb, text_mask

    def forward(self, **kwargs):
        self.batch = kwargs["batch"]
        batch = kwargs["batch"]
        self.phrase = batch["phrase"]
        self.file_name = batch["file_path"]
        texts = batch["phrase"]
        output = {}
        # print(texts)

        # encode text
        l_enc, l_emb, l_mask = self.encode_text(texts)
        l_input = l_emb if "word" in self.lang_fusion_type else l_enc
        l_input = l_input.to(dtype=torch.float32)
        batch["target_query"] = l_input

        # point cloud
        scene_pcs = batch["fps_points_scene"]
        obj_pcs = batch["fps_points_scene"]

        # pointnet_1: extract the geometric affordance feat.
        x_geo = self.geoafford_module(scene_pcs, obj_pcs)  # [B, C, Num_pts]
        # print("----x_geo shape:", x_geo.shape)

        # CLIP: extract the rgb feat.
        x_rgb = self.clipunet_module(batch=batch)  # [B, C_rgbfeat, H, W]
        # print("----x_rgb shape:", x_rgb["affordance"].shape)
        # print("-----scene_pcs shape:", scene_pcs.shape) # [B, Num_puts, 3]

        # merge
        x_align = self.concate(x_rgb["affordance"], scene_pcs, self.intrinsics).permute(
            0, 2, 1
        )  # [B, Num_pts, C_rgbfeat]
        # print("-----align shape:", x_align.shape)
        x = torch.cat((scene_pcs.permute(0, 2, 1), x_align, x_geo), dim=1)  # []

        x = self.fusion_point_moudule(x.permute(0, 1, 2))
        x = x.permute(0, 2, 1)
        # print("------fusion x shape:", x.shape)
        # print("------target shape:", batch["mask"].shape)
        output["affordance"] = x
        self.model_ouput = x
        self.model_pc = scene_pcs
        # print(x)

        return output

    def generate_heatmap(self, epoch):
        "generate heatmap during training"
        self.pc_heatmap_list = []
        pcs = self.model_pc  # [B, Num_points, 3]
        feat = self.model_ouput  # [B, Num_points, 1]
        # Normalize feat to [0, 1] for each batch independently
        min_feat = feat.min(dim=1, keepdim=True)[0]  # Find min in each batch
        max_feat = feat.max(dim=1, keepdim=True)[0]  # Find max in each batch
        normalized_feat = (feat - min_feat) / (
            max_feat - min_feat + 1e-6
        )  # Avoid division by zero
        # Convert normalized features to RGB using turbo colormap
        turbo_colormap = cm.get_cmap(
            "turbo", 256
        )  # Load turbo colormap with 256 levels
        normalized_feat_np = (
            normalized_feat.squeeze(-1).cpu().detach().numpy()
        )  # Shape: [B, Num_points]
        # Apply turbo colormap to normalized features (converts values to RGB)
        color_maps = turbo_colormap(normalized_feat_np)[
            :, :, :3
        ]  # Shape: [B, Num_points, 3], ignore alpha
        # Generate point cloud with Open3D
        for i in range(pcs.shape[0]):  # Iterate over batch
            points = pcs[i].cpu().numpy()  # Shape: [Num_points, 3]
            colors = color_maps[i]  # Shape: [Num_points, 3]

            # Create Open3D point cloud object
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            scene_id = self.file_name[i].split("/")[-2]
            obj_id = self.file_name[i].split("/")[-1]

            # Save the point cloud to a file (you can change the path format)
            # 节省空间，不保存
            # o3d.io.write_point_cloud(f"outputs/model_output/point_cloud/{scene_id}-{obj_id}-{epoch}_heatmap.ply", pcd)
            self.pc_heatmap_list.append(pcd)

    def inference_4cls(self, epoch):
        "Check the model outputs during training"
        pcs = self.model_pc  # [B, Num_points, 3]
        feat = self.model_ouput.permute(0, 2, 1)  # [B, Num_points, 4]

        color_map = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]).cpu()
        predicted_color = torch.argmax(feat, dim=-1).cpu()
        colors = color_map[predicted_color]

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pcs[0].cpu().numpy())
        point_cloud.colors = o3d.utility.Vector3dVector(colors[0].cpu().numpy())
        scene_id = self.file_name[0].split("/")[-2]
        obj_id = self.file_name[0].split("/")[-1]
        o3d.io.write_point_cloud(
            f"outputs/model_output/point_cloud/{scene_id}-{obj_id}-{epoch}.ply",
            point_cloud,
        )

    def inference_2cls(self):
        "Check the model outputs during training"
        pcs = self.model_pc  # [B, Num_points, 3]
        feat = self.model_ouput.permute(0, 2, 1)  # [B, Num_points, 2]

        color_map = torch.tensor([[0, 0, 0], [0, 1, 0]]).cpu()
        predicted_color = torch.argmax(feat, dim=-1).cpu()
        colors = color_map[predicted_color]

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pcs[0].cpu().numpy())
        point_cloud.colors = o3d.utility.Vector3dVector(colors[0].cpu().numpy())
        # o3d.visualization.draw([point_cloud])
        o3d.io.write_point_cloud("outputs/point_cloud.ply_2cls", point_cloud)

    def inference_heatmap_4cls(self, epoch):
        "conver the esimating value to self-dpsefined heatmap"
        pcs = self.model_pc  # [B, Num_points, 3]
        feat = self.model_ouput.permute(0, 2, 1)  # [B, Num_points, 4]
        class_1_feat = feat[:, :, 1].cpu()  # [B, Num_points]

        max_value = torch.max(class_1_feat)
        min_value = torch.min(class_1_feat)

        # normalization
        normalized_class_1_feat = (class_1_feat - min_value) / (max_value - min_value)
        # normalized_class_1_feat = class_1_feat
        flattened = normalized_class_1_feat.detach().numpy().flatten()
        cmap = plt.get_cmap("turbo")
        cmap = plt.get_cmap("viridis")
        color_mapped = cmap(flattened)[:, :3]  # 获取 RGB 值，忽略 alpha 通道

        # back to [B, Num_points, 3]
        colors = color_mapped.reshape(
            normalized_class_1_feat.shape[0], normalized_class_1_feat.shape[1], 3
        )

        # color type  float64，Open3D
        colors = colors.astype(np.float64)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pcs[0].cpu().numpy())
        point_cloud.colors = o3d.utility.Vector3dVector(colors[0])
        self.pc_heatmap = point_cloud
        scene_id = self.file_name[0].split("/")[-2]
        obj_id = self.file_name[0].split("/")[-1]
        o3d.io.write_point_cloud(
            f"outputs/model_output/point_cloud/{scene_id}-{obj_id}-{epoch}_heatmap.ply",
            point_cloud,
        )

    def color_backproj(self, pc_ori=None):
        "bactproject the color from the fps-points to the whole point cloud"
        cam_rotation_matrix = np.array([[1, 0, 0], [0, 0.8, -0.6], [0, 0.6, 0.8]])

        ply1 = o3d.io.read_point_cloud(
            "dataset/scene_RGBD_mask/id162_1/lamp_0004_orange/pc_ori.ply"
        )
        ply2 = self.pc_heatmap
        points_ply1 = np.asarray(ply1.points)
        points_ply1 = (np.linalg.inv(cam_rotation_matrix) @ points_ply1.T).T
        points_ply2 = np.asarray(ply2.points)
        colors_ply2 = np.asarray(ply2.colors)

        # KD tree ply2 ply1
        ply2_tree = o3d.geometry.KDTreeFlann(ply2)
        new_colors = np.zeros_like(points_ply1)

        for i, point in enumerate(points_ply1):
            [_, idx, dist] = ply2_tree.search_knn_vector_3d(point, 10)  # nearest 10
            nearest_colors = colors_ply2[idx]
            distances = np.array(dist)
            weights = 1 / (distances + 1e-6)
            weighted_avg_color = np.average(
                nearest_colors, axis=0, weights=weights
            )  # weighted avg
            new_colors[i] = weighted_avg_color

        ply1.colors = o3d.utility.Vector3dVector(new_colors)
        ply1.points = o3d.utility.Vector3dVector(points_ply1)

        # 保存结果
        o3d.io.write_point_cloud("outputs/point_cloud_heatmap_4cls_whole.ply", ply1)

    def pcdheatmap2img(self, is_mask=False):
        pcs_heatmap_list = self.pc_heatmap_list
        img_rgb_list = self.batch["image"]
        img_heatmap_list = []
        assert len(pcs_heatmap_list) == len(img_rgb_list)
        cam_rotation_matrix = np.array([[1, 0, 0], [0, 0.8, -0.6], [0, 0.6, 0.8]])
        intrinsics = np.array(
            [[591.0125, 0.0, 322.525], [0.0, 590.16775, 244.11084], [0.0, 0.0, 1.0]]
        )
        for i in range(len(pcs_heatmap_list)):
            pc = pcs_heatmap_list[i]
            img_rgb = img_rgb_list[i]
            image_np = img_rgb.cpu().detach().numpy().astype(np.float32)

            image_tensor = (
                torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0)
            )  # [B, C, H, W]

            #

            pc_points = np.array(pc.points)

            # if is_mask:
            pc_points = pc_points @ cam_rotation_matrix

            query_points = torch.tensor(pc_points, dtype=torch.float32).unsqueeze(
                0
            )  # [B, N, 3]

            pc_colors = np.array(pc.colors)
            query_colors = torch.tensor(pc_colors, dtype=torch.float32).unsqueeze(
                0
            )  # [B, N, 3]
            sample_ids = np.random.choice(np.arange(query_colors.size(1)), 2048)

            query_colors = query_colors[:, sample_ids]
            query_points = query_points[:, sample_ids]
            image_tensor = image_tensor.permute(0, 2, 3, 1)
            projector = ProjectColorOntoImage_v2()
            output_image = projector(
                image_grid=image_tensor,
                query_points=query_points,
                query_colors=query_colors,
                intrinsics=intrinsics,
            )
            # save_path = "exps/heatmap_backalign.png"

            for i, img in enumerate(output_image):
                if img.ndim == 3:

                    color_image = T.ToPILImage()(image_tensor[i].cpu())
                    pil_image = T.ToPILImage()(img.cpu())

                    image_np = np.clip(pil_image, 0, 255)
                    # color_image_np = np.array(color_image)
                    color_image_np = np.floor(color_image)
                    color_image_np = np.clip(color_image_np, 0, 255)
                    color_image_np = np.uint8(color_image_np)
                    # image_np = cv2.blur(image_np, (50, 50))

                    image_np = cv2.addWeighted(image_np, 0.3, color_image_np, 0.7, 0.0)
                    pil_image = Image.fromarray(np.uint8(image_np))
                    # pil_image = Image.fromarray(np.float32(image_np))
                    # pil_image.save(save_path)
                    # print(f"img is saved {save_path}")
                    img_heatmap_list.append(pil_image)
        return img_heatmap_list, self.file_name, self.phrase


@registry.register_affordance_model(name="GeoL_net_v2")
class GeoL_net_v2(nn.Module):
    def __init__(self, input_shape, target_input_shape):
        super().__init__()
        self.geoafford_module = GeoAffordModule(feat_dim=16)  # 0.3kw
        self.clipunet_module = CLIPUNet(
            input_shape=input_shape,
            target_input_shape=target_input_shape,
            output_dim=16,
        )  # 6kw parameters
        self.concate = FeatureConcat()
        self.attention = AttentionLayer(C_rgbfeat=16, direction_dim=1024)
        self.device = "cuda"  # cpu for dataset
        self.lang_fusion_type = "mult"  # hard code from CLIPlingunet
        self._load_clip()
        self.instrics = np.array(
            [[591.0125, 0.0, 322.525], [0.0, 590.16775, 244.11084], [0.0, 0.0, 1.0]]
        )
        self.fusion_point_moudule = FusionPointLayer()  # 0.3kw

    def _load_clip(self):
        model, _ = load_clip("RN50", device=self.device)
        self.clip_rn50 = build_model(model.state_dict()).to(self.device)  # 10kw frozen
        del model
        # Frozen clip
        for param in self.clip_rn50.parameters():
            param.requires_grad = False

    def encode_text(self, x):
        with torch.no_grad():
            tokens = tokenize(x).to(self.device)
            text_feat, text_emb = self.clip_rn50.encode_text_with_embeddings(tokens)

        text_mask = torch.where(tokens == 0, tokens, 1)  # [1, max_token_len]
        return text_feat, text_emb, text_mask

    def encode_direction(self, x):
        with torch.no_grad():
            tokens = tokenize(x).to(self.device)
            text_feat, text_emb = self.clip_rn50.encode_text_with_embeddings(tokens)
        text_mask = torch.where(tokens == 0, tokens, 1)  # [1, max_token_len]
        return text_feat, text_emb, text_mask

    def forward(self, **kwargs):
        self.batch = kwargs["batch"]
        batch = kwargs["batch"]
        self.phrase = batch["phrase"]
        self.file_name = batch["file_path"]
        texts = batch["phrase"]
        reference_obj_name = batch["reference_obj"]
        direction_text = batch["direction_text"]
        output = {}
        # print(texts)

        # encode text(object name)
        l_enc, l_emb, l_mask = self.encode_text(reference_obj_name)
        l_input = l_emb if "word" in self.lang_fusion_type else l_enc
        l_input = l_input.to(dtype=torch.float32)
        batch["target_query"] = l_input

        # encode text(direction)
        d_enc, d_emb, d_mask = self.encode_direction(direction_text)
        d_input = d_emb if "word" in self.lang_fusion_type else d_enc
        d_input = d_input.to(dtype=torch.float32)
        batch["direction_query"] = d_input

        # point cloud
        scene_pcs = batch["fps_points_scene"]
        obj_pcs = batch["fps_points_scene"]

        # pointnet_1: extract the geometric affordance feat.
        x_geo = self.geoafford_module(
            scene_pcs, obj_pcs
        )  # [B, C, Num_pts] [B, 4, Num_pts]

        # CLIP: extract the rgb feat.
        x_rgb = self.clipunet_module(batch=batch)  # [B, C_rgbfeat, H, W] [B, 16, H, W]

        # additional: merge the direction feat through transformer
        x_rgb["affordance"] = self.attention(x_rgb["affordance"], d_input)

        # merge
        x_align = self.concate(x_rgb["affordance"], scene_pcs, self.instrics).permute(
            0, 2, 1
        )  # [B, Num_pts, C_rgbfeat]
        x = torch.cat((scene_pcs.permute(0, 2, 1), x_align, x_geo), dim=1)  # []
        x = self.fusion_point_moudule(x.permute(0, 1, 2))
        x = x.permute(0, 2, 1)

        output["affordance"] = x
        self.model_ouput = x
        self.model_pc = scene_pcs

        return output

    def generate_heatmap(self, epoch):
        "generate heatmap during training"
        self.pc_heatmap_list = []
        pcs = self.model_pc  # [B, Num_points, 3]
        feat = self.model_ouput  # [B, Num_points, 1]
        # Normalize feat to [0, 1] for each batch independently
        min_feat = feat.min(dim=1, keepdim=True)[0]  # Find min in each batch
        max_feat = feat.max(dim=1, keepdim=True)[0]  # Find max in each batch
        normalized_feat = (feat - min_feat) / (
            max_feat - min_feat + 1e-6
        )  # Avoid division by zero
        # Convert normalized features to RGB using turbo colormap
        turbo_colormap = cm.get_cmap(
            "turbo", 256
        )  # Load turbo colormap with 256 levels
        normalized_feat_np = (
            normalized_feat.squeeze(-1).cpu().detach().numpy()
        )  # Shape: [B, Num_points]
        # Apply turbo colormap to normalized features (converts values to RGB)
        color_maps = turbo_colormap(normalized_feat_np)[
            :, :, :3
        ]  # Shape: [B, Num_points, 3], ignore alpha
        # Generate point cloud with Open3D
        for i in range(pcs.shape[0]):  # Iterate over batch
            points = pcs[i].cpu().numpy()  # Shape: [Num_points, 3]
            colors = color_maps[i]  # Shape: [Num_points, 3]

            # Create Open3D point cloud object
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            # Save the point cloud to a file (you can change the path format)
            # o3d.io.write_point_cloud(f"outputs/model_output/point_cloud/{scene_id}-{obj_id}-{epoch}_heatmap.ply", pcd)
            self.pc_heatmap_list.append(pcd)

    def inference_4cls(self, epoch):
        "Check the model outputs during training"
        pcs = self.model_pc  # [B, Num_points, 3]
        feat = self.model_ouput.permute(0, 2, 1)  # [B, Num_points, 4]

        color_map = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]).cpu()
        predicted_color = torch.argmax(feat, dim=-1).cpu()
        colors = color_map[predicted_color]

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pcs[0].cpu().numpy())
        point_cloud.colors = o3d.utility.Vector3dVector(colors[0].cpu().numpy())
        scene_id = self.file_name[0].split("/")[-2]
        obj_id = self.file_name[0].split("/")[-1]
        o3d.io.write_point_cloud(
            f"outputs/model_output/point_cloud/{scene_id}-{obj_id}-{epoch}.ply",
            point_cloud,
        )

    def inference_2cls(self):
        "Check the model outputs during training"
        pcs = self.model_pc  # [B, Num_points, 3]
        feat = self.model_ouput.permute(0, 2, 1)  # [B, Num_points, 2]

        color_map = torch.tensor([[0, 0, 0], [0, 1, 0]]).cpu()
        predicted_color = torch.argmax(feat, dim=-1).cpu()
        colors = color_map[predicted_color]

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pcs[0].cpu().numpy())
        point_cloud.colors = o3d.utility.Vector3dVector(colors[0].cpu().numpy())
        # o3d.visualization.draw([point_cloud])
        o3d.io.write_point_cloud("outputs/point_cloud.ply_2cls", point_cloud)

    def inference_heatmap_4cls(self, epoch):
        "conver the esimating value to self-dpsefined heatmap"
        pcs = self.model_pc  # [B, Num_points, 3]
        feat = self.model_ouput.permute(0, 2, 1)  # [B, Num_points, 4]
        class_1_feat = feat[:, :, 1].cpu()  # [B, Num_points]

        max_value = torch.max(class_1_feat)
        min_value = torch.min(class_1_feat)

        # normalization
        normalized_class_1_feat = (class_1_feat - min_value) / (max_value - min_value)
        # normalized_class_1_feat = class_1_feat
        flattened = normalized_class_1_feat.detach().numpy().flatten()
        cmap = plt.get_cmap("turbo")
        cmap = plt.get_cmap("viridis")
        color_mapped = cmap(flattened)[:, :3]  # 获取 RGB 值，忽略 alpha 通道

        # back to [B, Num_points, 3]
        colors = color_mapped.reshape(
            normalized_class_1_feat.shape[0], normalized_class_1_feat.shape[1], 3
        )

        # color type  float64，Open3D
        colors = colors.astype(np.float64)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pcs[0].cpu().numpy())
        point_cloud.colors = o3d.utility.Vector3dVector(colors[0])
        self.pc_heatmap = point_cloud
        scene_id = self.file_name[0].split("/")[-2]
        obj_id = self.file_name[0].split("/")[-1]
        o3d.io.write_point_cloud(
            f"outputs/model_output/point_cloud/{scene_id}-{obj_id}-{epoch}_heatmap.ply",
            point_cloud,
        )

    def color_backproj(self, pc_ori=None):
        "bactproject the color from the fps-points to the whole point cloud"
        cam_rotation_matrix = np.array([[1, 0, 0], [0, 0.8, -0.6], [0, 0.6, 0.8]])

        ply1 = o3d.io.read_point_cloud(
            "dataset/scene_RGBD_mask/id162_1/lamp_0004_orange/pc_ori.ply"
        )
        ply2 = self.pc_heatmap
        points_ply1 = np.asarray(ply1.points)
        points_ply1 = (np.linalg.inv(cam_rotation_matrix) @ points_ply1.T).T
        points_ply2 = np.asarray(ply2.points)
        colors_ply2 = np.asarray(ply2.colors)

        # KD tree ply2 ply1
        ply2_tree = o3d.geometry.KDTreeFlann(ply2)
        new_colors = np.zeros_like(points_ply1)

        for i, point in enumerate(points_ply1):
            [_, idx, dist] = ply2_tree.search_knn_vector_3d(point, 10)  # nearest 10
            nearest_colors = colors_ply2[idx]
            distances = np.array(dist)
            weights = 1 / (distances + 1e-6)
            weighted_avg_color = np.average(
                nearest_colors, axis=0, weights=weights
            )  # weighted avg
            new_colors[i] = weighted_avg_color

        ply1.colors = o3d.utility.Vector3dVector(new_colors)
        ply1.points = o3d.utility.Vector3dVector(points_ply1)

        o3d.io.write_point_cloud("outputs/point_cloud_heatmap_4cls_whole.ply", ply1)

    def pcdheatmap2img(self, is_mask=False):
        pcs_heatmap_list = self.pc_heatmap_list
        img_rgb_list = self.batch["image"]
        img_heatmap_list = []
        assert len(pcs_heatmap_list) == len(img_rgb_list)
        cam_rotation_matrix = np.array([[1, 0, 0], [0, 0.8, -0.6], [0, 0.6, 0.8]])
        intrinsics = np.array(
            [[591.0125, 0.0, 322.525], [0.0, 590.16775, 244.11084], [0.0, 0.0, 1.0]]
        )
        for i in range(len(pcs_heatmap_list)):
            pc = pcs_heatmap_list[i]
            img_rgb = img_rgb_list[i]
            image_np = img_rgb.cpu().detach().numpy().astype(np.float32)

            image_tensor = (
                torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0)
            )  # [B, C, H, W]

            #

            pc_points = np.array(pc.points)

            if is_mask:
                pc_points = pc_points @ cam_rotation_matrix
            # pc_points = pc_points @ cam_rotation_matrix # if mask

            # pc_points = pc_points @ np.linalg.inv(cam_rotation_matrix)

            query_points = torch.tensor(pc_points, dtype=torch.float32).unsqueeze(
                0
            )  # [B, N, 3]

            pc_colors = np.array(pc.colors)
            query_colors = torch.tensor(pc_colors, dtype=torch.float32).unsqueeze(
                0
            )  # [B, N, 3]
            sample_ids = np.random.choice(np.arange(query_colors.size(1)), 2048)

            query_colors = query_colors[:, sample_ids]
            query_points = query_points[:, sample_ids]
            image_tensor = image_tensor.permute(0, 2, 3, 1)
            projector = ProjectColorOntoImage_v2()
            output_image = projector(
                image_grid=image_tensor,
                query_points=query_points,
                query_colors=query_colors,
                intrinsics=intrinsics,
            )
            # save_path = "exps/heatmap_backalign.png"

            for i, img in enumerate(output_image):
                if img.ndim == 3:

                    color_image = T.ToPILImage()(image_tensor[i].cpu())
                    pil_image = T.ToPILImage()(img.cpu())

                    image_np = np.clip(pil_image, 0, 255)
                    # color_image_np = np.array(color_image)
                    color_image_np = np.floor(color_image)
                    color_image_np = np.clip(color_image_np, 0, 255)
                    color_image_np = np.uint8(color_image_np)
                    # image_np = cv2.blur(image_np, (50, 50))

                    image_np = cv2.addWeighted(image_np, 0.3, color_image_np, 0.7, 0.0)
                    pil_image = Image.fromarray(np.uint8(image_np))
                    # pil_image = Image.fromarray(np.float32(image_np))
                    # pil_image.save(save_path)
                    # print(f"img is saved {save_path}")
                    img_heatmap_list.append(pil_image)
        return img_heatmap_list, self.file_name, self.phrase


class AttentionLayer(nn.Module):
    def __init__(self, C_rgbfeat, direction_dim):
        super(AttentionLayer, self).__init__()

        # 将 direction_text 投影到 C_rgbfeat 维度
        self.direction_projection = nn.Linear(direction_dim, C_rgbfeat)

        # 用于生成 attention 权重的卷积层
        self.attention_conv = nn.Conv2d(C_rgbfeat, 1, kernel_size=1)

        # 用于加权和融合 x_rgb 和 direction_text 的卷积层
        self.fusion_conv = nn.Conv2d(C_rgbfeat, C_rgbfeat, kernel_size=1)

        # Layer normalization for x_rgb
        self.norm_rgb = nn.LayerNorm([C_rgbfeat, 480, 640])  # C * H * W

    def forward(self, x_rgb, direction_text):
        batch_size, C_rgbfeat, H, W = x_rgb.shape

        # 1. 投影 direction_text 到 (Batchsize, C_rgbfeat)
        direction_feats = self.direction_projection(
            direction_text
        )  # (Batchsize, C_rgbfeat)

        # 2. 将 direction_feats 调整为 (Batchsize, C_rgbfeat, 1, 1)，并广播到 (Batchsize, C_rgbfeat, H, W)
        direction_feats = (
            direction_feats.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        )

        # 3. 归一化 x_rgb
        x_rgb_norm = self.norm_rgb(x_rgb)

        # 4. 生成注意力权重 (Batchsize, 1, H, W)
        attn_weights = self.attention_conv(x_rgb_norm)
        attn_weights = torch.sigmoid(
            attn_weights
        )  # 使用 sigmoid 使注意力权重范围在 [0, 1] 之间

        # 5. 使用注意力权重加权 direction_feats
        attn_feats = attn_weights * direction_feats  # (Batchsize, C_rgbfeat, H, W)

        # 6. 融合加权后的 direction_feats 和原始的 x_rgb
        fused_feats = x_rgb + attn_feats  # (Batchsize, C_rgbfeat, H, W)

        # 7. 使用卷积进一步融合特征 (保持通道数不变)
        output = self.fusion_conv(fused_feats)  # (Batchsize, C_rgbfeat, H, W)

        return output


@registry.register_affordance_model(name="GeoL_net_v3")
class GeoL_net_v3(nn.Module):
    """
    GeoL_net_v3:
        1. input the anchor object position
        2. use perciefer architecture to inject the direction text
    """

    def __init__(self, input_shape, target_input_shape):
        super().__init__()
        self.geoafford_module = GeoAffordModule(feat_dim=16)  # 0.3kw
        self.clipunet_module = CLIPUNet(
            input_shape=input_shape,
            target_input_shape=target_input_shape,
            output_dim=64,
        )  # 6kw parameters
        self.concate = FeatureConcat()
        self.device = "cuda"  # cpu for dataset
        self.lang_fusion_type = "mult"  # hard code from CLIPlingunet
        self._load_clip()
        self.instrics = np.array(
            [[591.0125, 0.0, 322.525], [0.0, 590.16775, 244.11084], [0.0, 0.0, 1.0]]
        )
        self.fusion_point_moudule = FusionPointLayer()  # 0.3kw
        self.direction_mlp = nn.Sequential(nn.Linear(1024, 256), nn.ReLU())

        self.anchor_mlp = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 256), nn.ReLU()
        )

        self.feat_perceiver = FeaturePerceiver(
            transition_dim=115,
            condition_dim=256,
            #    time_emb_dim=64
        )

    def _load_clip(self):
        model, _ = load_clip("RN50", device=self.device)
        self.clip_rn50 = build_model(model.state_dict()).to(self.device)  # 10kw frozen
        del model
        # Frozen clip
        for param in self.clip_rn50.parameters():
            param.requires_grad = False

    def encode_text(self, x):
        with torch.no_grad():
            tokens = tokenize(x).to(self.device)
            text_feat, text_emb = self.clip_rn50.encode_text_with_embeddings(tokens)

        text_mask = torch.where(tokens == 0, tokens, 1)  # [1, max_token_len]
        return text_feat, text_emb, text_mask

    def encode_direction(self, x):
        with torch.no_grad():
            tokens = tokenize(x).to(self.device)
            text_feat, text_emb = self.clip_rn50.encode_text_with_embeddings(tokens)
        text_mask = torch.where(tokens == 0, tokens, 1)  # [1, max_token_len]
        return text_feat, text_emb, text_mask

    def forward(self, **kwargs):
        self.batch = kwargs["batch"]
        batch = kwargs["batch"]
        self.phrase = batch["phrase"]
        self.file_name = batch["file_path"]
        texts = batch["phrase"]
        reference_obj_name = batch["reference_obj"]
        direction_text = batch["direction_text"]
        anchor_pos = batch["anchor_position"]  # 新增 anchor position
        output = {}

        # encode text(object name)
        l_enc, l_emb, l_mask = self.encode_text(reference_obj_name)
        l_input = l_emb if "word" in self.lang_fusion_type else l_enc
        l_input = l_input.to(dtype=torch.float32)
        batch["target_query"] = l_input

        # encode text(direction)
        d_enc, d_emb, d_mask = self.encode_direction(direction_text)
        d_input = d_emb if "word" in self.lang_fusion_type else d_enc
        d_input = d_input.to(dtype=torch.float32)
        d_input = self.direction_mlp(d_input)  # direction: B*256
        batch["direction_query"] = d_input

        # point cloud
        scene_pcs = batch["fps_points_scene"]
        obj_pcs = batch["fps_points_scene"]

        # pointnet_1: extract the geometric affordance feat.
        x_geo = self.geoafford_module(
            scene_pcs, obj_pcs
        )  # [B, C, Num_pts] [B, 48, Num_pts]

        # CLIP: extract the rgb feat.
        x_rgb = self.clipunet_module(batch=batch)  # [B, C_rgbfeat, H, W] [B, 64, H, W]

        # merge 得到场景特征 F=115
        x_align = self.concate(x_rgb["affordance"], scene_pcs, self.instrics).permute(
            0, 2, 1
        )  # [B, Num_pts, C_rgbfeat]
        x_scene = torch.cat(
            (scene_pcs.permute(0, 2, 1), x_align, x_geo), dim=1
        )  # scene_pc, x_align, x_geo merge F=48+64+3 115 场景特征 [B, 115, 2048]
        x_scene = x_scene.permute(0, 2, 1)  # [B, 2048, 115]

        # 处理anchor position 特征
        # input: B*3
        x_anchor_position = self.anchor_mlp(anchor_pos)  # output: [B, 256]

        # 处理 direction text
        x_anchor_and_text = torch.cat(
            (x_anchor_position, d_input), dim=1
        )  # [B,  (256 + 256) ]
        x_anchor_and_text = x_anchor_and_text.unsqueeze(1)

        # perceive 处理x_scene and x_anchor_and_text
        x = self.feat_perceiver(x_scene, x_anchor_and_text)  # [B, 2048, 256]

        x = self.fusion_point_moudule(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)

        output["affordance"] = x
        self.model_ouput = x  # [B, 2048, 1]
        self.model_pc = scene_pcs

        return output

    def generate_heatmap(self, epoch):
        "generate heatmap during training"
        self.pc_heatmap_list = []
        pcs = self.model_pc  # [B, Num_points, 3]
        feat = self.model_ouput  # [B, Num_points, 1]
        # Normalize feat to [0, 1] for each batch independently
        min_feat = feat.min(dim=1, keepdim=True)[0]  # Find min in each batch
        max_feat = feat.max(dim=1, keepdim=True)[0]  # Find max in each batch
        normalized_feat = (feat - min_feat) / (
            max_feat - min_feat + 1e-6
        )  # Avoid division by zero
        # Convert normalized features to RGB using turbo colormap
        turbo_colormap = cm.get_cmap(
            "turbo", 256
        )  # Load turbo colormap with 256 levels
        normalized_feat_np = (
            normalized_feat.squeeze(-1).cpu().detach().numpy()
        )  # Shape: [B, Num_points]
        # Apply turbo colormap to normalized features (converts values to RGB)
        color_maps = turbo_colormap(normalized_feat_np)[
            :, :, :3
        ]  # Shape: [B, Num_points, 3], ignore alpha
        # Generate point cloud with Open3D
        for i in range(pcs.shape[0]):  # Iterate over batch
            points = pcs[i].cpu().numpy()  # Shape: [Num_points, 3]
            colors = color_maps[i]  # Shape: [Num_points, 3]

            # Create Open3D point cloud object
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            # Save the point cloud to a file (you can change the path format)
            # o3d.io.write_point_cloud(f"outputs/model_output/point_cloud/{scene_id}-{obj_id}-{epoch}_heatmap.ply", pcd)
            self.pc_heatmap_list.append(pcd)

    def pcdheatmap2img(self, is_mask=False):
        pcs_heatmap_list = self.pc_heatmap_list
        img_rgb_list = self.batch["image"]
        img_heatmap_list = []
        assert len(pcs_heatmap_list) == len(img_rgb_list)
        cam_rotation_matrix = np.array([[1, 0, 0], [0, 0.8, -0.6], [0, 0.6, 0.8]])
        intrinsics = np.array(
            [[591.0125, 0.0, 322.525], [0.0, 590.16775, 244.11084], [0.0, 0.0, 1.0]]
        )
        for i in range(len(pcs_heatmap_list)):
            pc = pcs_heatmap_list[i]
            img_rgb = img_rgb_list[i]
            image_np = img_rgb.cpu().detach().numpy().astype(np.float32)

            image_tensor = (
                torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0)
            )  # [B, C, H, W]

            #

            pc_points = np.array(pc.points)

            if is_mask:
                pc_points = pc_points @ cam_rotation_matrix
            # pc_points = pc_points @ cam_rotation_matrix # if mask

            # pc_points = pc_points @ np.linalg.inv(cam_rotation_matrix)

            query_points = torch.tensor(pc_points, dtype=torch.float32).unsqueeze(
                0
            )  # [B, N, 3]

            pc_colors = np.array(pc.colors)
            query_colors = torch.tensor(pc_colors, dtype=torch.float32).unsqueeze(
                0
            )  # [B, N, 3]
            sample_ids = np.random.choice(np.arange(query_colors.size(1)), 2048)

            query_colors = query_colors[:, sample_ids]
            query_points = query_points[:, sample_ids]
            image_tensor = image_tensor.permute(0, 2, 3, 1)
            projector = ProjectColorOntoImage_v2()
            output_image = projector(
                image_grid=image_tensor,
                query_points=query_points,
                query_colors=query_colors,
                intrinsics=intrinsics,
            )
            # save_path = "exps/heatmap_backalign.png"

            for i, img in enumerate(output_image):
                if img.ndim == 3:

                    color_image = T.ToPILImage()(image_tensor[i].cpu())
                    pil_image = T.ToPILImage()(img.cpu())

                    image_np = np.clip(pil_image, 0, 255)
                    # color_image_np = np.array(color_image)
                    color_image_np = np.floor(color_image)
                    color_image_np = np.clip(color_image_np, 0, 255)
                    color_image_np = np.uint8(color_image_np)
                    # image_np = cv2.blur(image_np, (50, 50))

                    image_np = cv2.addWeighted(image_np, 0.3, color_image_np, 0.7, 0.0)
                    pil_image = Image.fromarray(np.uint8(image_np))
                    # pil_image = Image.fromarray(np.float32(image_np))
                    # pil_image.save(save_path)
                    # print(f"img is saved {save_path}")
                    img_heatmap_list.append(pil_image)
        return img_heatmap_list, self.file_name, self.phrase


class FeaturePerceiver(nn.Module):

    def __init__(
        self,
        transition_dim,
        condition_dim,
        time_emb_dim=-1,
        encoder_q_input_channels=512,
        encoder_kv_input_channels=256,
        encoder_num_heads=8,
        encoder_widening_factor=1,
        encoder_dropout=0.1,
        encoder_residual_dropout=0.0,
        encoder_self_attn_num_layers=2,
        decoder_q_input_channels=256,
        decoder_kv_input_channels=512,
        decoder_num_heads=8,
        decoder_widening_factor=1,
        decoder_dropout=0.1,
        decoder_residual_dropout=0.0,
    ) -> None:
        super().__init__()

        self.encoder_q_input_channels = encoder_q_input_channels
        self.encoder_kv_input_channels = encoder_kv_input_channels
        self.encoder_num_heads = encoder_num_heads
        self.encoder_widening_factor = encoder_widening_factor
        self.encoder_dropout = encoder_dropout
        self.encoder_residual_dropout = encoder_residual_dropout
        self.encoder_self_attn_num_layers = encoder_self_attn_num_layers

        self.decoder_q_input_channels = decoder_q_input_channels
        self.decoder_kv_input_channels = decoder_kv_input_channels
        self.decoder_num_heads = decoder_num_heads
        self.decoder_widening_factor = decoder_widening_factor
        self.decoder_dropout = decoder_dropout
        self.decoder_residual_dropout = decoder_residual_dropout

        self.condition_adapter = nn.Linear(
            condition_dim, self.encoder_q_input_channels, bias=True
        )

        if time_emb_dim > 0:
            self.time_embedding_adapter = nn.Linear(
                time_emb_dim, self.encoder_q_input_channels, bias=True
            )
        else:
            self.time_embedding_adapter = None

        self.encoder_adapter = nn.Linear(
            transition_dim,
            self.encoder_kv_input_channels,
            bias=True,
        )
        self.decoder_adapter = nn.Linear(
            self.encoder_kv_input_channels, self.decoder_q_input_channels, bias=True
        )

        self.encoder_cross_attn = CrossAttentionLayer(
            num_heads=self.encoder_num_heads,
            num_q_input_channels=self.encoder_q_input_channels,
            num_kv_input_channels=self.encoder_kv_input_channels,
            widening_factor=self.encoder_widening_factor,
            dropout=self.encoder_dropout,
            residual_dropout=self.encoder_residual_dropout,
        )

        self.encoder_self_attn = SelfAttentionBlock(
            num_layers=self.encoder_self_attn_num_layers,
            num_heads=self.encoder_num_heads,
            num_channels=self.encoder_q_input_channels,
            widening_factor=self.encoder_widening_factor,
            dropout=self.encoder_dropout,
            residual_dropout=self.encoder_residual_dropout,
        )

        self.decoder_cross_attn = CrossAttentionLayer(
            num_heads=self.decoder_num_heads,
            num_q_input_channels=self.decoder_q_input_channels,
            num_kv_input_channels=self.decoder_kv_input_channels,
            widening_factor=self.decoder_widening_factor,
            dropout=self.decoder_dropout,
            residual_dropout=self.decoder_residual_dropout,
        )
        self.last_dim = self.decoder_q_input_channels

    def forward(
        self,
        x,
        condition_feat,
        time_embedding=None,
    ):
        """Forward pass of the ContactMLP.

        Args:
            x: input contact map, [bs, num_points, transition_dim]
            condition_feat: [bs, 1, condition_dim]
            time_embedding: [bs, 1, time_embedding_dim]

        Returns:
            Output contact map, [bs, num_points, contact_dim]
        """

        # encoder
        # import pdb; pdb.set_trace()
        x = x.float()
        enc_kv = self.encoder_adapter(x)  # [bs, num_points, enc_kv_dim]
        cond_feat = self.condition_adapter(condition_feat)  # [bs, 1, enc_q_dim]
        if time_embedding is not None and self.time_embedding_adapter is not None:
            time_embedding = self.time_embedding_adapter(
                time_embedding
            )  # [bs, 1, enc_q_dim]

            enc_q = torch.cat(
                [cond_feat, time_embedding], dim=1
            )  # [bs, 1 + 1, enc_q_dim]
        else:
            enc_q = cond_feat

        enc_q = self.encoder_cross_attn(enc_q, enc_kv).last_hidden_state
        enc_q = self.encoder_self_attn(enc_q).last_hidden_state

        # decoder
        dec_kv = enc_q
        dec_q = self.decoder_adapter(enc_kv)  # [bs, num_points, dec_q_dim]
        dec_q = self.decoder_cross_attn(
            dec_q, dec_kv
        ).last_hidden_state  # [bs, num_points, dec_q_dim]

        return dec_q


@registry.register_affordance_model(name="GeoL_net_v4")
class GeoL_net_v4(nn.Module):
    """
    GeoL_net_v4:
        1. adjust the parameters
        2. use the scene point position directly
        3. introduce one-hot to process direction text
    """

    def __init__(self, input_shape, target_input_shape):
        super().__init__()
        self.geoafford_module = GeoAffordModule(feat_dim=16)  # 0.3kw
        self.clipunet_module = CLIPUNet(
            input_shape=input_shape,
            target_input_shape=target_input_shape,
            output_dim=64,
        )  # 6kw parameters
        self.concate = FeatureConcat()
        self.device = "cuda"  # cpu for dataset
        self.lang_fusion_type = "mult"  # hard code from CLIPlingunet
        self._load_clip()
        self.instrics = np.array(
            [[591.0125, 0.0, 322.525], [0.0, 590.16775, 244.11084], [0.0, 0.0, 1.0]]
        )
        self.fusion_point_moudule = FusionPointLayer(input_dim=256)  # 0.3kw
        self.direction_mlp = nn.Sequential(nn.Linear(8, 256), nn.ReLU())

        self.anchor_mlp = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
        )

        self.feat_perceiver = FeaturePerceiver(
            transition_dim=115,
            condition_dim=256,
        )

        self.direction_encoder()

    def _load_clip(self):
        model, _ = load_clip("RN50", device=self.device)
        self.clip_rn50 = build_model(model.state_dict()).to(self.device)  # 10kw frozen
        del model
        # Frozen clip
        for param in self.clip_rn50.parameters():
            param.requires_grad = False

    def encode_text(self, x):
        with torch.no_grad():
            tokens = tokenize(x).to(self.device)
            text_feat, text_emb = self.clip_rn50.encode_text_with_embeddings(tokens)

        text_mask = torch.where(tokens == 0, tokens, 1)  # [1, max_token_len]
        return text_feat, text_emb, text_mask

    def direction_encoder(self):
        """
        encode the direction text through embedding
        """
        self.directions = [
            "Left",
            "Right",
            "Front",
            "Behind",
            "Left Front",
            "Right Front",
            "Left Behind",
            "Right Behind",
        ]
        # 创建一个词汇表，将每个方位映射到一个唯一的整数
        self.vocab = {word: idx for idx, word in enumerate(self.directions)}

    def forward(self, **kwargs):
        self.batch = kwargs["batch"]
        batch = kwargs["batch"]
        self.phrase = batch["phrase"]
        self.file_name = batch["file_path"]
        texts = batch["phrase"]
        reference_obj_name = batch["reference_obj"]
        direction_text = batch["direction_text"]
        anchor_pos = batch["anchor_position"]  # 新增 anchor position
        output = {}

        # encode text(object name)
        l_enc, l_emb, l_mask = self.encode_text(reference_obj_name)
        l_input = l_emb if "word" in self.lang_fusion_type else l_enc
        l_input = l_input.to(dtype=torch.float32)
        batch["target_query"] = l_input

        # encode text(direction)
        direction_inputs_indices = torch.tensor(
            list(map(self.vocab.get, batch["direction_text"]))
        )
        d_input = torch.eye(len(self.vocab))[
            direction_inputs_indices
        ].cuda()  # [batch_size, 8] one-hot
        d_input = self.direction_mlp(d_input)  # direction: [B, 256]
        # point cloud
        scene_pcs = batch["fps_points_scene"]
        obj_pcs = batch["fps_points_scene"]

        # pointnet_1: extract the geometric affordance feat.
        x_geo = self.geoafford_module(
            scene_pcs, obj_pcs
        )  # [B, C, Num_pts] [B, 48, Num_pts]

        # CLIP: extract the rgb feat.
        x_rgb = self.clipunet_module(batch=batch)  # [B, C_rgbfeat, H, W] [B, 64, H, W]

        # merge 得到场景特征 F=115
        x_align = self.concate(x_rgb["affordance"], scene_pcs, self.instrics).permute(
            0, 2, 1
        )  # [B, Num_pts, C_rgbfeat]
        x_scene = torch.cat(
            (scene_pcs.permute(0, 2, 1), x_align, x_geo), dim=1
        )  # scene_pc, x_align, x_geo merge F=48+64+3 115 场景特征 [B, 115, 2048]
        x_scene = x_scene.permute(0, 2, 1)  # [B, 2048, 3]

        # 处理anchor position 特征
        # input: B*3
        x_anchor_position = self.anchor_mlp(anchor_pos)  # output: [B, 128]

        # 处理 direction text
        x_anchor_and_text = torch.cat(
            (x_anchor_position, d_input), dim=1
        )  # [B,  (128 + 256) ]
        x_anchor_and_text = x_anchor_and_text.unsqueeze(1)  # [B, 1, 128+256]

        # perceive 处理x_scene and x_anchor_and_text
        x = self.feat_perceiver(x_scene, x_anchor_and_text)  # [B, 2048, 256]

        x = self.fusion_point_moudule(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)

        output["affordance"] = x
        self.model_ouput = x  # [B, 2048, 1]
        self.model_pc = scene_pcs

        return output

    def generate_heatmap(self, epoch):
        "generate heatmap during training"
        self.pc_heatmap_list = []
        pcs = self.model_pc  # [B, Num_points, 3]
        feat = self.model_ouput  # [B, Num_points, 1]
        # Normalize feat to [0, 1] for each batch independently
        min_feat = feat.min(dim=1, keepdim=True)[0]  # Find min in each batch
        max_feat = feat.max(dim=1, keepdim=True)[0]  # Find max in each batch
        normalized_feat = (feat - min_feat) / (
            max_feat - min_feat + 1e-6
        )  # Avoid division by zero
        # Convert normalized features to RGB using turbo colormap
        turbo_colormap = cm.get_cmap(
            "turbo", 256
        )  # Load turbo colormap with 256 levels
        normalized_feat_np = (
            normalized_feat.squeeze(-1).cpu().detach().numpy()
        )  # Shape: [B, Num_points]
        # Apply turbo colormap to normalized features (converts values to RGB)
        color_maps = turbo_colormap(normalized_feat_np)[
            :, :, :3
        ]  # Shape: [B, Num_points, 3], ignore alpha
        # Generate point cloud with Open3D
        for i in range(pcs.shape[0]):  # Iterate over batch
            points = pcs[i].cpu().numpy()  # Shape: [Num_points, 3]
            colors = color_maps[i]  # Shape: [Num_points, 3]

            # Create Open3D point cloud object
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            self.pc_heatmap_list.append(pcd)

    def pcdheatmap2img(self, is_mask=False):
        pcs_heatmap_list = self.pc_heatmap_list
        img_rgb_list = self.batch["image"]
        img_heatmap_list = []
        assert len(pcs_heatmap_list) == len(img_rgb_list)
        cam_rotation_matrix = np.array([[1, 0, 0], [0, 0.8, -0.6], [0, 0.6, 0.8]])
        intrinsics = np.array(
            [[591.0125, 0.0, 322.525], [0.0, 590.16775, 244.11084], [0.0, 0.0, 1.0]]
        )
        for i in range(len(pcs_heatmap_list)):
            pc = pcs_heatmap_list[i]
            img_rgb = img_rgb_list[i]
            image_np = img_rgb.cpu().detach().numpy().astype(np.float32)

            image_tensor = (
                torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0)
            )  # [B, C, H, W]
            pc_points = np.array(pc.points)

            if is_mask:
                pc_points = pc_points @ cam_rotation_matrix

            query_points = torch.tensor(pc_points, dtype=torch.float32).unsqueeze(
                0
            )  # [B, N, 3]

            pc_colors = np.array(pc.colors)
            query_colors = torch.tensor(pc_colors, dtype=torch.float32).unsqueeze(
                0
            )  # [B, N, 3]
            sample_ids = np.random.choice(np.arange(query_colors.size(1)), 2048)

            query_colors = query_colors[:, sample_ids]
            query_points = query_points[:, sample_ids]
            image_tensor = image_tensor.permute(0, 2, 3, 1)
            projector = ProjectColorOntoImage_v2()
            output_image = projector(
                image_grid=image_tensor,
                query_points=query_points,
                query_colors=query_colors,
                intrinsics=intrinsics,
            )

            for i, img in enumerate(output_image):
                if img.ndim == 3:

                    color_image = T.ToPILImage()(image_tensor[i].cpu())
                    pil_image = T.ToPILImage()(img.cpu())

                    image_np = np.clip(pil_image, 0, 255)

                    color_image_np = np.floor(color_image)
                    color_image_np = np.clip(color_image_np, 0, 255)
                    color_image_np = np.uint8(color_image_np)

                    image_np = cv2.addWeighted(image_np, 0.3, color_image_np, 0.7, 0.0)
                    pil_image = Image.fromarray(np.uint8(image_np))
                    img_heatmap_list.append(pil_image)
        return img_heatmap_list, self.file_name, self.phrase


@registry.register_affordance_model(name="GeoL_net_v5")
class GeoL_net_v5(nn.Module):
    """
    GeoL_net_v5:
        1. Merge the anchor position info with scene feature in advance
        2. Use Perceiver to merge the scene feature and direction text
    """

    def __init__(self, input_shape, target_input_shape):
        super().__init__()
        self.geoafford_module = GeoAffordModule(feat_dim=16)  # 0.3kw
        self.clipunet_module = CLIPUNet(
            input_shape=input_shape,
            target_input_shape=target_input_shape,
            output_dim=64,
        )  # 6kw parameters
        self.concate = FeatureConcat()
        self.device = "cuda"  # cpu for dataset
        self.lang_fusion_type = "mult"  # hard code from CLIPlingunet
        self._load_clip()
        self.instrics = np.array(
            [[591.0125, 0.0, 322.525], [0.0, 590.16775, 244.11084], [0.0, 0.0, 1.0]]
        )
        self.fusion_point_moudule = FusionPointLayer(input_dim=256)  # 0.3kw
        self.direction_mlp = nn.Sequential(nn.Linear(8, 256), nn.ReLU())

        self.anchor_mlp = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
        )

        self.feat_perceiver = FeaturePerceiver(
            transition_dim=3, condition_dim=8, time_emb_dim=2
        )

        self.direction_encoder()

    def _load_clip(self):
        model, _ = load_clip("RN50", device=self.device)
        self.clip_rn50 = build_model(model.state_dict()).to(self.device)  # 10kw frozen
        del model
        # Frozen clip
        for param in self.clip_rn50.parameters():
            param.requires_grad = False

    def encode_text(self, x):
        with torch.no_grad():
            tokens = tokenize(x).to(self.device)
            text_feat, text_emb = self.clip_rn50.encode_text_with_embeddings(tokens)

        text_mask = torch.where(tokens == 0, tokens, 1)  # [1, max_token_len]
        return text_feat, text_emb, text_mask

    def direction_encoder(self):
        """
        encode the direction text through embedding
        """
        self.directions = [
            "Left",
            "Right",
            "Front",
            "Behind",
            "Left Front",
            "Right Front",
            "Left Behind",
            "Right Behind",
        ]
        # 创建一个词汇表，将每个方位映射到一个唯一的整数
        self.vocab = {word: idx for idx, word in enumerate(self.directions)}

    def forward(self, **kwargs):
        self.batch = kwargs["batch"]
        batch = kwargs["batch"]
        self.phrase = batch["phrase"]
        self.file_name = batch["file_path"]
        texts = batch["phrase"]
        reference_obj_name = batch["reference_obj"]
        direction_text = batch["direction_text"]
        anchor_pos = batch["anchor_position"]  # 新增 anchor position
        output = {}

        # encode text(object name)
        # l_enc, l_emb, l_mask = self.encode_text(reference_obj_name)
        # l_input = l_emb if 'word' in self.lang_fusion_type else l_enc
        # l_input = l_input.to(dtype=torch.float32)
        # batch["target_query"] = l_input

        # encode text(direction)
        direction_inputs_indices = torch.tensor(
            list(map(self.vocab.get, batch["direction_text"]))
        )
        d_input = torch.eye(len(self.vocab))[
            direction_inputs_indices
        ].cuda()  # [batch_size, 8] one-hot
        # d_input = self.direction_mlp(d_input) # direction: [B, 256]
        # point cloud
        scene_pcs = batch["fps_points_scene"]
        # obj_pcs = batch["fps_points_scene"]

        # pointnet_1: extract the geometric affordance feat.
        # x_geo = self.geoafford_module(scene_pcs, obj_pcs) # [B, C, Num_pts] [B, 48, Num_pts]

        # CLIP: extract the rgb feat.
        # x_rgb = self.clipunet_module(batch=batch) # [B, C_rgbfeat, H, W] [B, 64, H, W]

        # merge 得到场景特征 F=115
        # x_align = self.concate(x_rgb["affordance"], scene_pcs, self.instrics).permute(0,2,1) # [B, Num_pts, C_rgbfeat]
        # x_scene = torch.cat((scene_pcs.permute(0,2,1), x_align, x_geo),dim=1) # scene_pc, x_align, x_geo merge F=48+64+3 115 场景特征 [B, 115, 2048]
        # each position minues anchor position
        x_scene = scene_pcs - anchor_pos.unsqueeze(1)  # [B, Num_points, 3]
        # x_scene = x_scene.permute(0,2,1) # [B, 2048, 3]

        # 处理anchor position 特征
        # input: B*3
        # x_anchor_position =self.anchor_mlp(anchor_pos) # output: [B, 128]

        # 处理 direction text
        x_anchor_and_text = d_input  # [B,  (256) ]
        x_anchor_and_text = x_anchor_and_text.unsqueeze(1)  # [B, 1, 8]

        # perceive 处理x_scene and x_anchor_and_text
        x = self.feat_perceiver(x_scene, x_anchor_and_text)  # [B, 2048, 256]

        x = self.fusion_point_moudule(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)

        output["affordance"] = x
        self.model_ouput = x  # [B, 2048, 1]
        self.model_pc = scene_pcs

        return output

    def generate_heatmap(self, epoch):
        "generate heatmap during training"
        self.pc_heatmap_list = []
        pcs = self.model_pc  # [B, Num_points, 3]
        feat = self.model_ouput  # [B, Num_points, 1]
        normalized_feat = feat.sigmoid()
        # Normalize feat to [0, 1] for each batch independently
        # min_feat = feat.min(dim=1, keepdim=True)[0]  # Find min in each batch
        # max_feat = feat.max(dim=1, keepdim=True)[0]  # Find max in each batch
        # normalized_feat = (feat - min_feat) / (max_feat - min_feat + 1e-6)  # Avoid division by zero
        # Convert normalized features to RGB using turbo colormap
        turbo_colormap = cm.get_cmap(
            "turbo", 256
        )  # Load turbo colormap with 256 levels
        normalized_feat_np = (
            normalized_feat.squeeze(-1).cpu().detach().numpy()
        )  # Shape: [B, Num_points]
        # Apply turbo colormap to normalized features (converts values to RGB)
        color_maps = turbo_colormap(normalized_feat_np)[
            :, :, :3
        ]  # Shape: [B, Num_points, 3], ignore alpha
        # Generate point cloud with Open3D
        for i in range(pcs.shape[0]):  # Iterate over batch
            points = pcs[i].cpu().numpy()  # Shape: [Num_points, 3]
            colors = color_maps[i]  # Shape: [Num_points, 3]

            # Create Open3D point cloud object
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            self.pc_heatmap_list.append(pcd)

    def pcdheatmap2img(self, is_mask=False):
        pcs_heatmap_list = self.pc_heatmap_list
        img_rgb_list = self.batch["image"]
        img_heatmap_list = []
        assert len(pcs_heatmap_list) == len(img_rgb_list)
        cam_rotation_matrix = np.array([[1, 0, 0], [0, 0.8, -0.6], [0, 0.6, 0.8]])
        intrinsics = np.array(
            [[591.0125, 0.0, 322.525], [0.0, 590.16775, 244.11084], [0.0, 0.0, 1.0]]
        )
        for i in range(len(pcs_heatmap_list)):
            pc = pcs_heatmap_list[i]
            img_rgb = img_rgb_list[i]
            image_np = img_rgb.cpu().detach().numpy().astype(np.float32)

            image_tensor = (
                torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0)
            )  # [B, C, H, W]
            pc_points = np.array(pc.points)

            if is_mask:
                pc_points = pc_points @ cam_rotation_matrix

            query_points = torch.tensor(pc_points, dtype=torch.float32).unsqueeze(
                0
            )  # [B, N, 3]

            pc_colors = np.array(pc.colors)
            query_colors = torch.tensor(pc_colors, dtype=torch.float32).unsqueeze(
                0
            )  # [B, N, 3]
            sample_ids = np.random.choice(np.arange(query_colors.size(1)), 2048)

            query_colors = query_colors[:, sample_ids]
            query_points = query_points[:, sample_ids]
            image_tensor = image_tensor.permute(0, 2, 3, 1)
            projector = ProjectColorOntoImage_v2()
            output_image = projector(
                image_grid=image_tensor,
                query_points=query_points,
                query_colors=query_colors,
                intrinsics=intrinsics,
            )

            for i, img in enumerate(output_image):
                if img.ndim == 3:

                    color_image = T.ToPILImage()(image_tensor[i].cpu())
                    pil_image = T.ToPILImage()(img.cpu())

                    image_np = np.clip(pil_image, 0, 255)

                    color_image_np = np.floor(color_image)
                    color_image_np = np.clip(color_image_np, 0, 255)
                    color_image_np = np.uint8(color_image_np)

                    image_np = cv2.addWeighted(image_np, 0.3, color_image_np, 0.7, 0.0)
                    pil_image = Image.fromarray(np.uint8(image_np))
                    img_heatmap_list.append(pil_image)
        return img_heatmap_list, self.file_name, self.phrase


@registry.register_affordance_model(name="GeoL_net_v6")
class GeoL_net_v6(nn.Module):
    """
    GeoL_net_v6:
        1. Merge the anchor position info with scene feature in advance
        2. Use Perceiver to merge the scene feature and direction text
        3. use lagger model compated to v5
        4. still ingore the image info
    """

    def __init__(self, input_shape, target_input_shape):
        super().__init__()
        self.geoafford_module = GeoAffordModule(feat_dim=16)  # 0.3kw
        self.clipunet_module = CLIPUNet(
            input_shape=input_shape,
            target_input_shape=target_input_shape,
            output_dim=64,
        )  # 6kw parameters
        self.concate = FeatureConcat()
        self.device = "cuda"  # cpu for dataset
        self.lang_fusion_type = "mult"  # hard code from CLIPlingunet
        self._load_clip()
        self.instrics = np.array(
            [[591.0125, 0.0, 322.525], [0.0, 590.16775, 244.11084], [0.0, 0.0, 1.0]]
        )
        # self.fusion_point_moudule = FusionPointLayer(input_dim=256) #0.3kw
        self.direction_mlp = nn.Sequential(
            nn.Linear(8, 256),
            nn.TransformerDecoderLayer(
                d_model=256, nhead=4, dim_feedforward=512, batch_first=True
            ),
        )

        self.anchor_mlp = nn.Sequential(
            nn.Linear(3, 256),
            nn.TransformerEncoderLayer(
                d_model=256, nhead=4, dim_feedforward=512, batch_first=True
            ),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(1024 + 256, 256),
            nn.TransformerEncoderLayer(
                d_model=256, nhead=4, dim_feedforward=512, batch_first=True
            ),
        )
        self.fusion_point_moudule = nn.Sequential(
            nn.Linear(256, 256),
            nn.TransformerEncoderLayer(
                d_model=256, nhead=8, dim_feedforward=512, batch_first=True
            ),
            nn.Linear(256, 64),
            nn.TransformerEncoderLayer(
                d_model=64, nhead=4, dim_feedforward=256, batch_first=True
            ),
            nn.Linear(64, 1),
        )

        self.feat_perceiver = FeaturePerceiver(
            transition_dim=118, condition_dim=256, time_emb_dim=32
        )

    def _load_clip(self):
        model, _ = load_clip("RN50", device=self.device)
        self.clip_rn50 = build_model(model.state_dict()).to(self.device)  # 10kw frozen
        del model
        # Frozen clip
        for param in self.clip_rn50.parameters():
            param.requires_grad = False

    def encode_text(self, x):
        with torch.no_grad():
            tokens = tokenize(x).to(self.device)
            text_feat, text_emb = self.clip_rn50.encode_text_with_embeddings(tokens)

        text_mask = torch.where(tokens == 0, tokens, 1)  # [1, max_token_len]
        return text_feat, text_emb, text_mask

    def direction_encoder(self, x):
        """
        encode the direction text through embedding
        """
        with torch.no_grad():
            tokens = tokenize(x).to(self.device)
            text_feat, text_emb = self.clip_rn50.encode_text_with_embeddings(tokens)
        text_mask = torch.where(tokens == 0, tokens, 1)  # [1, max_token_len]
        return text_feat, text_emb, text_mask

        # self.directions = ["Left", "Right", "Front", "Behind", "Left Front", "Right Front", "Left Behind", "Right Behind"]
        # self.vocab = {word: idx for idx, word in enumerate(self.directions)}

    def update_text(self, direction_text_batch):
        """
        update the direction text
        """
        for i, text in enumerate(direction_text_batch):
            if text == "Left":
                direction_text_batch[i] = (
                    "Left, After a long and exhausting day at work, I left the office feeling both relieved and tired, looking forward to finally going home, where I could rest and unwind peacefully\
                    The left part of my car is damaged. I need to take it to the repair shop to get it fixed."
                )
            elif text == "Right":
                direction_text_batch[i] = (
                    "Right, The right side of the brain is responsible for creativity. do you like to draw or paint? If so, you are using the right side of your brain\
                    Blue is the color of the sky on a clear day. The sky is blue because of the way the Earth's atmosphere scatters light from the sun."
                )
            elif text == "Front":
                direction_text_batch[i] = (
                    "Front is the best. The front of the house is where the garden is located. The garden is a beautiful place to relax and enjoy the outdoors.\
                    TUM is a university in Munich, Germany. It is located in the front of the city, near the city center. "
                )
            elif text == "Behind":
                direction_text_batch[i] = (
                    "Behind hahaha The cat is hiding behind the couch. I can see its tail sticking out from behind the couch.\
                    Zhejiang University is a university in Hangzhou, China. It is located behind the West Lake, a famous tourist attraction in the city."
                )
        return direction_text_batch

    def forward(self, **kwargs):
        self.batch = kwargs["batch"]
        batch = kwargs["batch"]
        self.phrase = batch["phrase"]
        self.file_name = batch["file_path"]
        texts = batch["phrase"]
        reference_obj_name = batch["reference_obj"]
        direction_text = self.update_text(batch["direction_text"])
        anchor_pos = batch["anchor_position"]  # 新增 anchor position
        output = {}

        # encode text(object name)
        l_enc, l_emb, l_mask = self.encode_text(
            direction_text
        )  # encode the direction text
        l_input = l_emb if "word" in self.lang_fusion_type else l_enc
        l_input = l_input.to(dtype=torch.float32)
        batch["target_query"] = l_input

        d_enc, d_emb, d_mask = self.direction_encoder(
            direction_text
        )  # encode the direction text
        d_input = d_enc.to(dtype=torch.float32)

        # point cloud
        scene_pcs = batch["fps_points_scene"]
        obj_pcs = batch["fps_points_scene"]

        # pointnet_1: extract the geometric affordance feat.
        obj_pcs = obj_pcs - anchor_pos.unsqueeze(1)  # [B, Num_points, 3]
        x_geo = self.geoafford_module(
            scene_pcs, obj_pcs
        )  # [B, C, Num_pts] [B, 48, Num_pts]

        # CLIP: extract the rgb feat.
        x_rgb = self.clipunet_module(batch=batch)  # [B, C_rgbfeat, H, W] [B, 64, H, W]

        # merge 得到场景特征
        x_align = self.concate(x_rgb["affordance"], scene_pcs, self.instrics).permute(
            0, 2, 1
        )  # [B, 64, 2048]
        x_scene = torch.cat(
            (scene_pcs.permute(0, 2, 1), obj_pcs.permute(0, 2, 1), x_align, x_geo),
            dim=1,
        )  # 场景特征 [B, 118, 2048]
        # each position minues anchor position
        x_scene = x_scene.permute(0, 2, 1)  # [B, 2048, 118]

        # 处理anchor position 特征
        # input: B*3
        x_anchor_position = self.anchor_mlp(anchor_pos)  # output: [B, 256]

        # 处理 direction text
        x_anchor_and_text = torch.cat(
            (x_anchor_position, d_input), dim=1
        )  # [B,  (256 + 1024) ]
        x_anchor_and_text = x_anchor_and_text.unsqueeze(1)  # [B, 1, 256+1024]
        x_anchor_and_text = self.cond_mlp(x_anchor_and_text)  # [B, 256]

        # perceive 处理x_scene and x_anchor_and_text
        x = self.feat_perceiver(x_scene, x_anchor_and_text)  # [B, 2048, 256]

        x = self.fusion_point_moudule(x)
        # x = x.permute(0,2,1)

        output["affordance"] = x
        self.model_ouput = x
        self.model_pc = scene_pcs

        return output

    def generate_heatmap(self, epoch):
        "generate heatmap during training"
        self.pc_heatmap_list = []
        pcs = self.model_pc  # [B, Num_points, 3]
        feat = self.model_ouput  # [B, Num_points, 1]
        # Normalize feat to [0, 1] for each batch independently
        feat = feat.sigmoid()
        # Normalize feat to [0, 1] for eac
        min_feat = feat.min(dim=1, keepdim=True)[0]  # Find min in each batch
        max_feat = feat.max(dim=1, keepdim=True)[0]  # Find max in each batch
        normalized_feat = (feat - min_feat) / (
            max_feat - min_feat + 1e-6
        )  # Avoid division by zero
        # Convert normalized features to RGB using turbo colormap
        turbo_colormap = cm.get_cmap("turbo")  # Load turbo colormap with 256 levels
        normalized_feat_np = (
            normalized_feat.squeeze(-1).cpu().detach().numpy()
        )  # Shape: [B, Num_points]
        # Apply turbo colormap to normalized features (converts values to RGB)
        color_maps = turbo_colormap(normalized_feat_np)[
            :, :, :3
        ]  # Shape: [B, Num_points, 3], ignore alpha
        # Generate point cloud with Open3D
        for i in range(pcs.shape[0]):  # Iterate over batch
            points = pcs[i].cpu().numpy()  # Shape: [Num_points, 3]
            colors = color_maps[i]  # Shape: [Num_points, 3]

            # Create Open3D point cloud object
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            self.pc_heatmap_list.append(pcd)

    def pcdheatmap2img(self, is_mask=False):
        pcs_heatmap_list = self.pc_heatmap_list
        img_rgb_list = self.batch["image"]
        img_heatmap_list = []
        assert len(pcs_heatmap_list) == len(img_rgb_list)
        cam_rotation_matrix = np.array([[1, 0, 0], [0, 0.8, -0.6], [0, 0.6, 0.8]])
        intrinsics = np.array(
            [[591.0125, 0.0, 322.525], [0.0, 590.16775, 244.11084], [0.0, 0.0, 1.0]]
        )
        for i in range(len(pcs_heatmap_list)):
            pc = pcs_heatmap_list[i]
            img_rgb = img_rgb_list[i]
            image_np = img_rgb.cpu().detach().numpy().astype(np.float32)

            image_tensor = (
                torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0)
            )  # [B, C, H, W]
            pc_points = np.array(pc.points)

            if is_mask:
                pc_points = pc_points @ cam_rotation_matrix

            query_points = torch.tensor(pc_points, dtype=torch.float32).unsqueeze(
                0
            )  # [B, N, 3]

            pc_colors = np.array(pc.colors)
            query_colors = torch.tensor(pc_colors, dtype=torch.float32).unsqueeze(
                0
            )  # [B, N, 3]
            sample_ids = np.random.choice(np.arange(query_colors.size(1)), 2048)

            query_colors = query_colors[:, sample_ids]
            query_points = query_points[:, sample_ids]
            image_tensor = image_tensor.permute(0, 2, 3, 1)
            projector = ProjectColorOntoImage_v2()
            output_image = projector(
                image_grid=image_tensor,
                query_points=query_points,
                query_colors=query_colors,
                intrinsics=intrinsics,
            )

            for i, img in enumerate(output_image):
                if img.ndim == 3:

                    color_image = T.ToPILImage()(image_tensor[i].cpu())
                    pil_image = T.ToPILImage()(img.cpu())

                    image_np = np.clip(pil_image, 0, 255)

                    color_image_np = np.floor(color_image)
                    color_image_np = np.clip(color_image_np, 0, 255)
                    color_image_np = np.uint8(color_image_np)

                    image_np = cv2.addWeighted(image_np, 0.3, color_image_np, 0.7, 0.0)
                    pil_image = Image.fromarray(np.uint8(image_np))
                    img_heatmap_list.append(pil_image)
        return img_heatmap_list, self.file_name, self.phrase


@registry.register_affordance_model(name="GeoL_net_v7")
class GeoL_net_v7(nn.Module):
    """
    GeoL_net_v6:
        1. adapt to 8192 points in the scene point cloud
        2. use PointMLP from afford motion
        3. add the distance field
    """

    def __init__(self, input_shape, target_input_shape):
        super().__init__()
        self.ContactMLP = ContactMLP(
            contact_dim=64, point_feat_dim=64, text_feat_dim=256
        )
        self.clipunet_module = CLIPUNet(
            input_shape=input_shape,
            target_input_shape=target_input_shape,
            output_dim=64,
        )  # 6kw parameters
        self.concate = FeatureConcat()
        self.device = "cuda"  # cpu for dataset
        self.lang_fusion_type = "mult"  # hard code from CLIPlingunet
        self._load_clip()
        self.instrics = np.array(
            [[591.0125, 0.0, 322.525], [0.0, 590.16775, 244.11084], [0.0, 0.0, 1.0]]
        )
        # self.fusion_point_moudule = FusionPointLayer(input_dim=256) #0.3kw
        self.direction_mlp = nn.Sequential(
            nn.Linear(8, 256),
            nn.TransformerEncoderLayer(
                d_model=256, nhead=4, dim_feedforward=512, batch_first=True
            ),
        )

        self.dist_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.TransformerEncoderLayer(
                d_model=64, nhead=4, dim_feedforward=128, batch_first=True
            ),
        )

        self.obj_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.TransformerEncoderLayer(
                d_model=64, nhead=4, dim_feedforward=128, batch_first=True
            ),
        )

        self.anchor_mlp = nn.Sequential(
            nn.Linear(3, 256),
            nn.TransformerEncoderLayer(
                d_model=256, nhead=4, dim_feedforward=512, batch_first=True
            ),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.TransformerEncoderLayer(
                d_model=256, nhead=4, dim_feedforward=512, batch_first=True
            ),
        )
        self.fusion_point_moudule = nn.Sequential(
            nn.Linear(256, 256),
            nn.TransformerEncoderLayer(
                d_model=256, nhead=8, dim_feedforward=512, batch_first=True
            ),
            nn.Linear(256, 64),
            nn.TransformerEncoderLayer(
                d_model=64, nhead=4, dim_feedforward=128, batch_first=True
            ),
            nn.Linear(64, 1),
        )

        self.feat_perceiver = FeaturePerceiver(
            transition_dim=134, condition_dim=256, time_emb_dim=32
        )

        self.directions = [
            "Left",
            "Right",
            "Front",
            "Behind",
            "Left Front",
            "Right Front",
            "Left Behind",
            "Right Behind",
        ]
        self.vocab = {word: idx for idx, word in enumerate(self.directions)}

    def _load_clip(self):
        model, _ = load_clip("RN50", device=self.device)
        self.clip_rn50 = build_model(model.state_dict()).to(self.device)  # 10kw frozen
        del model
        # Frozen clip
        for param in self.clip_rn50.parameters():
            param.requires_grad = False

    def encode_text(self, x):
        with torch.no_grad():
            tokens = tokenize(x).to(self.device)
            text_feat, text_emb = self.clip_rn50.encode_text_with_embeddings(tokens)

        text_mask = torch.where(tokens == 0, tokens, 1)  # [1, max_token_len]
        return text_feat, text_emb, text_mask

    def calculate_distance_field(self, scene_pcs, anchor_pos):
        """
        calculate the distance field from anchor position
        """
        sig = 1.0
        diff = scene_pcs - anchor_pos.unsqueeze(1)  # [B, Num_points, 3]
        distances = torch.norm(diff, dim=2, keepdim=True)  # [B, Num_points, 1]
        dist_filed = torch.exp(-(distances**2) / (2 * sig**2))
        return dist_filed

    def forward(self, **kwargs):
        self.batch = kwargs["batch"]
        batch = kwargs["batch"]
        self.phrase = batch["phrase"]
        self.file_name = batch["file_path"]
        texts = batch["phrase"]
        reference_obj_name = batch["reference_obj"]
        direction_text = batch["direction_text"]
        anchor_pos = batch["anchor_position"]  # [B, 3]
        dist_field = self.calculate_distance_field(
            batch["fps_points_scene"], anchor_pos
        )  # [B, Num_points, 1]
        dist_field = self.dist_mlp(dist_field)  # [B, Num_points, 64]

        output = {}

        # encode text(object name)
        l_enc, l_emb, l_mask = self.encode_text(
            direction_text
        )  # use direction input as target query
        l_input = l_emb if "word" in self.lang_fusion_type else l_enc
        l_input = l_input.to(dtype=torch.float32)
        batch["target_query"] = l_input  # [B, 1024]
        # encode text(direction) one hot
        direction_inputs_indices = torch.tensor(
            list(map(self.vocab.get, batch["direction_text"]))
        )
        d_input = torch.eye(len(self.vocab))[
            direction_inputs_indices
        ].cuda()  # [batch_size, 8] one-hot

        d_input = self.direction_mlp(d_input)  # direction: [B, 256]

        # point cloud
        scene_pcs = batch["fps_points_scene"]
        obj_pcs = batch["fps_points_scene"] - anchor_pos.unsqueeze(
            1
        )  # [B, Num_points, 3]

        obj_pcs_feat = self.obj_mlp(obj_pcs)  # [B, Num_points, 64]
        x_geo = self.ContactMLP(
            dist_field, obj_pcs_feat, d_input.unsqueeze(1)
        )  # [B, Num_points, 64]

        # CLIP: extract the rgb feat.
        x_rgb = self.clipunet_module(batch=batch)  # [B, C_rgbfeat, H, W] [B, 64, H, W]

        # merge 得到场景特征
        x_align = self.concate(x_rgb["affordance"], scene_pcs, self.instrics).permute(
            0, 2, 1
        )  # [B, 64, num_points]
        x_scene = torch.cat(
            (scene_pcs, obj_pcs, x_align.permute(0, 2, 1), x_geo), dim=2
        )  # 场景特征 [B, num_points, 138]
        # each position minues anchor position
        # x_scene = x_scene.permute(0,2,1) # [B, 2048, 118]

        # 处理anchor position 特征
        # input: B*3
        x_anchor_position = self.anchor_mlp(anchor_pos)  # output: [B, 256]

        # 处理 direction text
        x_anchor_and_text = torch.cat(
            (x_anchor_position, d_input), dim=1
        )  # [B,  (256 + 256) ]
        x_anchor_and_text = x_anchor_and_text.unsqueeze(1)  # [B, 1, 256]
        x_anchor_and_text = self.cond_mlp(x_anchor_and_text)  # [B, 256]

        # perceive 处理x_scene and x_anchor_and_text
        x = self.feat_perceiver(x_scene, x_anchor_and_text)  # [B, 2048, 256]

        x = self.fusion_point_moudule(x)
        # x = x.permute(0,2,1)

        output["affordance"] = x
        self.model_ouput = x
        self.model_pc = scene_pcs

        return output

    def generate_heatmap(self, epoch):
        "generate heatmap during training"
        self.pc_heatmap_list = []
        pcs = self.model_pc  # [B, Num_points, 3]
        feat = self.model_ouput  # [B, Num_points, 1]
        # Normalize feat to [0, 1] for each batch independently
        feat = feat.sigmoid()
        # Normalize feat to [0, 1] for eac
        min_feat = feat.min(dim=1, keepdim=True)[0]  # Find min in each batch
        max_feat = feat.max(dim=1, keepdim=True)[0]  # Find max in each batch
        normalized_feat = (feat - min_feat) / (
            max_feat - min_feat + 1e-6
        )  # Avoid division by zero
        # Convert normalized features to RGB using turbo colormap
        turbo_colormap = cm.get_cmap("turbo")  # Load turbo colormap with 256 levels
        normalized_feat_np = (
            normalized_feat.squeeze(-1).cpu().detach().numpy()
        )  # Shape: [B, Num_points]
        # Apply turbo colormap to normalized features (converts values to RGB)
        color_maps = turbo_colormap(normalized_feat_np)[
            :, :, :3
        ]  # Shape: [B, Num_points, 3], ignore alpha
        # Generate point cloud with Open3D
        for i in range(pcs.shape[0]):  # Iterate over batch
            points = pcs[i].cpu().numpy()  # Shape: [Num_points, 3]
            colors = color_maps[i]  # Shape: [Num_points, 3]

            # Create Open3D point cloud object
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            self.pc_heatmap_list.append(pcd)

    def pcdheatmap2img(self, is_mask=False):
        pcs_heatmap_list = self.pc_heatmap_list
        img_rgb_list = self.batch["image"]
        img_heatmap_list = []
        assert len(pcs_heatmap_list) == len(img_rgb_list)
        cam_rotation_matrix = np.array([[1, 0, 0], [0, 0.8, -0.6], [0, 0.6, 0.8]])
        intrinsics = np.array(
            [[591.0125, 0.0, 322.525], [0.0, 590.16775, 244.11084], [0.0, 0.0, 1.0]]
        )
        for i in range(len(pcs_heatmap_list)):
            pc = pcs_heatmap_list[i]
            img_rgb = img_rgb_list[i]
            image_np = img_rgb.cpu().detach().numpy().astype(np.float32)

            image_tensor = (
                torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0)
            )  # [B, C, H, W]
            pc_points = np.array(pc.points)

            if is_mask:
                pc_points = pc_points @ cam_rotation_matrix

            query_points = torch.tensor(pc_points, dtype=torch.float32).unsqueeze(
                0
            )  # [B, N, 3]

            pc_colors = np.array(pc.colors)
            query_colors = torch.tensor(pc_colors, dtype=torch.float32).unsqueeze(
                0
            )  # [B, N, 3]
            sample_ids = np.random.choice(np.arange(query_colors.size(1)), 2048)

            query_colors = query_colors[:, sample_ids]
            query_points = query_points[:, sample_ids]
            image_tensor = image_tensor.permute(0, 2, 3, 1)
            projector = ProjectColorOntoImage_v2()
            output_image = projector(
                image_grid=image_tensor,
                query_points=query_points,
                query_colors=query_colors,
                intrinsics=intrinsics,
            )

            for i, img in enumerate(output_image):
                if img.ndim == 3:

                    color_image = T.ToPILImage()(image_tensor[i].cpu())
                    pil_image = T.ToPILImage()(img.cpu())

                    image_np = np.clip(pil_image, 0, 255)

                    color_image_np = np.floor(color_image)
                    color_image_np = np.clip(color_image_np, 0, 255)
                    color_image_np = np.uint8(color_image_np)

                    image_np = cv2.addWeighted(image_np, 0.3, color_image_np, 0.7, 0.0)
                    pil_image = Image.fromarray(np.uint8(image_np))
                    img_heatmap_list.append(pil_image)
        return img_heatmap_list, self.file_name, self.phrase


@registry.register_affordance_model(name="GeoL_net_v8")
class GeoL_net_v8(nn.Module):
    """
    GeoL_net_v8:
        1. use double perceiver structure
    """

    def __init__(self, input_shape, target_input_shape):
        super().__init__()
        self.geoafford_module = GeoAffordModule(feat_dim=16)  # 0.3kw
        self.clipunet_module = CLIPUNet(
            input_shape=input_shape,
            target_input_shape=target_input_shape,
            output_dim=64,
        )  # 6kw parameters
        self.concate = FeatureConcat()
        self.device = "cuda"  # cpu for dataset
        self.lang_fusion_type = "mult"  # hard code from CLIPlingunet
        self._load_clip()
        self.instrics = np.array(
            [[591.0125, 0.0, 322.525], [0.0, 590.16775, 244.11084], [0.0, 0.0, 1.0]]
        )

        # self.instrics = np.array([[607.0125 ,   0.     , 636.525  ], [  0.     , 607.16775, 367.11084], [  0.     ,   0.     ,   1.     ]]) # for real world data
        # self.fusion_point_moudule = FusionPointLayer(input_dim=256) #0.3kw
        self.direction_mlp = nn.Sequential(
            nn.Linear(1024, 256),
            nn.TransformerEncoderLayer(
                d_model=256, nhead=4, dim_feedforward=512, batch_first=True
            ),
        )

        self.anchor_mlp = nn.Sequential(
            nn.Linear(3, 256),
            nn.TransformerEncoderLayer(
                d_model=256, nhead=4, dim_feedforward=512, batch_first=True
            ),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.TransformerEncoderLayer(
                d_model=256, nhead=4, dim_feedforward=512, batch_first=True
            ),
        )
        self.fusion_point_moudule = nn.Sequential(
            nn.Linear(256, 256),
            nn.TransformerEncoderLayer(
                d_model=256, nhead=8, dim_feedforward=512, batch_first=True
            ),
            nn.Linear(256, 64),
            nn.TransformerEncoderLayer(
                d_model=64, nhead=4, dim_feedforward=256, batch_first=True
            ),
            nn.Linear(64, 1),
        )

        self.feat_perceiver = FeaturePerceiver(
            transition_dim=118, condition_dim=256, time_emb_dim=32
        )

        self.feat_perceiver_2 = FeaturePerceiver(
            transition_dim=256, condition_dim=256, time_emb_dim=32
        )

    def _load_clip(self):
        model, _ = load_clip("RN50", device=self.device)
        self.clip_rn50 = build_model(model.state_dict()).to(self.device)  # 10kw frozen
        del model
        # Frozen clip
        for param in self.clip_rn50.parameters():
            param.requires_grad = False

    def encode_text(self, x):
        with torch.no_grad():
            tokens = tokenize(x).to(self.device)
            text_feat, text_emb = self.clip_rn50.encode_text_with_embeddings(tokens)

        text_mask = torch.where(tokens == 0, tokens, 1)  # [1, max_token_len]
        return text_feat, text_emb, text_mask

    def direction_encoder(self, x):
        """
        encode the direction text through embedding
        """
        with torch.no_grad():
            tokens = tokenize(x).to(self.device)
            text_feat, text_emb = self.clip_rn50.encode_text_with_embeddings(tokens)
        text_mask = torch.where(tokens == 0, tokens, 1)  # [1, max_token_len]
        return text_feat, text_emb, text_mask

        # self.directions = ["Left", "Right", "Front", "Behind", "Left Front", "Right Front", "Left Behind", "Right Behind"]
        # self.vocab = {word: idx for idx, word in enumerate(self.directions)}

    def update_text(self, direction_text_batch):
        """
        update the direction text
        """
        for i, text in enumerate(direction_text_batch):
            if text == "Left":
                direction_text_batch[i] = (
                    "Left, After a long and exhausting day at work, I left the office feeling both relieved and tired, looking forward to finally going home, where I could rest and unwind peacefully\
                    The left part of my car is damaged. I need to take it to the repair shop to get it fixed."
                )
            elif text == "Right":
                direction_text_batch[i] = (
                    "Right, The right side of the brain is responsible for creativity. do you like to draw or paint? If so, you are using the right side of your brain\
                    Blue is the color of the sky on a clear day. The sky is blue because of the way the Earth's atmosphere scatters light from the sun."
                )
            elif text == "Front":
                direction_text_batch[i] = (
                    "Front is the best. The front of the house is where the garden is located. The garden is a beautiful place to relax and enjoy the outdoors.\
                    TUM is a university in Munich, Germany. It is located in the front of the city, near the city center. "
                )
            elif text == "Behind":
                direction_text_batch[i] = (
                    "Behind hahaha The cat is hiding behind the couch. I can see its tail sticking out from behind the couch.\
                    Zhejiang University is a university in Hangzhou, China. It is located behind the West Lake, a famous tourist attraction in the city."
                )
            elif text == "Left Front":
                direction_text_batch[i] = (
                    "Left Front yes it is. The Left Front is a political party in the United States. It was founded in 2004 by a group of former members of the Democratic Party."
                )
            elif text == "Right Front":
                direction_text_batch[i] = (
                    "Right Front well done. The Right Front china is a desktop publishing software application developed by Adobe Systems. It is used to create documents, such as newsletters, brochures, and flyers."
                )
            elif text == "Left Behind":
                direction_text_batch[i] = (
                    "Left Behind is a novel by Tim LaHaye and Jerry B. Jenkins that was first published in 1995. It is the first book in the Left Behind series, which has sold over 65 million copies worldwide."
                )
            elif text == "Right Behind":
                direction_text_batch[i] = (
                    "Right Behind is a song by the American rock band Nine Inch Nails. It was released as the third single from their second studio album, The Downward Spiral (1994)."
                )
        return direction_text_batch

    def forward(self, **kwargs):
        self.batch = kwargs["batch"]
        batch = kwargs["batch"]
        self.phrase = batch["phrase"]
        self.file_name = batch["file_path"]
        texts = batch["phrase"]
        reference_obj_name = batch["reference_obj"]
        direction_text = self.update_text(batch["direction_text"])
        anchor_pos = batch["anchor_position"]  # 新增 anchor position
        output = {}

        # encode text(object name)
        l_enc, l_emb, l_mask = self.encode_text(
            reference_obj_name
        )  # encode the direction text
        l_input = l_emb if "word" in self.lang_fusion_type else l_enc
        l_input = l_input.to(dtype=torch.float32)
        batch["target_query"] = l_input

        d_enc, d_emb, d_mask = self.direction_encoder(
            direction_text
        )  # encode the direction text
        d_input = d_enc.to(dtype=torch.float32)
        d_input = self.direction_mlp(d_input)

        # point cloud
        scene_pcs = batch["fps_points_scene"]
        obj_pcs = batch["fps_points_scene"]

        # pointnet_1: extract the geometric affordance feat.
        obj_pcs = obj_pcs - anchor_pos.unsqueeze(1)  # [B, Num_points, 3]
        x_geo = self.geoafford_module(
            scene_pcs, obj_pcs
        )  # [B, C, Num_pts] [B, 48, Num_pts]

        # CLIP: extract the rgb feat.
        x_rgb = self.clipunet_module(batch=batch)  # [B, C_rgbfeat, H, W] [B, 64, H, W]

        # merge 得到场景特征
        x_align = self.concate(x_rgb["affordance"], scene_pcs, self.instrics).permute(
            0, 2, 1
        )  # [B, 64, 2048]
        x_scene = torch.cat(
            (scene_pcs.permute(0, 2, 1), obj_pcs.permute(0, 2, 1), x_align, x_geo),
            dim=1,
        )  # 场景特征 [B, 118, 2048]
        # each position minues anchor position
        x_scene = x_scene.permute(0, 2, 1)  # [B, 2048, 118]

        # 处理anchor position 特征
        # input: B*3
        x_anchor_position = self.anchor_mlp(anchor_pos)  # output: [B, 256]

        # 处理 direction text
        x_anchor_and_text = torch.cat(
            (x_anchor_position, d_input), dim=1
        )  # [B,  (256 + 256) ]
        x_anchor_and_text = x_anchor_and_text.unsqueeze(1)  # [B, 1, 256+256]
        x_anchor_and_text = self.cond_mlp(x_anchor_and_text)  # [B, 256]

        # perceive 处理x_scene and x_anchor_and_text
        x = self.feat_perceiver(x_scene, x_anchor_and_text)  # [B, 2048, 256]

        # 2nd perceiver 处理 x and d_input
        x = self.feat_perceiver_2(x, d_input.unsqueeze(1))

        x = self.fusion_point_moudule(x)
        # x = x.permute(0,2,1)

        output["affordance"] = x
        self.model_ouput = x
        self.model_pc = scene_pcs

        return output


@registry.register_affordance_model(name="GeoL_net_v9")
class GeoL_net_v8(nn.Module):
    """
    GeoL_net_v9:
        1. use data based on kinect cfg
        2. camera instrics is different
        3. image size is different
        4. intrinscis as input
    """

    def __init__(self, input_shape, target_input_shape, intrinsics=None):
        super().__init__()
        self.geoafford_module = GeoAffordModule(feat_dim=16)  # 0.3kw
        self.clipunet_module = CLIPUNet(
            input_shape=input_shape,
            target_input_shape=target_input_shape,
            output_dim=64,
        )  # 6kw parameters
        self.concate = FeatureConcat()
        self.device = "cuda"  # cpu for dataset
        self.lang_fusion_type = "mult"  # hard code from CLIPlingunet
        self._load_clip()
        if intrinsics is None:
            self.intrinsics = np.array(
                [
                    [607.09912 / 2, 0.0, 636.85083 / 2],
                    [0.0, 607.05212 / 2, 367.35952 / 2],
                    [0.0, 0.0, 1.0],
                ]
            )
        else:
            self.intrinsics = intrinsics

        # self.instrics = np.array([[607.0125 ,   0.     , 636.525  ], [  0.     , 607.16775, 367.11084], [  0.     ,   0.     ,   1.     ]]) # for real world data
        # self.fusion_point_moudule = FusionPointLayer(input_dim=256) #0.3kw
        self.direction_mlp = nn.Sequential(
            nn.Linear(1024, 256),
            nn.TransformerEncoderLayer(
                d_model=256, nhead=4, dim_feedforward=512, batch_first=True
            ),
        )

        self.anchor_mlp = nn.Sequential(
            nn.Linear(3, 256),
            nn.TransformerEncoderLayer(
                d_model=256, nhead=4, dim_feedforward=512, batch_first=True
            ),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.TransformerEncoderLayer(
                d_model=256, nhead=4, dim_feedforward=512, batch_first=True
            ),
        )
        self.fusion_point_moudule = nn.Sequential(
            nn.Linear(256, 256),
            nn.TransformerEncoderLayer(
                d_model=256, nhead=8, dim_feedforward=512, batch_first=True
            ),
            nn.Linear(256, 64),
            nn.TransformerEncoderLayer(
                d_model=64, nhead=4, dim_feedforward=256, batch_first=True
            ),
            nn.Linear(64, 1),
        )

        self.feat_perceiver = FeaturePerceiver(
            transition_dim=118, condition_dim=256, time_emb_dim=32
        )

        self.feat_perceiver_2 = FeaturePerceiver(
            transition_dim=256, condition_dim=256, time_emb_dim=32
        )

    def _load_clip(self):
        model, _ = load_clip("RN50", device=self.device)
        self.clip_rn50 = build_model(model.state_dict()).to(self.device)  # 10kw frozen
        del model
        # Frozen clip
        for param in self.clip_rn50.parameters():
            param.requires_grad = False

    def encode_text(self, x):
        with torch.no_grad():
            tokens = tokenize(x).to(self.device)
            text_feat, text_emb = self.clip_rn50.encode_text_with_embeddings(tokens)

        text_mask = torch.where(tokens == 0, tokens, 1)  # [1, max_token_len]
        return text_feat, text_emb, text_mask

    def direction_encoder(self, x):
        """
        encode the direction text through embedding
        """
        with torch.no_grad():
            tokens = tokenize(x).to(self.device)
            text_feat, text_emb = self.clip_rn50.encode_text_with_embeddings(tokens)
        text_mask = torch.where(tokens == 0, tokens, 1)  # [1, max_token_len]
        return text_feat, text_emb, text_mask

        # self.directions = ["Left", "Right", "Front", "Behind", "Left Front", "Right Front", "Left Behind", "Right Behind"]
        # self.vocab = {word: idx for idx, word in enumerate(self.directions)}

    def update_text(self, direction_text_batch):
        """
        update the direction text
        """
        for i, text in enumerate(direction_text_batch):
            if text == "Left":
                direction_text_batch[i] = (
                    "Left, After a long and exhausting day at work, I left the office feeling both relieved and tired, looking forward to finally going home, where I could rest and unwind peacefully\
                    The left part of my car is damaged. I need to take it to the repair shop to get it fixed."
                )
            elif text == "Right":
                direction_text_batch[i] = (
                    "Right, The right side of the brain is responsible for creativity. do you like to draw or paint? If so, you are using the right side of your brain\
                    Blue is the color of the sky on a clear day. The sky is blue because of the way the Earth's atmosphere scatters light from the sun."
                )
            elif text == "Front":
                direction_text_batch[i] = (
                    "Front is the best. The front of the house is where the garden is located. The garden is a beautiful place to relax and enjoy the outdoors.\
                    TUM is a university in Munich, Germany. It is located in the front of the city, near the city center. "
                )
            elif text == "Behind":
                direction_text_batch[i] = (
                    "Behind hahaha The cat is hiding behind the couch. I can see its tail sticking out from behind the couch.\
                    Zhejiang University is a university in Hangzhou, China. It is located behind the West Lake, a famous tourist attraction in the city."
                )
            elif text == "Left Front":
                direction_text_batch[i] = (
                    "Left Front yes it is. The Left Front is a political party in the United States. It was founded in 2004 by a group of former members of the Democratic Party."
                )
            elif text == "Right Front":
                direction_text_batch[i] = (
                    "Right Front well done. The Right Front china is a desktop publishing software application developed by Adobe Systems. It is used to create documents, such as newsletters, brochures, and flyers."
                )
            elif text == "Left Behind":
                direction_text_batch[i] = (
                    "Left Behind is a novel by Tim LaHaye and Jerry B. Jenkins that was first published in 1995. It is the first book in the Left Behind series, which has sold over 65 million copies worldwide."
                )
            elif text == "Right Behind":
                direction_text_batch[i] = (
                    "Right Behind is a song by the American rock band Nine Inch Nails. It was released as the third single from their second studio album, The Downward Spiral (1994)."
                )
        return direction_text_batch

    def forward(self, **kwargs):
        self.batch = kwargs["batch"]
        batch = kwargs["batch"]
        self.phrase = batch["phrase"]
        self.file_name = batch["file_path"]
        texts = batch["phrase"]
        reference_obj_name = batch["reference_obj"]
        direction_text = self.update_text(batch["direction_text"])
        anchor_pos = batch["anchor_position"]  # 新增 anchor position
        output = {}

        # encode text(object name)
        l_enc, l_emb, l_mask = self.encode_text(
            reference_obj_name
        )  # encode the direction text
        l_input = l_emb if "word" in self.lang_fusion_type else l_enc
        l_input = l_input.to(dtype=torch.float32)
        batch["target_query"] = l_input

        d_enc, d_emb, d_mask = self.direction_encoder(
            direction_text
        )  # encode the direction text
        d_input = d_enc.to(dtype=torch.float32)
        d_input = self.direction_mlp(d_input)

        # point cloud
        scene_pcs = batch["fps_points_scene"]
        obj_pcs = batch["fps_points_scene"]

        # pointnet_1: extract the geometric affordance feat.
        obj_pcs = obj_pcs - anchor_pos.unsqueeze(1)  # [B, Num_points, 3]
        x_geo = self.geoafford_module(
            scene_pcs, obj_pcs
        )  # [B, C, Num_pts] [B, 48, Num_pts]

        # CLIP: extract the rgb feat.
        x_rgb = self.clipunet_module(batch=batch)  # [B, C_rgbfeat, H, W] [B, 64, H, W]

        # merge 得到场景特征
        x_align = self.concate(x_rgb["affordance"], scene_pcs, self.intrinsics).permute(
            0, 2, 1
        )  # [B, 64, 2048]
        x_scene = torch.cat(
            (scene_pcs.permute(0, 2, 1), obj_pcs.permute(0, 2, 1), x_align, x_geo),
            dim=1,
        )  # 场景特征 [B, 118, 2048]
        # each position minues anchor position
        x_scene = x_scene.permute(0, 2, 1)  # [B, 2048, 118]

        # 处理anchor position 特征
        # input: B*3
        x_anchor_position = self.anchor_mlp(anchor_pos)  # output: [B, 256]

        # 处理 direction text
        x_anchor_and_text = torch.cat(
            (x_anchor_position, d_input), dim=1
        )  # [B,  (256 + 256) ]
        x_anchor_and_text = x_anchor_and_text.unsqueeze(1)  # [B, 1, 256+256]
        x_anchor_and_text = self.cond_mlp(x_anchor_and_text)  # [B, 256]

        # perceive 处理x_scene and x_anchor_and_text
        x = self.feat_perceiver(x_scene, x_anchor_and_text)  # [B, 2048, 256]

        # 2nd perceiver 处理 x and d_input
        x = self.feat_perceiver_2(x, d_input.unsqueeze(1))

        x_feat = x
        x = self.fusion_point_moudule(x)
        # x = x.permute(0,2,1)

        output["affordance"] = x
        output["affordance_feat"] = x_feat
        self.model_ouput = x
        self.model_pc = scene_pcs

        return output

@registry.register_affordance_encoder(name="AffordanceEncoder")
class AffordanceEncoder(nn.Module):
    """
    AffordanceEncoder used in the diffusion model to get the map features
    Compared to v9, skip the transformer encoder layer in the end , output dim is 64
    """

    def __init__(self, input_shape, target_input_shape, intrinsics=None):
        super().__init__()
        self.geoafford_module = GeoAffordModule(feat_dim=16)  # 0.3kw
        self.clipunet_module = CLIPUNet(
            input_shape=input_shape,
            target_input_shape=target_input_shape,
            output_dim=64,
        )  # 6kw parameters
        self.concate = FeatureConcat()
        self.device = "cuda"  # cpu for dataset
        self.lang_fusion_type = "mult"  # hard code from CLIPlingunet
        self._load_clip()
        if intrinsics is None:
            self.intrinsics = np.array(
                [
                    [607.09912 / 2, 0.0, 636.85083 / 2],
                    [0.0, 607.05212 / 2, 367.35952 / 2],
                    [0.0, 0.0, 1.0],
                ]
            )
        else:
            self.intrinsics = intrinsics

        # self.instrics = np.array([[607.0125 ,   0.     , 636.525  ], [  0.     , 607.16775, 367.11084], [  0.     ,   0.     ,   1.     ]]) # for real world data
        # self.fusion_point_moudule = FusionPointLayer(input_dim=256) #0.3kw
        self.direction_mlp = nn.Sequential(
            nn.Linear(1024, 256),
            nn.TransformerEncoderLayer(
                d_model=256, nhead=4, dim_feedforward=512, batch_first=True
            ),
        )

        self.anchor_mlp = nn.Sequential(
            nn.Linear(3, 256),
            nn.TransformerEncoderLayer(
                d_model=256, nhead=4, dim_feedforward=512, batch_first=True
            ),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.TransformerEncoderLayer(
                d_model=256, nhead=4, dim_feedforward=512, batch_first=True
            ),
        )
        self.fusion_point_moudule = nn.Sequential(
            nn.Linear(256, 256),
            nn.TransformerEncoderLayer(
                d_model=256, nhead=8, dim_feedforward=512, batch_first=True
            ),
            nn.Linear(256, 64),
            nn.TransformerEncoderLayer(
                d_model=64, nhead=4, dim_feedforward=256, batch_first=True
            ),
            nn.Linear(64, 1),
        )

        self.feat_perceiver = FeaturePerceiver(
            transition_dim=118, condition_dim=256, time_emb_dim=32
        )

        self.feat_perceiver_2 = FeaturePerceiver(
            transition_dim=256, condition_dim=256, time_emb_dim=32
        )

    def _load_clip(self):
        model, _ = load_clip("RN50", device=self.device)
        self.clip_rn50 = build_model(model.state_dict()).to(self.device)  # 10kw frozen
        del model
        # Frozen clip
        for param in self.clip_rn50.parameters():
            param.requires_grad = False

    def encode_text(self, x):
        with torch.no_grad():
            tokens = tokenize(x).to(self.device)
            text_feat, text_emb = self.clip_rn50.encode_text_with_embeddings(tokens)

        text_mask = torch.where(tokens == 0, tokens, 1)  # [1, max_token_len]
        return text_feat, text_emb, text_mask

    def direction_encoder(self, x):
        """
        encode the direction text through embedding
        """
        with torch.no_grad():
            tokens = tokenize(x).to(self.device)
            text_feat, text_emb = self.clip_rn50.encode_text_with_embeddings(tokens)
        text_mask = torch.where(tokens == 0, tokens, 1)  # [1, max_token_len]
        return text_feat, text_emb, text_mask

        # self.directions = ["Left", "Right", "Front", "Behind", "Left Front", "Right Front", "Left Behind", "Right Behind"]
        # self.vocab = {word: idx for idx, word in enumerate(self.directions)}

    def update_text(self, direction_text_batch):
        """
        update the direction text
        """
        for i, text in enumerate(direction_text_batch):
            if text == "Left":
                direction_text_batch[i] = (
                    "Left, After a long and exhausting day at work, I left the office feeling both relieved and tired, looking forward to finally going home, where I could rest and unwind peacefully\
                    The left part of my car is damaged. I need to take it to the repair shop to get it fixed."
                )
            elif text == "Right":
                direction_text_batch[i] = (
                    "Right, The right side of the brain is responsible for creativity. do you like to draw or paint? If so, you are using the right side of your brain\
                    Blue is the color of the sky on a clear day. The sky is blue because of the way the Earth's atmosphere scatters light from the sun."
                )
            elif text == "Front":
                direction_text_batch[i] = (
                    "Front is the best. The front of the house is where the garden is located. The garden is a beautiful place to relax and enjoy the outdoors.\
                    TUM is a university in Munich, Germany. It is located in the front of the city, near the city center. "
                )
            elif text == "Behind":
                direction_text_batch[i] = (
                    "Behind hahaha The cat is hiding behind the couch. I can see its tail sticking out from behind the couch.\
                    Zhejiang University is a university in Hangzhou, China. It is located behind the West Lake, a famous tourist attraction in the city."
                )
            elif text == "Left Front":
                direction_text_batch[i] = (
                    "Left Front yes it is. The Left Front is a political party in the United States. It was founded in 2004 by a group of former members of the Democratic Party."
                )
            elif text == "Right Front":
                direction_text_batch[i] = (
                    "Right Front well done. The Right Front china is a desktop publishing software application developed by Adobe Systems. It is used to create documents, such as newsletters, brochures, and flyers."
                )
            elif text == "Left Behind":
                direction_text_batch[i] = (
                    "Left Behind is a novel by Tim LaHaye and Jerry B. Jenkins that was first published in 1995. It is the first book in the Left Behind series, which has sold over 65 million copies worldwide."
                )
            elif text == "Right Behind":
                direction_text_batch[i] = (
                    "Right Behind is a song by the American rock band Nine Inch Nails. It was released as the third single from their second studio album, The Downward Spiral (1994)."
                )
        return direction_text_batch

    def forward(self, **kwargs):
        self.batch = kwargs["batch"]
        batch = kwargs["batch"]
        self.phrase = batch["phrase"]
        self.file_name = batch["file_path"]
        texts = batch["phrase"]
        reference_obj_name = batch["reference_obj"]
        direction_text = self.update_text(batch["direction_text"])
        anchor_pos = batch["anchor_position"]  # 新增 anchor position
        output = {}

        # encode text(object name)
        l_enc, l_emb, l_mask = self.encode_text(
            reference_obj_name
        )  # encode the direction text
        l_input = l_emb if "word" in self.lang_fusion_type else l_enc
        l_input = l_input.to(dtype=torch.float32)
        batch["target_query"] = l_input

        d_enc, d_emb, d_mask = self.direction_encoder(
            direction_text
        )  # encode the direction text
        d_input = d_enc.to(dtype=torch.float32)
        d_input = self.direction_mlp(d_input)

        # point cloud
        scene_pcs = batch["fps_points_scene"]
        obj_pcs = batch["fps_points_scene"]

        # pointnet_1: extract the geometric affordance feat.
        obj_pcs = obj_pcs - anchor_pos.unsqueeze(1)  # [B, Num_points, 3]
        x_geo = self.geoafford_module(
            scene_pcs, obj_pcs
        )  # [B, C, Num_pts] [B, 48, Num_pts]

        # CLIP: extract the rgb feat.
        x_rgb = self.clipunet_module(batch=batch)  # [B, C_rgbfeat, H, W] [B, 64, H, W]

        # merge 得到场景特征
        x_align = self.concate(x_rgb["affordance"], scene_pcs, self.intrinsics).permute(
            0, 2, 1
        )  # [B, 64, 2048]
        x_scene = torch.cat(
            (scene_pcs.permute(0, 2, 1), obj_pcs.permute(0, 2, 1), x_align, x_geo),
            dim=1,
        )  # 场景特征 [B, 118, 2048]
        # each position minues anchor position
        x_scene = x_scene.permute(0, 2, 1)  # [B, 2048, 118]

        # 处理anchor position 特征
        # input: B*3
        x_anchor_position = self.anchor_mlp(anchor_pos)  # output: [B, 256]

        # 处理 direction text
        x_anchor_and_text = torch.cat(
            (x_anchor_position, d_input), dim=1
        )  # [B,  (256 + 256) ]
        x_anchor_and_text = x_anchor_and_text.unsqueeze(1)  # [B, 1, 256+256]
        x_anchor_and_text = self.cond_mlp(x_anchor_and_text)  # [B, 256]

        # perceive 处理x_scene and x_anchor_and_text
        x = self.feat_perceiver(x_scene, x_anchor_and_text)  # [B, 2048, 256]

        # 2nd perceiver 处理 x and d_input
        x_feat = self.feat_perceiver_2(x, d_input.unsqueeze(1)) # [B, 2048, 256]

        x = self.fusion_point_moudule(x_feat) # skipped in encoder
        # x = x.permute(0,2,1)

        output["affordance"] = x
        output["affordance_feat"] = x_feat
        self.model_ouput = x
        self.model_pc = scene_pcs

        return output

if __name__ == "__main__":
    num_points = 2048
    model = GeoL_net_v8(input_shape=(3, 480, 640), target_input_shape=(3, 128, 128))
    model = model.to("cuda")
    batch = {}
    batch_size = 2
    batch["phrase"] = ["red ball", "blue ball"]
    batch["reference_obj"] = ["ball", "ball"]
    batch["direction_text"] = ["Left", "Behind"]
    batch["anchor_position"] = torch.rand(batch_size, 3).cuda()
    batch["fps_points_scene"] = torch.rand(batch_size, num_points, 3).cuda()
    batch["image"] = torch.rand(batch_size, 3, 480, 640).cuda()
    batch["mask"] = torch.rand(batch_size, num_points, 1).cuda()
    batch["file_path"] = [
        "dataset/scene_RGBD_mask/id162_1/lamp_0004_orange/pc_ori.ply",
        "dataset/scene_RGBD_mask/id162_1/lamp_0004_orange/pc_ori.ply",
    ]

    output = model(batch=batch)["affordance"]
    print(output.shape)
    loss_fn = BinaryCELoss(config=None)
    loss = loss_fn(output, batch["mask"])
    print(loss)
