import torch
import torch.nn as nn
import torchvision
import pytorch3d
#print(torch.__version__) # 2.0.0
#print(torch.version.cuda) # 11.7 cudatoolkit 11.8.0 pytorch-cuda 11.7
#print(torchvision.__version__) # 0.15.0
#print(pytorch3d.__version__) # 0.7.5
import open3d as o3d
#print(o3d.__version__) # 0.18.0
import matplotlib.pyplot as plt
from GeoL_net.models.clip_unet import CLIPUNet
from GeoL_net.models.geo_net import GeoAffordModule, FusionPointLayer
from GeoL_net.models.modules import FeatureConcat, Project3D
from GeoL_net.core.registry import registry
import numpy as np
from PIL import Image
from clip.model import build_model, load_clip, tokenize

@registry.register_affordance_model(name="GeoL_net")
class GeoL_net(nn.Module):
    def __init__(self, input_shape, target_input_shape):
        super().__init__()
        self.geoafford_module = GeoAffordModule(feat_dim=16) #0.3kw

        self.clipunet_module = CLIPUNet(input_shape=input_shape, target_input_shape=target_input_shape, output_dim=16) #6kw parameters

        self.concate = FeatureConcat()
        self.device = "cuda" # cpu for dataset
        self.lang_fusion_type = 'mult' # hard code from CLIPlingunet
        self._load_clip()
        self.instrics = np.array([[591.0125 ,   0.     , 322.525  ],
            [  0.     , 590.16775, 244.11084],
            [  0.     ,   0.     ,   1.     ]])
        self.fusion_point_moudule = FusionPointLayer() #0.3kw


    def _load_clip(self):
        model, _ = load_clip("RN50", device=self.device)
        self.clip_rn50 = build_model(model.state_dict()).to(self.device) #10kw frozen
        del model
        #Frozen clip
        for param in self.clip_rn50.parameters():
            param.requires_grad = False
    
    def encode_text(self, x):
        with torch.no_grad():
            tokens = tokenize(x).to(self.device)
            text_feat, text_emb = self.clip_rn50.encode_text_with_embeddings(tokens)

        text_mask = torch.where(tokens==0, tokens, 1)  # [1, max_token_len]
        return text_feat, text_emb, text_mask

    
    def forward(self, **kwargs):
        batch = kwargs["batch"]
        self.phrase = batch["phrase"]
        self.file_name = batch["file_path"]
        texts = batch["phrase"]
        output = {}
        #print(texts)

        # encode text
        l_enc, l_emb, l_mask = self.encode_text(texts)
        l_input = l_emb if 'word' in self.lang_fusion_type else l_enc
        l_input = l_input.to(dtype=torch.float32)
        batch["target_query"] = l_input

        # point cloud
        scene_pcs = batch["fps_points_scene"]
        obj_pcs = batch["fps_points_scene"]
        

        # pointnet_1: extract the geometric affordance feat.
        x_geo = self.geoafford_module(scene_pcs, obj_pcs) # [B, C, Num_pts]
        #print("----x_geo shape:", x_geo.shape)

        # CLIP: extract the rgb feat.
        x_rgb = self.clipunet_module(batch=batch) # [B, C_rgbfeat, H, W]
        #print("----x_rgb shape:", x_rgb["affordance"].shape)
        #print("-----scene_pcs shape:", scene_pcs.shape) # [B, Num_puts, 3]

        # merge
        x_align = self.concate(x_rgb["affordance"], scene_pcs, self.instrics).permute(0,2,1) # [B, Num_pts, C_rgbfeat]
        #print("-----align shape:", x_align.shape)
        x = torch.cat((scene_pcs.permute(0,2,1), x_align, x_geo),dim=1) # []
        #print("------x shape:", x.shape)
        x = self.fusion_point_moudule(x.permute(0,2,1))
        #x = x.permute(0,2,1)
        #print("------fusion x shape:", x.shape)
        #print("------target shape:", batch["mask"].shape)
        output["affordance"] = x
        self.model_ouput = x
        self.model_pc = scene_pcs
        #print(x)

        return output
    
    def inference_4cls(self, epoch):
        " Check the model outputs during training"
        pcs = self.model_pc # [B, Num_points, 3]
        feat = self.model_ouput.permute(0,2,1) # [B, Num_points, 4]

        color_map = torch.tensor([
            [1,0,0],
            [0,1,0],
            [0,0,1],
            [0,0,0]
        ]).cpu()
        predicted_color = torch.argmax(feat, dim=-1).cpu()
        colors = color_map[predicted_color]

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pcs[0].cpu().numpy())
        point_cloud.colors = o3d.utility.Vector3dVector(colors[0].cpu().numpy())
        scene_id = self.file_name[0].split("/")[-2]
        obj_id = self.file_name[0].split("/")[-1]
        o3d.io.write_point_cloud(f"outputs/model_output/point_cloud/{scene_id}-{obj_id}-{epoch}.ply", point_cloud)

        # visualize and render the point cloud
        #pcd = point_cloud
        #R = pcd.get_rotation_matrix_from_axis_angle([np.pi, 0, 0])

        #pcd.rotate(R, center=(0, 0, 0))
        #vis = o3d.visualization.Visualizer()
        #vis.create_window(visible=False)
        #vis.add_geometry(pcd)
        #vis.poll_events()
        #vis.update_renderer()
        #temp_image_path = "outputs/point_cloud_4cls_temp.png"
        #img = vis.capture_screen_image(temp_image_path)
        #img = np.asarray(img) * 255
        #img = img.astype(np.uint8)
        #image_pil = Image.fromarray(img.clip(0,255)).astype(np.uint8)
        
        #vis.destroy_window()

        #return image_pil, self.phrase[0], self.file_name[0]



    def inference_2cls(self):
        " Check the model outputs during training"
        pcs = self.model_pc # [B, Num_points, 3]
        feat = self.model_ouput.permute(0,2,1) # [B, Num_points, 2]

        color_map = torch.tensor([
            [0,0,0],
            [0,1,0]
        ]).cpu()
        predicted_color = torch.argmax(feat, dim=-1).cpu()
        colors = color_map[predicted_color]

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pcs[0].cpu().numpy())
        point_cloud.colors = o3d.utility.Vector3dVector(colors[0].cpu().numpy())
        #o3d.visualization.draw([point_cloud])
        o3d.io.write_point_cloud("outputs/point_cloud.ply_2cls", point_cloud)
    
    def inference_heatmap_4cls(self):
        "conver the esimating value to self-dpsefined heatmap"
        pcs = self.model_pc # [B, Num_points, 3]
        feat = self.model_ouput.permute(0,2,1) # [B, Num_points, 4]
        class_1_feat = feat[:, :, 1].cpu()  # [B, Num_points]

        max_value = torch.max(class_1_feat)
        min_value = torch.min(class_1_feat)

        # normalization
        normalized_class_1_feat = (class_1_feat - min_value) / (max_value - min_value)
        #normalized_class_1_feat = class_1_feat
        flattened = normalized_class_1_feat.detach().numpy().flatten()
        cmap = plt.get_cmap('turbo')
        color_mapped = cmap(flattened)[:, :3]  # 获取 RGB 值，忽略 alpha 通道

        # back to [B, Num_points, 3]
        colors = color_mapped.reshape(normalized_class_1_feat.shape[0], normalized_class_1_feat.shape[1], 3)

        # color type  float64，Open3D
        colors = colors.astype(np.float64)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pcs[0].cpu().numpy())
        point_cloud.colors = o3d.utility.Vector3dVector(colors[0])
        self.pc_heatmap = point_cloud
        o3d.io.write_point_cloud("outputs/point_cloud_heatmap_4cls.ply", point_cloud)
        self.color_backproj()

    def color_backproj(self, pc_ori=None):
        " bactproject the color from the fps-points to the whole point cloud"
        cam_rotation_matrix = np.array([
        [1, 0, 0],
        [0,0.8,-0.6],
        [0,0.6,0.8]
        ])  

        ply1 = o3d.io.read_point_cloud("dataset/scene_RGBD_mask/id162_1/lamp_0004_orange/pc_ori.ply")
        ply2 = self.pc_heatmap
        points_ply1 = np.asarray(ply1.points)
        points_ply1 = (np.linalg.inv(cam_rotation_matrix) @ points_ply1.T).T
        points_ply2 = np.asarray(ply2.points)
        colors_ply2 = np.asarray(ply2.colors)

        # KD tree ply2 ply1
        ply2_tree = o3d.geometry.KDTreeFlann(ply2)
        new_colors = np.zeros_like(points_ply1)

        for i, point in enumerate(points_ply1):
            [_, idx, dist] = ply2_tree.search_knn_vector_3d(point, 10) # nearest 10
            nearest_colors = colors_ply2[idx]
            distances = np.array(dist)
            weights = 1 / (distances + 1e-6)
            weighted_avg_color = np.average(nearest_colors, axis=0, weights=weights) # weighted avg
            new_colors[i] = weighted_avg_color

        ply1.colors = o3d.utility.Vector3dVector(new_colors)
        ply1.points = o3d.utility.Vector3dVector(points_ply1)


        # 保存结果
        o3d.io.write_point_cloud("outputs/point_cloud_heatmap_4cls_whole.ply", ply1)