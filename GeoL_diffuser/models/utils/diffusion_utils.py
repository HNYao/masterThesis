import time
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from clip.model import build_model, load_clip, tokenize

# import pointnet 
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG
from pointnet2.models.pointnet2_ssg_sem import PointNet2SemSegSSG
from pointnet2_ops import pointnet2_utils
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from thirdpart.Pointnet2_PyTorch.pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG
from GeoL_net.core.registry import registry

def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32, device="cuda"):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype, device=device)

class WeightedLoss(nn.Module): 
    def __init__(self):
        super(WeightedLoss, self).__init__()
        # TODO: add weighted loss
    
    def forward(self, pred, target, weighted=1.0):
        """
        pred, target: [batch_size, dim]
        """
        loss = self._loss(pred, target)
        WeightedLoss = (loss * weighted).mean()
        return WeightedLoss

class WeightedL1(WeightedLoss):
    def _loss(self, pred, target):
        return torch.abs(pred - target)

class WeightedL2(WeightedLoss):
    def _loss(self, pred, target):
        return F.mse_loss(pred, target, reduction="none")

Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2,
}

def extract(a, t, x_shape):
    """
    get the data of timestep t from buffer and reshape

    """
    b, *_ = t.shape
    out = a.gather(-1,t)
    return out.reshape(b, *((1,)*(len(x_shape)-1)))

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        device = x.device 
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        
        return emb

class EMA():
    def __init__(self, decay):
        self.decay = decay
    
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.decay + (1 - self.decay) * new

    def update_model_average(self, ema_model, current_model):
        for current_params, ema_params in zip(current_model.parameters(), ema_model.parameters()):
            old, new = ema_params.data, current_params.data
            ema_params.data = self.update_average(old, new)


class AffordanceEncoder(nn.Module):
    def __init__(self, state_dim=1, hidden_dim=256, device="cuda"):
        super(AffordanceEncoder, self).__init__()
        self.device = device
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )

    def forward(self, state):
        return self.encoder(state)

class PCPositionEncoder(nn.Module):
    def __init__(self, state_dim=3, hidden_dim=256, device="cuda"):
        super(PCPositionEncoder, self).__init__()
        self.device = device
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )

    def forward(self, state):
        #import pdb; pdb.set_trace()
        state = state.float()
        return self.encoder(state).to(self.device)

class PCPositionAndAffordanceEncoder(nn.Module):
    '''
    encode the point cloud position and affordance
    '''
    def  __init__(self, state_dim=4, hidden_dim=256, device="cuda"):
        super(PCPositionAndAffordanceEncoder, self).__init__()
        self.device = device
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.encoder = PointNet2SemSegMSG_with_affordance(True).cuda()
    
    def forward(self, state):
        return self.encoder(state)

class ObjectNameEncoder(nn.Module):
    """
    encode the object text name by clip
    """
    def __init__(self, out_dim, device='cuda') -> None:
        super().__init__()
        self.device = device
        self.out_dim = out_dim
        self._load_clip()
        self.fc1 = nn.Linear(1024, 2048).to(self.device)
        self.fc2 = nn.Linear(2048, out_dim).to(self.device)

    def _load_clip(self):
        model, _ = load_clip("RN50", device=self.device)
        self.clip_rn50 = build_model(model.state_dict()).to(self.device) #10kw frozen
        del model
        #Frozen clip
        for param in self.clip_rn50.parameters():
            param.requires_grad = False
    
    def encode_text(self, x):
        with torch.no_grad():
            
            tokens = tokenize(x).to('cuda')
            
            self.clip_rn50.cuda()
            text_feat, text_emb = self.clip_rn50.encode_text_with_embeddings(tokens)

        return text_feat, text_emb

    def forward(self, x):
        
        text_feat, text_emb= self.encode_text(x)
        x = torch.tensor(text_feat, dtype=torch.float32).to(self.device)
        self.fc1.cuda()
        self.fc2.cuda()
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).unsqueeze(1).repeat(1, 2048, 1)

        return x

class ObjectNameEncoder_v2(nn.Module):
    """
    encode the object text name by clip
    """
    def __init__(self, out_dim, device='cuda') -> None:
        super().__init__()
        self.device = device
        self.out_dim = out_dim
        self._load_clip()
        self.mlp = nn.Sequential(
            nn.Linear(1024, 2048).to(self.device),
            nn.Mish(),
            nn.Linear(2048, 512).to(self.device),
            nn.Mish(),
            nn.Linear(512, out_dim).to(self.device),
        )
    def _load_clip(self):
        model, _ = load_clip("RN50", device=self.device)
        self.clip_rn50 = build_model(model.state_dict()).to(self.device) #10kw frozen
        del model
        #Frozen clip
        for param in self.clip_rn50.parameters():
            param.requires_grad = False
    
    def encode_text(self, x):
        with torch.no_grad():
            
            tokens = tokenize(x).to('cuda')
            
            self.clip_rn50.cuda()
            text_feat, text_emb = self.clip_rn50.encode_text_with_embeddings(tokens)

        return text_feat, text_emb

    def forward(self, x):
        
        text_feat, text_emb= self.encode_text(x)
        x = torch.tensor(text_feat, dtype=torch.float32).to(self.device)
        x = self.mlp(x)

        return x


class ObjectPCEncoder(nn.Module):
    """
    encode the object point cloud
    """
    def __init__(self, device='cuda') -> None:
        super().__init__()
        self.device = device
        self.encoder = PointNet2SemSegMSG(True).cuda()
    def forward(self, x):
        x = self.encoder(x)
        return x

class ObjectPCEncoder_v2(nn.Module):
    """
    encode the object point cloud
    """
    def __init__(self, device='cuda') -> None:
        super().__init__()
        self.device = device
        self.encoder = nn.Sequential(
            PointNet2SemSegMSG(True).cuda(),
            nn.Mish(),
        )
        self.linear_1 = nn.Linear(64, 1)
        self.mish = nn.Mish()
        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.Mish(),
            nn.Linear(256, 64),
        )
    def forward(self, x):
        x = self.encoder(x).permute(0, 2, 1)
        x = self.mish(x)
        x= self.linear_1(x).squeeze(-1)
        x = self.mlp(x)
        return x

class PCxyPositionEncoder(nn.Module):
    def __init__(self, state_dim=2, hidden_dim=256, device="cuda"):
        super(PCxyPositionEncoder, self).__init__()
        self.device = device
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, state):
        return self.encoder(state)

class ReduceNet(nn.Module):
    def __init__(self, state_dim=4, hidden_dim=2048, device="cuda"):
        super(ReduceNet, self).__init__()
        self.conv1 = nn.Conv1d(state_dim, hidden_dim, 1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.mish = nn.Mish()

    def forward(self, x):
        x = x.permute(0, 2, 1).cuda()
        self.conv1.cuda()
        self.conv2.cuda()
        self.mish.cuda()
        self.pool.cuda()
        self.fc.cuda()

        x = self.mish(self.conv1(x))
        x = self.mish(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x= self.fc(x)
        return x
    


class PointNet2SemSegMSG_with_affordance(PointNet2SemSegSSG):
    """extract the features from point cloud and affordance"""
    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        c_in = 1
        c_in_ori = c_in
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=256,
                radii=[150, 250], # change 
                nsamples=[16, 32],
                mlps=[[c_in, 16, 16, 32], [c_in, 32, 32, 64]],
                use_xyz=True,
            )
        )
        c_out_0 = 32 + 64

        c_in = c_out_0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=64,
                radii=[250, 450],
                nsamples=[8, 16],
                mlps=[[c_in, 64, 64, 128], [c_in, 64, 96, 128]],
                use_xyz=True,
            )
        )
        c_out_1 = 128 + 128

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[256 + c_in_ori, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + c_out_0, 256]))
        #self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_1, 512]))


        self.fc_lyaer = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.Mish(),
            nn.Dropout(0.5),
            nn.Conv1d(128, 64, kernel_size=1),
        )


class PointNet2SemSegMSG(PointNet2SemSegSSG):
    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        c_in = 0
        c_in_ori = c_in
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[30, 60], # change 
                nsamples=[16, 32],
                mlps=[[c_in, 32], [c_in, 64]],
                use_xyz=True,
            )
        )
        c_out_0 = 32 + 64

        c_in = c_out_0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=32,
                radii=[40, 60],
                nsamples=[8, 16],
                mlps=[[c_in, 64, 128], [c_in, 64, 128]],
                use_xyz=True,
            )
        )
        c_out_1 = 128 + 128

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[256 + c_in_ori, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + c_out_0, 256]))
        #self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_1, 512]))


        self.fc_lyaer = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.Mish(),
            nn.Dropout(0.5),
            nn.Conv1d(128, 64, kernel_size=1),
        )


class MapFeatureExtractor(nn.Module):
    def  __init__(
            self
    ):
        super(MapFeatureExtractor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.Mish(),
            nn.Linear(32, 64),
        )
    def weighted_features(self, query, scene, feats, distances, alpha=1.0, k=None):

        if k is not None:

            topk_indices = torch.topk(-distances, k, dim=-1).indices  # [b, n_query, k]

            selected_feats = torch.gather(feats[:, None, :, :].expand(-1, query.shape[1], -1, -1), 2, topk_indices[..., None].expand(-1, -1, -1, feats.shape[-1]))
            selected_distances = torch.gather(distances, 2, topk_indices)

            weights = torch.exp(-alpha * selected_distances)  # [b, n_query, k]
        else:
            weights = torch.exp(-alpha * distances)  # [b, n_query, n]
            selected_feats = feats[:, None, :, :].expand(-1, query.shape[1], -1, feats.shape[-1])  # [b, n_query, n, 1]

        weighted_feats = torch.sum(weights[:, :, :, None] * selected_feats, dim=2)  # [b, n_query, 1]
        normalization = torch.sum(weights, dim=2, keepdim=True)  # [b, n_query, 1]

        return weighted_feats / (normalization + 1e-6)



        
    def forward(self, query_points, scene_pc, scene_pc_feats, alpha=1.0):
        """
        query_points: [batch_size, num_points, 3]
        scene_pc: [batch_size, num_points, 3]
        scene_pc_feats: [batch_size, num_points, 1]

        query_feats: [batch_size, num_points, 2]
        """

        ##### weighted, distance
        distances = torch.norm(query_points[:, :, None, :2] - scene_pc[:, None, :, :2], dim=-1)

        weights =  torch.exp(-distances * alpha)
        weighted_feats = torch.sum(weights[:, :, :, None] * scene_pc_feats[:, None, :, :], dim=2)
        normalization = torch.sum(weights, dim=2, keepdim=True)
        query_feats = weighted_feats / normalization

        ####  nearest
        #nearest_indices = torch.argmin(distances, dim=-1)
        #b, n_query = nearest_indices.shape
        #batch_indices = torch.arange(b, device=nearest_indices.device).unsqueeze(-1).repeat(1, n_query)
        #query_feats = scene_pc_feats[batch_indices, nearest_indices]

        #### wieghted, x, y split
        #k1 = 1024
        #k2 = 64

        #dist_x = torch.abs(query_points[:, :, None, 0] - scene_pc[:, None, :, 0])  # [b, n_query, n]
        #dist_y = torch.abs(query_points[:, :, None, 1] - scene_pc[:, None, :, 1])  # [b, n_query, n]

        #all_feats_x = self.weighted_features(query_points, scene_pc, scene_pc_feats, dist_x, k=None)  # [b, n_query, 1]
        #all_feats_y = self.weighted_features(query_points, scene_pc, scene_pc_feats, dist_y, k=None)  # [b, n_query, 1]

        #topk1_feats_x = self.weighted_features(query_points, scene_pc, scene_pc_feats, dist_x, k=k1)  # [b, n_query, 1]
        #topk1_feats_y = self.weighted_features(query_points, scene_pc, scene_pc_feats, dist_y, k=k1)  # [b, n_query, 1]

        #topk2_feats_x = self.weighted_features(query_points, scene_pc, scene_pc_feats, dist_x, k=k2)  # [b, n_query, 1]
        #topk2_feats_y = self.weighted_features(query_points, scene_pc, scene_pc_feats, dist_y, k=k2)  # [b, n_query, 1]

        # 拼接特征
        #query_feats = torch.cat([
        #    all_feats_x, all_feats_y,
        #    topk1_feats_x, topk1_feats_y,
        #    topk2_feats_x, topk2_feats_y
        #], dim=-1)  # [b, n_query, 6]

        return query_feats

class AffordanceFeatureExtractor(nn.Module):
    def __init__(
            self
    ):
        self.weights = 'outputs/checkpoints/GeoL_v9_10K_meter_retrain/ckpt_29.pth'
        self.state_affordance_dict = torch.load(self.weights, map_location='cpu')
        self.intrinsic = np.array([[303.54, 0.0, 318.4], [0.0, 303.526, 183.679], [0.0, 0.0, 1.0]])
        self.affordance_encoder = registry.get_affordance_encoder("AffordanceEncoder")(
            input_shape=[3, 360,640], 
            target_input_shape=[3, 128, 128],
            intrinsic=self.intrinsic,
            ).to('cuda')
        self.affordance_encoder.load_state_dict(self.state_affordance_dict['state_dict'])
    
    def forward(self, query_points, scene_pc, scene_pc_feats, alpha=5.0):
        """
        query_points: [batch_size, num_points, 3]
        scene_pc: [batch_size, num_points, 3]
        scene_pc_feats: [batch_size, num_points, 1]

        query_feats: [batch_size, num_points, 1]
        """
        rgb_image_file_path = "dataset/scene_RGBD_mask_v2_kinect_cfg/id744/bottle_0002_mixture/with_obj/test_pbr/000000/rgb/000000.jpg"
        depth_image_file_path = "dataset/scene_RGBD_mask_v2_kinect_cfg/id744/bottle_0002_mixture/with_obj/test_pbr/000000/depth_noise/000000.png"

        self.affordance_encoder.eval()        



        

