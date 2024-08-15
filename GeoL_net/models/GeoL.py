import torch
import torch.nn as nn
from GeoL_net.models.clip_unet import CLIPUNet
from GeoL_net.models.geo_net import GeoAffordModule
from GeoL_net.models.modules import FeatureConcat, Project3D
from GeoL_net.core.registry import registry
from clip.model import build_model, load_clip, tokenize

@registry.register_affordance_model(name="GeoL_net")
class GeoL_net(nn.Module):
    def __init__(self, input_shape, target_input_shape):
        super().__init__()
        self.geoafford_module = GeoAffordModule(feat_dim=16)
        self.clipunet_module = CLIPUNet(input_shape=input_shape, target_input_shape=target_input_shape)
        self.concate = FeatureConcat()

        self.device = "cuda" # cpu for dataset
        self.lang_fusion_type = 'mult' # hard code from CLIPlingunet
        self._load_clip()

    def _load_clip(self):
        model, _ = load_clip("RN50", device=self.device)
        self.clip_rn50 = build_model(model.state_dict()).to(self.device)
        del model
    
    def encode_text(self, x):
        with torch.no_grad():
            tokens = tokenize(x).to(self.device)
            text_feat, text_emb = self.clip_rn50.encode_text_with_embeddings(tokens)

        text_mask = torch.where(tokens==0, tokens, 1)  # [1, max_token_len]
        return text_feat, text_emb, text_mask

    
    def forward(self, **kwargs):
        batch = kwargs["batch"]
        texts = batch["phrase"]
        print(texts)
        l_enc, l_emb, l_mask = self.encode_text(texts)
        l_input = l_emb if 'word' in self.lang_fusion_type else l_enc
        l_input = l_input.to(dtype=torch.float32)
        batch["target_query"] = l_input
        scene_pcs = batch["fps_points_scene"]
        obj_pcs = batch["fps_points_scene"]

        x_geo = self.geoafford_module(scene_pcs, obj_pcs)
        print("----x_geo shape:", x_geo.shape)
        x_rgb = self.clipunet_module(batch=batch)
        print("----x_rgb shape:", x_rgb["affordance"].shape)
        #x = self.concate(scene_pcs, obj_pcs)

        return x_rgb
