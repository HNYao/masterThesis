import numpy as np
import copy
import torch
import torch.nn as nn   
import torch.nn.functional as F
import pytorch_lightning as pl
import open3d as o3d
import cv2
import clip
from models.diffusion import Diffusion

VLM, VLM_TRANSFORM = clip.load("ViT-B/16", jit=False)
VLM.eval()
for p in VLM.parameters():
    p.requires_grad = False

class PoseDiffusionModel(pl.LightningModule):
    def __init__(self, algo_config, train_config):
        super().__init__()
        self.algo_config = algo_config
        self.train_config = train_config
        self.nets = nn.ModuleDict()

        # TODO: conditioning parsing

        # Initialize diffuser
        policy_kwargs = self.algo_config.model
        self.nets['policy'] = Diffusion(**policy_kwargs)

        # set up EMA
        self.use_ema = self.algo_config.ema.use_ema

        self.curr_train_step = 0
    
    def forward(
            self,
            data_batch,
            num_samp=1,

    ):
        curr_policy = self.nets['policy']
        output = curr_policy(data_batch, num_samp)
        return output

    def configure_optimizers(self):
        optim_params = self.algo_config.optimization
        return optim.Adam(
            params=self.nets['policy'].parameters(),
            lr=optim_params.learning_rate,
        )

    def training_step_end(self, *args, **kwargs):
        self.curr_train_step += 1
    
    def training_step(self, data_batch, batch_idx):
        loss = self.nets['policy'].loss(data_batch)
        return loss
    
    def validation_step(self, data_batch, *args, **kwargs):
        curr_policy = self.nets['policy']
        curr_policy.compute_losses(data_batch)
        out = curr_policy(
            data_batch,
            num_samp=1,  # self.algo_config.training.num_eval_samples,
            return_diffusion=False,
            return_guidance_losses=False,  # FIXME: this is a hack to avoid guidance
            apply_guidance=False,  # FIXME: this is a hack to avoid guidance
        )
        
