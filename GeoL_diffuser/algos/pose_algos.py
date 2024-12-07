import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
import cv2
import clip
from GeoL_diffuser.models.diffusion import Diffusion
import GeoL_diffuser.models.tensor_utils as TensorUtils
import torch.optim as optim
from GeoL_diffuser.models.helpers import EMA
from GeoL_net.core.registry import registry
from GeoL_diffuser.dataset.dataset import *


@registry.register_diffusion_model(name="GeoL_diffuser")
class PoseDiffusionModel(nn.Module):
    def __init__(self, algo_config):
        super().__init__()
        self.algo_config = algo_config
        # self.train_config = train_config
        self.nets = nn.ModuleDict()

        # conditioning parsing: set in the trainer

        # Initialize diffuser
        policy_kwargs = self.algo_config.model_config
        self.nets["policy"] = Diffusion(**policy_kwargs)

        # set up EMA
        self.use_ema = self.algo_config.ema.use_ema
        if self.use_ema:
            print("DIFFUSER: using EMA... val and get_action will use ema model")
            self.ema = EMA(algo_config.ema.ema_decay)
            self.ema_policy = copy.deepcopy(self.nets["policy"])
            self.ema_policy.requires_grad_(False)
            self.ema_update_every = algo_config.ema.ema_step
            self.ema_start_step = algo_config.ema.ema_start_step
            self.reset_parameters()

        self.curr_train_step = 0

    @property
    def checkpoint_monitor_keys(self):
        if self.use_ema:
            return {"valLoss": "val/ema_losses_diffusion_loss"}
        else:
            return {"valLoss": "val/losses_diffusion_loss"}

    def forward(
        self,
        data_batch,
        num_samp=10,
        return_guidance_losses=True,
        class_free_guide_w=-1,
        apply_guidance=False,
        guide_clean=True,
    ):
        curr_policy = self.nets["policy"]
        if self.use_ema:
            curr_policy = self.ema_policy
        output = curr_policy(
            data_batch,
            num_samp,
            return_guidance_losses,
            class_free_guide_w=class_free_guide_w,
            apply_guidance=apply_guidance,
            guide_clean=guide_clean,
        )
        pose_xyz_pred = output["pose_xyz_pred"]
        B, N, H, _ = pose_xyz_pred.shape
        output["pose_xyz_pred"] = pose_xyz_pred.view(B, N * H, -1)

        # TODO: check the guidance losses
        # not using guidance in training
        #if "guide_losses" in output:
        #    for k, v in output["guide_losses"].items():
        #        v = TensorUtils.detach(v)
        #        output["guide_losses"][k] = v.view(B, N * H)

        return output

    def reset_parameters(self):
        self.ema_policy.load_state_dict(self.nets["policy"].state_dict())

    def step_ema(self, step):
        if step < self.ema_start_step:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_policy, self.nets["policy"])

    def configure_optimizers(self):
        optim_params = self.algo_config.optimization
        return optim.Adam(
            params=self.nets["policy"].parameters(),
            lr=optim_params.learning_rate,
        )

    # def training_step_end(self, *args, **kwargs):
    #    self.curr_train_step += 1

    def get_loss(self, data_batch, batch_idx):
        losses = self.nets["policy"].compute_losses(data_batch)

        # summarize losses
        total_loss = 0
        for lk, l in losses.items():
            losses[lk] = l * self.algo_config.training.loss_weights[lk]
            total_loss += losses[lk]

        return {
            "loss": total_loss,
            "all_losses": losses,
        }

    # def validation_step(self, data_batch, *args, **kwargs):
    #    curr_policy = self.nets['policy']
    #    #curr_policy.compute_losses(data_batch)
    #    losses = TensorUtils.detach(curr_policy.compute_losses(data_batch))
    #
    #    return_dict = {"losses:": losses}
    #
    #    return return_dict
