import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from collections import OrderedDict
from GeoL_diffuser.models.utils.guidance_loss import DiffuserGuidance
from GeoL_diffuser.dataset.dataset import *
from torch.utils.data import DataLoader
from GeoL_net.models.GeoL import FeaturePerceiver
from GeoL_diffuser.models.temporal import TemporalMapUnet, TemporalMapUnet_v2

import GeoL_diffuser.models.tensor_utils as TensorUtils
from GeoL_diffuser.models.utils.guidance_loss import DiffuserGuidance
from GeoL_diffuser.models.utils.diffusion_utils import *


class Diffusion(nn.Module):
    def __init__(
        self,
        loss_type,
        beta_schedule="cosine",
        clip_denoised=True,
        predict_epsilon=False,
        supervise_epsilons=False,
        horizon=80,
        use_map_features=True,
        **kwargs,
    ):
        super(Diffusion, self).__init__()
        self.state_dim = kwargs["obs_dim"]
        self.action_dim = kwargs["act_dim"]
        self.hidden_dim = kwargs["hidden_dim"]
        self.T = kwargs["T"]
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon 
        self.supervise_epsilons = supervise_epsilons
        self.device = torch.device(kwargs["device"])
        self.use_map_features = use_map_features

        self.horizon = horizon
        self.output_dim = 3
        self.base_dim = 32  # time_dim
        
        if self.use_map_features:
            self.model_state_dim = self.state_dim + 1
        else:
            self.model_state_dim = self.state_dim

        self.model = TemporalMapUnet(
            horizon=self.horizon,
            transition_dim=self.model_state_dim, # 3 + 1
            cond_dim=18, # 64 + 64
            output_dim=self.output_dim,
            dim=self.base_dim,  # time_dim
            dim_mults=(2, 4, 8),
            use_perceiver=True,
        ).to(self.device)

        # feat extractor
        #self.pc_position_encoder = PCPositionEncoder(state_dim=3, hidden_dim=256, device=self.device)
        #self.obj_position_encoder = ObjectPCEncoder_v2().to(self.device).eval() # NOTE: test v2
        #self.pc_position_xy_affordance_encoder = PCxyPositionEncoder(state_dim=2, hidden_dim=64)
        #self.object_name_encoder = ObjectNameEncoder_v2(out_dim=4).to(self.device).eval() # NOTE: test v2
        #self.top_affordance_encoder_position_encoder = TopAffordancePositionEncoder().to(self.device).eval()

        self.map_feature_extractor = MapFeatureExtractor().to(self.device)
        #self.test = nn.Linear(128, 1).to(self.device).eval()

        self.step = 0

        if beta_schedule == "linear":
            betas = torch.linspace(
                0.0001, 0.02, self.T, dtype=torch.float32, device=self.device
            )  # beta params
        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(
                timesteps=self.T, dtype=torch.float32, device=self.device
            )

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0).to(
            self.device
        )  # e.g. [1, 2, 3] -> [1, 1*2, 1*2*3]
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=self.device), alphas_cumprod[:-1]]
        )

        # resigter as buffer
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # forward process
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )

        # backward process
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # calculations for class-free guidance
        self.sqrt_alphas_over_one_minus_alphas_cumprod = torch.sqrt(
            alphas_cumprod / (1.0 - alphas_cumprod)
        )
        self.sqrt_recip_one_minus_alphas_cumprod = 1.0 / torch.sqrt(
            1.0 - alphas_cumprod
        )

        ## get loss coefficients and initialize objective
        self.loss_fn = Losses[loss_type]()

        # for guided sampling
        self.current_guidance = None

    def set_guidance(self, guidance):
        """
        Instantiates test-time guidance functions using the list of configs (dicts) passed in.
        """
        self.current_guidance = guidance # no guidance


    # ------------------------------------------ aux_info ------------------------------------------#
    def get_aux_info(self, data_batch):
        cond_fill_value = -1

        affordance = data_batch[
            "affordance"
        ]  # [batch_size, num_points, affordance_dim=1]
        pc_position = data_batch[
            "pc_position"
        ]  # [batch_size, num_points, pc_position_dim=3]

        #object_name = data_batch["object_name"]  # [batch_size, ]
        #object_pc_position = data_batch["object_pc_position"]  # [batch_size, obj_points=512, object_pc_position_dim=3]

        top_avg_position = self.top_avg_position(affordance, pc_position, topk=10)
        top_avg_position = self.scale_xyz_pose(top_avg_position, data_batch["gt_pose_xyz_min_bound"], data_batch["gt_pose_xyz_max_bound"])
        top_positions = self.top_position(affordance, pc_position, topk=5)
        #top_positions = self.scale_xyz_pose(top_positions, data_batch["gt_pose_xyz_min_bound"], data_batch["gt_pose_xyz_max_bound"])
        #top_positions = top_positions.view(-1, 5*3)
        #top_avg_position = self.top_affordance_encoder_position_encoder(top_avg_position)

        affordance_non_cond = torch.randn_like(affordance)
        top_avg_position_non_cond = self.top_avg_position(affordance_non_cond, pc_position, topk=10)
        top_avg_position_non_cond = self.scale_xyz_pose(top_avg_position_non_cond, data_batch["gt_pose_xyz_min_bound"], data_batch["gt_pose_xyz_max_bound"]) 
        top_positions_non_cond = self.top_position(affordance_non_cond, pc_position, topk=5)
        #top_positions_non_cond = self.scale_xyz_pose(top_positions_non_cond, data_batch["gt_pose_xyz_min_bound"], data_batch["gt_pose_xyz_max_bound"])
        #top_positions_non_cond = top_positions_non_cond.view(-1, 5*3)

        cond_feat = torch.cat(
            [   

                top_avg_position, # [barch_size, 3]
                top_positions # [barch_size, 15]
            ],
            dim=1,
        )
        non_cond_feat = torch.cat(
            [   

                top_avg_position_non_cond, # [barch_size, 3]
                top_positions_non_cond # [barch_size, 15]
            ],
            dim=1,
        )

        # TODO: combine the feats
        aux_info = {
            "cond_feat": cond_feat,  
            "non_cond_feat": non_cond_feat,  
            "gt_pose_4d_min_bound": data_batch["gt_pose_4d_min_bound"],  # [batch_size, 4]
            "gt_pose_4d_max_bound": data_batch["gt_pose_4d_max_bound"],
            "gt_pose_xyz_min_bound": data_batch["gt_pose_xyz_min_bound"],  # [batch_size, 3] x, y, z
            "gt_pose_xyz_max_bound": data_batch["gt_pose_xyz_max_bound"],
            "pc_position": data_batch["pc_position"],
            "affordance": data_batch["affordance"],
        }

        return aux_info
    
    def top_avg_position(self, affordance, pc_position, topk=10):
        """
        get the topk positions and average them
        
        affordance: [batch_size, num_points, 1]
        pc_position: [batch_size, num_points, 3]
        
        return: [batch_size, 3]
        """
        affordance = affordance.squeeze(-1)
        top_indices = torch.topk(affordance, topk, dim=1).indices
        batch_indices = torch.arange(affordance.size(0), device=pc_position.device).unsqueeze(1)
        top_positions = pc_position[batch_indices, top_indices]
        top_avg_position = top_positions.mean(dim=1)

        return top_avg_position
    
    def top_position(self, affordance, pc_position, topk=5):
        """
        get the topk positions and average them
        
        affordance: [batch_size, num_points, 1]
        pc_position: [batch_size, num_points, 3]
        
        return: [batch_size, topk*3]
        """
        affordance = affordance.squeeze(-1)
        top_indices = torch.topk(affordance, topk, dim=1).indices
        batch_indices = torch.arange(affordance.size(0), device=pc_position.device).unsqueeze(1)
        top_positions = pc_position[batch_indices, top_indices]
        top_positions = top_positions.view(-1, topk * 3)

        return top_positions



    # ------------------------------------------ scale and descale ------------------------------------------#

    def scale_xyz_pose(self, pose_xyz, xyz_min_bound, xyz_max_bound):
        """
        scale the pose_xyz to [-1, 1]
        pose_xyR: B * H * 3
        """
        if len(pose_xyz.shape) == 3:
            min_bound_batch = xyz_min_bound.unsqueeze(1)
            max_bound_batch = xyz_max_bound.unsqueeze(1)
        elif len(pose_xyz.shape) == 4:
            min_bound_batch = xyz_min_bound.unsqueeze(1).unsqueeze(1)
            max_bound_batch = xyz_max_bound.unsqueeze(1).unsqueeze(1)
        elif len(pose_xyz.shape) == 2:
            min_bound_batch = xyz_min_bound
            max_bound_batch = xyz_max_bound

        scale = max_bound_batch - min_bound_batch
        pose_xyz = (pose_xyz - min_bound_batch) / (scale + 1e-6)
        pose_xyz = 2 * pose_xyz - 1
        pose_xyz = pose_xyz.clamp(-1, 1)

        return pose_xyz


    def descale_xyz_pose(self, pose_xyz, xyz_min_bound, xyz_max_bound):
        """
        descale the pose_xyz to the original range
        pose_xyz: B * N * H * 3
        """
        if len(pose_xyz.shape) == 3:
            min_bound_batch = xyz_min_bound.unsqueeze(1)
            max_bound_batch = xyz_max_bound.unsqueeze(1)
        elif len(pose_xyz.shape) == 4:
            min_bound_batch = xyz_min_bound.unsqueeze(1).unsqueeze(1)
            max_bound_batch = xyz_max_bound.unsqueeze(1).unsqueeze(1)
        else:
            raise ValueError("Invalid shape of the input pose_xyz")

        scale = max_bound_batch - min_bound_batch
        pose_xyz = (pose_xyz + 1) / 2
        pose_xyz = pose_xyz * scale + min_bound_batch
        return pose_xyz

    # ------------------------------------------ TBD ------------------------------------------#

    def get_loss_weights(self, action_weight, discount):
        """
        sets loss coefficients for trajectory

        action_weight   : float
            coefficient on first action loss
        discount   : float
            multiplies t^th timestep of trajectory loss by discount**t
        """
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum("h,t->ht", discounts, dim_weights)
        ## manually set a0 weight
        loss_weights[0, -self.action_dim :] = action_weight

        return loss_weights

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def guidance(self, x, t, data_batch, aux_info, num_samp=1, return_grad_of=None):
        """
        estimate the gradient of rule reward w.r.t. the input 
        Input:
            x: [batch_size*num_samp, time_steps, feature_dim].  scaled input .
            data_batch: additional info.
            aux_info: additional info.
            return_grad_of: which variable to take gradient of guidance loss wrt, if not given,
                            takes wrt the input x.
        """
        assert (
            self.current_guidance is not None
        ), "Must instantiate guidance object before calling"
        bsize = int(x.size(0) / num_samp)
        horizon = x.size(1)
        with torch.enable_grad():
            # compute losses and gradient
            x_loss = x.reshape((bsize, num_samp, horizon, -1))
            tot_loss, per_losses = self.current_guidance.compute_guidance_loss(
                x_loss, t, data_batch
            )
            tot_loss.backward()
            guide_grad = x.grad if return_grad_of is None else return_grad_of.grad

            return guide_grad, per_losses

    def predict_start_from_noise(self, x, t, pred_noise, force_noise=False):
        """
        get the x_0 (e.g. denoised img) from x_t and noise
        x_0 = xt - sqrt(1 - alpha_t cumprod) * noise / sqrt(alpha_t cumprod)

        x: x in step t
        """
        if force_noise:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * pred_noise
            )
        else:
            return pred_noise

    def predict_noise_from_start(self, x_t, t, x_start):
        return (
            extract(
                self.sqrt_recip_one_minus_alphas_cumprod.to(x_t.device), t, x_t.shape
            )
            * x_t
            - extract(
                self.sqrt_alphas_over_one_minus_alphas_cumprod.to(x_t.device),
                t,
                x_t.shape,
            )
            * x_start
        )

    def p_mean_variance(self, x, t, aux_info={}, class_free_guide_w=0.0):
        if self.use_map_features:            
            map_feat = self.query_map_feat_grid(x.detach(), aux_info) # [b, h, feat_dim]
            x_inp = torch.cat([x, map_feat], dim=-1)
        else:
            x_inp = x
        model_prediction = self.model(
            x_inp, aux_info["cond_feat"], t
        )  # x.shape = [b, h, state_dim=4] t.shape = [b, ], aux_info["cond_feat"].shape = [b, condition_dim]
        if class_free_guide_w != 0:
            x_non_cond_inp = x.clone()
            if self.use_map_features:
                map_feat = self.query_map_feat_grid(x_non_cond_inp.detach(), aux_info)
                x_non_cond_inp = torch.cat([x_non_cond_inp, map_feat], dim=-1)
            else:
                x_non_cond_inp = x_non_cond_inp
          
            model_non_cond_prediction = self.model(
                x_non_cond_inp, aux_info["non_cond_feat"], t
            )
            if not self.predict_epsilon:
                model_pred_noise = self.predict_noise_from_start(
                    x_t=x, t=t, x_start=model_prediction
                )
                model_non_cond_pred_noise = self.predict_noise_from_start(
                    x_t=x, t=t, x_start=model_non_cond_prediction
                )
                class_free_guid_noise = (
                    (1 + class_free_guide_w) * model_pred_noise
                    - class_free_guide_w * model_non_cond_pred_noise
                )  # compose noise
                model_prediction = self.predict_start_from_noise(
                    x=x, t=t, pred_noise=class_free_guid_noise, force_noise=True
                )
            else:
                model_pred_noise = model_prediction
                model_non_cond_pred_noise = model_non_cond_prediction
                class_free_guid_noise = (
                    1 + class_free_guide_w
                ) * model_pred_noise - class_free_guide_w * model_non_cond_pred_noise
                model_prediction = class_free_guid_noise
        else:
            if not self.predict_epsilon:
                model_pred_noise = self.predict_noise_from_start(
                    x_t=x, t=t, x_start=model_prediction
                )
                model_prediction = self.predict_start_from_noise(
                    x=x, t=t, pred_noise=model_pred_noise, force_noise=True
                )

        x_recon = self.predict_start_from_noise(
            x=x, t=t, pred_noise=model_prediction, force_noise=self.predict_epsilon
        )
        x_recon.clamp_(-1, 1)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_recon, x, t
        )

        return (
            model_mean,
            posterior_variance,
            posterior_log_variance,
            (x_recon, x, t),
        )  # log makes it stable

    @torch.no_grad()
    def p_sample(
        self,
        x,
        t,
        data_batch,
        aux_info,
        num_samp=1,
        class_free_guide_w=0.0,
        apply_guidance=True,
        guide_clean=True,
        eval_final_guide_loss=False,  # NOTE: guide_clean is usually true
        *args,
        **kwargs,
    ):
        """
        denosie, single step
        """
        b, *_, device = *x.shape, x.device
        with_func = torch.no_grad

        if self.current_guidance is not None and apply_guidance and guide_clean:
            x = x.detach()
            x.requires_grad_()
            with_func = torch.enable_grad

        # get the mean and variance
        with with_func():
            model_mean, _, model_log_variance, q_posterior_in = self.p_mean_variance(
                x=x, t=t, aux_info=aux_info, class_free_guide_w=class_free_guide_w
            )

        # random noise
        noise = torch.randn_like(x)
        sigma = (0.5 * model_log_variance).exp()
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        # compute guidance
        guide_losses = None
        guide_grad = torch.zeros_like(model_mean)
        if self.current_guidance is not None and apply_guidance:
            assert (
                not self.predict_epsilon
            ), "Guidance not implemented for epsilon prediction"
            if guide_clean:  # Return gradients of x_{t-1}
                # We want to guide the predicted clean traj from model, not the noisy one
                model_clean_pred = q_posterior_in[0]
                x_guidance = model_clean_pred
                return_grad_of = x
            else:  # Return requires_grad gradients of x_0
                x_guidance = model_mean.clone().detach()
                return_grad_of = x_guidance
                x_guidance.requires_grad_()
            # TODO: Look into how gradient computation in guidance works, and implement our own guidance
            guide_grad, guide_losses = self.guidance(
                x_guidance,
                t,
                data_batch,
                aux_info,
                num_samp=num_samp,
                return_grad_of=return_grad_of,
            )

            # NOTE: empirally, scaling by the variance (sigma) seems to degrade results
            guide_grad = nonzero_mask * guide_grad  # * sigma

        noise = nonzero_mask * sigma * noise

        # 2
        if self.current_guidance is not None and guide_clean:
            assert (
                not self.predict_epsilon
            ), "Guidance not implemented for epsilon prediction"
            # perturb clean trajectory
            guided_clean = (
                q_posterior_in[0] - guide_grad
            )  # x_0' = x_0 - grad (The use of guidance)
            # use the same noisy input again
            guided_x_t = q_posterior_in[1]  # x_{t}
            # re-compute next step distribution with guided clean & noisy trajectories => q(x_{t-1}|x_{t}, x_0')
            # And remember in the training process, we want to make the output of every diffusion step to be x_0
            model_mean, _, _ = self.q_posterior(
                x_start=guided_clean, x_t=guided_x_t, t=q_posterior_in[2]
            )
            # NOTE: variance is not dependent on x_start, so it won't change. Therefore, fine to use same noise.
            x_out = model_mean + noise
        else:
            x_out = model_mean - guide_grad + noise

        # 3 evaluate guidance loss at the end. even if not applying guidance during sampling
        if (
            self.current_guidance is not None
            and eval_final_guide_loss
        ):
            assert (
                not self.predict_epsilon
            ), "Guidance not implemented for epsilon prediction"
            _, guide_losses = self.guidance(
                x_out.clone().detach().requires_grad_(),
                t,
                data_batch,
                aux_info,
                num_samp=num_samp,
            )

        return x_out, guide_losses

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        data_batch,
        aux_info,
        num_samp,
        return_guidance_losses=False,
        class_free_guide_w=0.0,
        apply_guidance=True,
        guide_clean=False,
        *args,
        **kwargs,
    ):
        """
        denosise, loop
        """
        device = self.device
        batch_size = shape[0]
        x = torch.randn(shape, device=device)  # random noise

        x = TensorUtils.join_dimensions(
            x, begin_axis=0, end_axis=2
        )  # [batch_size * num_samp, horizon, state_dim]
        aux_info = TensorUtils.repeat_by_expand_at(aux_info, repeats=num_samp, dim=0)

        if self.current_guidance is not None and not apply_guidance:
            print(
                "WARNING: not using guidance during sampling, only evaluating guidance loss at very end..."
            )

        for i in reversed(range(0, self.T)):  # reverse, denoise from the last step
            t = torch.full(
                (batch_size * num_samp,), i, device=device, dtype=torch.long
            )  # timestep
            x, guide_losses = self.p_sample(  # TODO: x, guide_losses = self.p_sample
                x,
                t,
                data_batch,
                aux_info,
                num_samp=num_samp,
                class_free_guide_w=class_free_guide_w,
                apply_guidance=apply_guidance,
                guide_clean=guide_clean,
                eval_final_guide_loss=(i == 0),
            )  # denoise

        x = TensorUtils.reshape_dimensions(
            x, begin_axis=0, end_axis=1, target_dims=(batch_size, num_samp)
        )
        out_dict = {"pred_pose_xyz": x}
        if return_guidance_losses:
            out_dict.update({"guide_losses": guide_losses})

        return out_dict

    @torch.no_grad()
    def condition_sample(
        self, data_batch, aux_info, num_samp=1, class_free_guide_w=0.0, **kwargs
    ):
        """
        sample
        """
        batch_size = data_batch["affordance"].shape[0]

        shape = (
            batch_size,
            num_samp,
            self.horizon,
            self.state_dim,
        )  # [batch_size, num_samp=1, horizon=80, state_dim=4]
        action = self.p_sample_loop(
            shape,
            data_batch,
            aux_info,
            num_samp=num_samp,
            class_free_guide_w=class_free_guide_w,
            **kwargs,
        )
        action["pred_pose_xyx"] = action["pred_pose_xyz"].clamp(-1, 1)
        return action
    
    def query_map_feat_grid(self, x, aux_info):
        """
        query the affordance feature from the map (whole point cloud scene)
        """
        query_points = self.descale_xyz_pose(x, aux_info["gt_pose_xyz_min_bound"], aux_info["gt_pose_xyz_max_bound"])
        #self.visualize_seed_points(query_points, aux_info)
        points_feature = self.map_feature_extractor(query_points, aux_info["pc_position"], aux_info["affordance"])

        return points_feature 

    def visualize_seed_points(self, query_points, aux_info):
        """
        visualize the seed points
        """
        batch_size = query_points.shape[0]
        query_points = query_points.detach().cpu().numpy()
        pc_position = aux_info['pc_position'].detach().cpu().numpy()
        affordance = aux_info['affordance'].detach().cpu().numpy()
        min_bound = aux_info['gt_pose_xyz_min_bound'].detach().cpu().numpy()
        max_bound = aux_info['gt_pose_xyz_max_bound'].detach().cpu().numpy()
        for i in range  (batch_size):
            point_cloud = pc_position[i]
            seed_points = query_points[i]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)
            pcd_colors = np.zeros_like(point_cloud)
            pcd_colors[:, 0] = affordance[i].squeeze() 
            pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

            seed_spheres = []
            bound_spheres = []
            for seed in seed_points:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
                sphere.translate(seed)  # Move the sphere to the seed point
                sphere.paint_uniform_color([0.0, 0.0, 1.0])  # Color the sphere red
                seed_spheres.append(sphere)

            for bound in [min_bound[i], max_bound[i]]:
                bound_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
                bound_sphere.translate(bound)  # Move the sphere to the seed point
                sphere.paint_uniform_color([0.0, 1.0, 0.0])
                bound_spheres.append(bound_sphere)

        # Visualize the point cloud and seed points
            
        
            o3d.visualization.draw_geometries([pcd] )
            o3d.visualization.draw_geometries([pcd] + seed_spheres)
            o3d.visualization.draw_geometries([pcd] + seed_spheres+bound_spheres)
            break




    # ------------------- Training ----------------

    def q_sample(self, x_start, t, noise=None):
        """
        x_start: x_0

        return: x_t
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, t, aux_info={}):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)

        if self.use_map_features:
            map_feat = self.query_map_feat_grid(x_noisy.detach(), aux_info)
            x_noisy_inp = torch.cat([x_noisy, map_feat], dim=-1)
        else:
            x_noisy_inp = x_noisy

        model_prediction = self.model(x_noisy_inp, aux_info["cond_feat"], t)
        x_recon = self.predict_start_from_noise(
            x=x_noisy,
            t=t,
            pred_noise=model_prediction,
            force_noise=self.predict_epsilon,
        )

        if not self.predict_epsilon:
            noise_pred = self.predict_noise_from_start(
                x_t=x_noisy, t=t, x_start=x_recon
            )
        else:
            x_recon = self.predict_start_from_noise(
                x=x_noisy, t=t, pred_noise=model_prediction, force_noise=True
            )
            noise_pred = model_prediction

        if self.supervise_epsilons:
            assert self.predict_epsilon
            loss = self.loss_fn(noise_pred, noise)
        else:
            assert not self.predict_epsilon
            loss = self.loss_fn(x_recon, x_start)

        return loss

    def loss(self, x, aux_info={}):
        batch_size = len(x)
        t = torch.randint(0, self.T, (batch_size,), device=self.device).long()
        return self.p_losses(x, t, aux_info=aux_info)

    def compute_losses(self, data_batch):
        aux_info = self.get_aux_info(data_batch)

        pose_xyz = data_batch["gt_pose_xyz"]
        gt_pose_xyz_min_bound = data_batch["gt_pose_xyz_min_bound"]
        gt_pose_xyz_max_bound = data_batch["gt_pose_xyz_max_bound"]

        x = self.scale_xyz_pose(pose_xyz, gt_pose_xyz_min_bound, gt_pose_xyz_max_bound)
        diffusion_loss = self.loss(x, aux_info=aux_info)
        losses = OrderedDict(diffusion_loss=diffusion_loss)
        return losses

    def forward(
        self,
        data_batch,
        num_samp=1,
        return_guidance_losses=False,
        class_free_guide_w=0,
        apply_guidance=False,
        guide_clean=False,
    ):
        aux_info = self.get_aux_info(data_batch)
        cond_samp_out = self.condition_sample(
            data_batch,
            aux_info=aux_info,
            num_samp=num_samp,
            class_free_guide_w=class_free_guide_w,
            apply_guidance=apply_guidance,
            guide_clean=guide_clean,
            return_guidance_losses=return_guidance_losses,
        )
        pose_xyz_scaled = cond_samp_out["pred_pose_xyz"]
        # import pdb

        # pdb.set_trace()
        gt_pose_xyz_min_bound = data_batch["gt_pose_xyz_min_bound"]
        gt_pose_xyz_max_bound = data_batch["gt_pose_xyz_max_bound"]

        pose_xyz = self.descale_xyz_pose(
            pose_xyz_scaled, gt_pose_xyz_min_bound, gt_pose_xyz_max_bound
        )

        outputs = {"pose_xyz_pred": pose_xyz}
        if "guide_losses" in cond_samp_out:
            outputs["guide_losses"] = cond_samp_out["guide_losses"]

        return outputs


if __name__ == "__main__":

    device = "cuda"
    num_epoch = 1000
    diffuser = Diffusion(
        loss_type="l1",
        beta_schedule="cosine",
        clip_denoised=True,
        predict_epsilon=False,
        supervise_epsilons=False,
        obs_dim=3,
        act_dim=2,
        hidden_dim=256,
        T=50,
        device=device,
    ).to("cuda")

    data_batch = {}
    data_batch["affordance"] = torch.randn(2, 2048, 1).to(device)
    data_batch["pc_position"] = torch.randn(2, 2048, 3).to(device)
    data_batch["object_name"] = ["black keyboard", "white mouse"]
    data_batch["object_pc_position"] = torch.randn(2, 512, 3).to(device)
    data_batch["gt_pose_4d"] = torch.randn(2, 80, 4).to(device)
    data_batch["gt_pose_4d_min_bound"] = torch.randn(2, 4).to(device)
    data_batch["gt_pose_4d_max_bound"] = torch.randn(2, 4).to(device)
    data_batch["pc_position_xy_affordance"] = torch.randn(2, 512, 2).to(device)
    data_batch["gt_pose_xyz"] = torch.randn(2, 80, 3).to(device)
    data_batch["gt_pose_xyz_min_bound"] = torch.randn(2, 3).to(device)
    data_batch["gt_pose_xyz_max_bound"] = torch.randn(2, 3).to(device)
    

    optimizer = torch.optim.Adam(diffuser.parameters(), lr=1e-3)

    for _ in range(2000):
        batch = data_batch
        for key in batch.keys():
            if key != "object_name":
                batch[key] = batch[key].to(device).float()
        data_batch = batch
        aux_info = diffuser.get_aux_info(data_batch)
        out_info = diffuser(data_batch)
        loss = diffuser.compute_losses(data_batch=data_batch)
        optimizer.zero_grad()
        loss["diffusion_loss"].backward()
        optimizer.step()

        print("pred: ", out_info["pose_xyz_pred"][0])
        print("  gt: ", data_batch["gt_pose_xyz"][0][0])
        print(" min:", data_batch["gt_pose_xyz_min_bound"][0])
        print(" max:", data_batch["gt_pose_xyz_max_bound"][0])
        print(f"step:{_}, loss: {loss['diffusion_loss'].item()}")

