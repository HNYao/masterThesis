import time
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy
from GeoL_diffuser.models.utils.guidance_loss import verify_guidance_config_list, DiffuserGuidance
from clip.model import build_model, load_clip, tokenize


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

class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, device, t_dim=16):
        super(MLP, self).__init__()
        
        self.device = device
        self.t_dim = t_dim
        self.a_dim = action_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim*2),
            nn.Mish(),
            nn.Linear(t_dim*2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )

        self.final_layer = nn.Linear(hidden_dim, action_dim)

        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight) # or xavier_normal
                nn.init.zeros_(m.bias)
    
    def forward(self, x, time, state):
        t_emb = self.time_mlp(time)
        x = torch.cat([x, state, t_emb], dim=1)
        x = self.mid_layer(x)
        return self.final_layer(x)

class AffordanceEncoder(nn.Module):
    def __init__(self, state_dim=1, hidden_dim=256, device="cuda"):
        super(AffordanceEncoder, self).__init__()
        self.device = device
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
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
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, state):
        return self.encoder(state).to(self.device)

class ObjectNameEncoder(nn.Module):
    """
    encode the object text name by clip
    """
    def __init__(self, out_dim, device) -> None:
        super().__init__()
        self.device = device
        self.out_dim = out_dim
        self._load_clip()

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

        return text_feat, text_emb

    def forward(self, x):
        text_feat, text_emb= self.encode_text(x)
        return text_feat

class MapUnet(nn.Module):
    def __init__(self,
                 transition_dim,
                 cond_dim, 
                 time_dim,
                 dim_mults = (1, 2, 4, 8)) -> None:
        super().__init__()

        dims = [transition_dim, *map(lambda m: time_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )

        condim = cond_dim + time_dim

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolution = len(in_out)
    



class Diffusion(nn.Module):
    def __init__(
            self, 
            loss_type, 
            beta_schedule="cosine", 
            clip_denoised=True, 
            predict_epsilon=True ,
            **kwargs
        ): 
        super(Diffusion, self).__init__()
        self.state_dim = kwargs["obs_dim"]
        self.action_dim = kwargs["act_dim"]
        self.hidden_dim = kwargs["hidden_dim"]
        self.T = kwargs["T"]
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon
        self.device = torch.device(kwargs["device"])

        self.model = MLP(self.state_dim, self.action_dim, self.hidden_dim, self.device).to(kwargs["device"])

        # feat extractor
        self.affordance_encoder = AffordanceEncoder(state_dim=1, hidden_dim=256, device=self.device)
        self.pc_position_encoder = PCPositionEncoder(state_dim=3, hidden_dim=256, device=self.device)
        self.obj_position_encoder = PCPositionEncoder(state_dim=3, hidden_dim=256, device=self.device)
        self.object_name_encoder = ObjectNameEncoder(out_dim=512, device=self.device)

        self.ema_model = deepcopy(self.model)
        self.ema = EMA(decay = 0.999)
        self.ema_decay = 0.999
        self.ema_start = 2000
        self.ema_update_freq = 1
        self.step = 0

        if beta_schedule == "linear":
            betas = torch.linspace(0.0001, 0.02, self.T, dtype=torch.float32, device=self.device) # beta params
        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps=self.T, dtype=torch.float32, device=self.device)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0).to(self.device) # e.g. [1, 2, 3] -> [1, 1*2, 1*2*3] 
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), alphas_cumprod[:-1]])

        # resigter as buffer
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # 前向过程
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))

        # 反向过程
        posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped",torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))
        self.register_buffer("posterior_mean_coef1",betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2",(1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

        # calculations for class-free guidance
        self.sqrt_alphas_over_one_minus_alphas_cumprod = torch.sqrt(alphas_cumprod / (1.0 - alphas_cumprod))
        self.sqrt_recip_one_minus_alphas_cumprod = 1.0 / torch.sqrt(1. - alphas_cumprod)

        ## get loss coefficients and initialize objective
        # TODO: complete the loss function
        self.loss_fn = Losses[loss_type]()

        # for guided sampling
        self.current_guidance = None
    
    #------------------------------------------ aux_info ------------------------------------------#
    def get_aux_info(self, data_batch):
        affordance = data_batch["affordance"] #[batch_size, num_points, affordance_dim=1]
        pc_position = data_batch["pc_position"] #[batch_size, num_points, pc_position_dim=3]
        object_name = data_batch["object_name"] #[batch_size, ]
        object_pc_position = data_batch["object_pc_position"] #[batch_size, obj_points, object_pc_position_dim=3]

        affordance_feat = self.affordance_encoder(affordance) # [batch_size, num_points, hidden_dim]
        pc_position_feat = self.pc_position_encoder(pc_position) # [batch_size, num_points, hidden_dim]
        object_name_feat = self.object_name_encoder(object_name) # [batch_size, hidden_dim]
        object_pc_position_feat = self.pc_position_encoder(object_pc_position) # [batch_size, obj_points, hidden_dim]

        # TODO: combine the feats
        aux_info = {
            "cond_feat": torch.cat([affordance_feat, pc_position_feat], dim=1),
            "non_cond_feat": torch.cat([affordance_feat, pc_position_feat], dim=1) # TODO: add new feats
        }

        return aux_info

        


    #------------------------------------------ scale and descale ------------------------------------------#
    def scale_pose(self, pose_4d, min_bound, max_bound):
        """
        scale the pose_4d to [-1, 1]
        """
        return 2 * (pose_4d - min_bound) / (max_bound - min_bound) - 1
    
    def descale_pose(self, pose_4d_scaled, min_bound, max_bound):
        """
        descale the pose_4d to the original range
        """
        return 0.5 * (pose_4d_scaled + 1) * (max_bound - min_bound) + min_bound

    #------------------------------------------ guidance utils ------------------------------------------#

    def set_guidance(self, guidance_config_list, example_batch=None):
        '''
        Instantiates test-time guidance functions using the list of configs (dicts) passed in.
        '''
        if guidance_config_list is not None:
            if len(guidance_config_list) > 0 and verify_guidance_config_list(guidance_config_list):
                print('Instantiating test-time guidance with configs:')
                print(guidance_config_list)
                self.current_guidance = DiffuserGuidance(guidance_config_list, example_batch)

    def update_guidance(self, **kwargs):
        if self.current_guidance is not None:
            self.current_guidance.update(**kwargs)

    def clear_guidance(self):
        self.current_guidance = None

    def update_ema(self):
        self.step += 1
        if self.step % self.ema_update_freq == 0:
            if self.step < self.ema_start:
                self.ema_model.load_state_dict(self.model.state_dict())
            else:
                self.ema.update_model_average(self.ema_model, self.model)
    #------------------------------------------ TBD ------------------------------------------#

    def get_loss_weights(self, action_weight, discount):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)
        ## manually set a0 weight
        loss_weights[0, -self.action_dim:] = action_weight

        return loss_weights

    def q_posterior(self, x_start, x, t):
        """
        x_start: 预测得到的x_0
        x: 当前步x
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x.shape) * x
        )
        posterior_variance = extract(self.posterior_variance, t, x.shape)
        posterior_log_variance = extract(
            self.posterior_log_variance_clipped, t, x.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance
    


    
    def predict_start_from_noise(self, x, t, pred_noise):
        """
            get the x_0 (e.g. denoised img) from x_t and noise 
            x_0 = xt - sqrt(1 - alpha_t cumprod) * noise / sqrt(alpha_t cumprod)

            x: x in step t
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * pred_noise
        )


    def p_mean_variance(self, x, t, aux_info):
        model_prediction = self.model(x, t, aux_info["cond_feat"])
        class_free_guid_w = 0.5 # NOTE: hard-coded for now
        if class_free_guid_w != 0:
            x_non_cond = x.clone()
            model_non_cond_prediction = self.model(x_non_cond, aux_info["non_cond_feat"], t=t)
            if not self.predict_epsilon:
                model_pred_noise = self.predict_noise_from_start(x_t=x, t=t, x_start=model_prediction)
                model_non_cond_pred_noise = self.predict_noise_from_start(x_t=x_non_cond, t=t, x_start=model_non_cond_prediction)
                class_free_guid_noise = (
                    (1 + class_free_guid_w) * model_pred_noise - class_free_guid_w * model_non_cond_pred_noise
                ) # compose noise

                model_prediction = self.predict_start_from_noise(x=x, t=t, pred_noise=class_free_guid_noise)
        x_recon = self.predict_start_from_noise(x, t, model_prediction)
        x_recon.clamp_(-1, 1)
        
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_recon, x, t)
        
        return model_mean, posterior_variance, posterior_log_variance, (x_recon, x, t) # log makes it stable

    def p_sample(self, x, t, data_batch, aux_info, num_samp=1, *args, **kwargs):
        """
        denosie, single step
        """
        b, *_, device = *x.shape, x.device

        # get the mean and variance
        model_mean, _, model_log_variance, q_posterior_in = self.p_mean_variance(x=x, t=t, aux_info=aux_info)

        # random noise
        noise = torch.randn_like(x)
        
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise # pred image



    def p_sample_loop(
            self, 
            shape, 
            data_batch,
            aux_info,
            num_samp,
            *args, 
            **kwargs):
        """
        denosise, loop
        """
        device = self.device
        batch_size = shape[0]
        x = torch.randn(shape, device=device) # random noise

        for i in reversed(range(0, self.T)): # reverse, denoise from the last step
            t = torch.full((batch_size * num_samp,), i, device=device, dtype=torch.long) # timestep
            x = self.p_sample(x, t, data_batch, aux_info, num_samp=num_samp, *args, **kwargs) # denoise
        
        # TODO: add unormalize
        out_dict = {"pred_pose_4d": x}

        return out_dict


    def condition_sample(self, data_batch, aux_info, num_samp=1,  *args, **kwargs):
        """
        TODO
        """
        batch_size = data_batch["affordance"].shape[0]
        
        shape = (batch_size, num_samp, self.state_dim)
        action = self.p_sample_loop(shape, data_batch, aux_info, num_samp=num_samp, *args, **kwargs)
        return action.clamp_(-1, 1)

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
    
    def p_losses(self, x_start, state, t, weights=1.0):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start, t, noise)

        x_recon = self.model(x_noisy, t, state)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)
        return loss
    
    def loss(self, x, state, weights=1.0):
        batch_size = len(x)
        t = torch.randint(0, self.T, (batch_size,), device=self.device).long()
        return self.p_losses(x, state, t, weights)
    

    def forward(self, data_batch, num_samp=1, *args, **kwargs):
        aux_info = self.get_aux_info(data_batch)
        cond_samp_out = self.condition_sample(
            data_batch,
            horizon=None,
            aux_info=aux_info,
            num_samp=num_samp, 
            *args, 
            **kwargs)
        pose_4d_scaled = cond_samp_out["pred_pose_4d"]

        gt_pose_4d_min_bound = data_batch["gt_pose_4d_min_bound"]
        gt_pose_4d_max_bound = data_batch["gt_pose_4d_max_bound"]

        pose_4d = self.descale_pose(pose_4d_scaled, gt_pose_4d_min_bound, gt_pose_4d_max_bound)

        outputs = {"predictions": pose_4d}
        
        return outputs


if __name__ == "__main__":

    device = "cuda"
    num_epoch = 1000
    diffuser = Diffusion(
        loss_type="l1",
        beta_schedule="cosine",
        clip_denoised=True,
        predict_epsilon=True,
        obs_dim=4,
        act_dim=2,
        hidden_dim=256,
        T=16,
        device=device
    ).to("cuda")

    data_batch = {}
    data_batch["affordance"] = torch.randn(2, 2048, 1).to(device)
    data_batch["pc_position"] = torch.randn(2, 2048, 3).to(device)
    data_batch["object_name"] = ["black keyboard", "white mouse"]
    data_batch["object_pc_position"] = torch.randn(2, 512, 3).to(device)


    aux_info = diffuser.get_aux_info(data_batch)
    
    for _ in range(1):
        out_info = diffuser.condition_sample(data_batch, aux_info, num_samp=1)