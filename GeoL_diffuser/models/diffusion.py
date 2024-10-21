import time
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy
from GeoL_diffuser.models.utils.guidance_loss import verify_guidance_config_list


def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
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
    return torch.tensor(betas_clipped, dtype=dtype)

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


class Diffusion(nn.Module):
    def __init__(
            self, 
            loss_type, 
            action_weight=1.0, 
            loss_discount=1.0, 
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

        self.ema_model = deepcopy(self.model)
        self.ema = EMA(decay = 0.999)
        self.ema_decay = 0.999
        self.ema_start = 2000
        self.ema_update_freq = 1
        self.step = 0

        if beta_schedule == "linear":
            betas = torch.linspace(0.0001, 0.02, self.T, dtype=torch.float32, device=self.device) # beta params
        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps=self.T, dtype=torch.float32)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0) # e.g. [1, 2, 3] -> [1, 1*2, 1*2*3] 
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


    def p_mean_variance(self, x, t, s):
        pred_noise = self.model(x, t, s) # noise predicted by the model
        x_recon = self.predict_start_from_noise(x, t, pred_noise) # x_0 predicted based on the model predicted from the model
        x_recon.clamp_(-1, 1) # for stability
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_recon, x, t)
        return model_mean, posterior_log_variance # log makes it stable

    def p_sample(self, x, t, s):
        """
        denosie, single step
        """
        b, *_, device = *x.shape, x.device

        # get the mean and variance
        model_mean, model_log_variance = self.p_mean_variance(x, t, s)

        # random noise
        noise = torch.randn_like(x)
        
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise # pred image



    def p_sample_loop(self, state, shape, *args, **kwargs):
        """
        denosise, loop
        """
        device = self.device
        batch_size = state.shape[0]
        x = torch.randn(shape, device=device, requires_grad=False)

        for i in reversed(range(0, self.T)): # reverse, denoise from the last step
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, state)
        
        # TODO: add unormalize

        return x


    def sample(self, state, *args, **kwargs):
        """
        state: [batch_size, state_dim]
        """
        batch_size = state.shape[0]
        shape = [batch_size, self.action_dim]
        action = self.p_sample_loop(state, shape, *args, **kwargs)
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
    

    def forward(self, state, *args, **kwargs):
        return self.sample(state, *args, **kwargs)


if __name__ == "__main__":

    device = "cpu"
    num_epoch = 1000
    x = torch.randn(256, 2, device=device)
    state = torch.randn(256, 11, device=device)
    model = Diffusion(loss_type="l2",predict_epsilon=True, obs_dim=11, act_dim=2, hidden_dim=256,  T=100, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

    for i in range(num_epoch):
        action = model(state)
        loss = model.loss(x, state)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        print(f"epoch: {i}, loss: {loss.item()}")

    action = model(state)

    loss = model.loss(x, state)

    print(f"action: {action}, loss: {loss.item()}")

        