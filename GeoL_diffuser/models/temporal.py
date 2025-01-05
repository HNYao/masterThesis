#
# Based on Diffuser: https://github.com/jannerm/diffuser/blob/main/diffuser/models/temporal.py
#

import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import math
from GeoL_diffuser.models.layers_2d import (
    Downsample1d,
    DownsampleMLP,
    Upsample1d,
    UpsampleMLP,
    Conv1dBlock,
    PerceiverBlock,
)

from GeoL_net.models.GeoL import FeaturePerceiver


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


class ResidualTemporalMapBlockConcat(nn.Module):

    def __init__(
        self, inp_channels, out_channels, time_embed_dim, horizon, kernel_size=5
    ):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(time_embed_dim, out_channels),
            Rearrange("batch t -> batch t 1"),
        )

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(inp_channels, out_channels, kernel_size),
                Conv1dBlock(out_channels, out_channels, kernel_size),
            ]
        )

        self.residual_conv = (
            nn.Conv1d(inp_channels, out_channels, 1)
            if inp_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t):
        """
        x : [ batch_size x inp_channels x horizon ]
        t : [ batch_size x embed_dim ]
        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class TemporalMapUnet(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,  # additional dimension concatenated with the time dimension
        output_dim,
        dim=32,  # time_dimesion
        dim_mults=(1, 2, 4, 8),
        use_perceiver=False,
    ):
        super().__init__()

        ResidualTemporalMapBlock = ResidualTemporalMapBlockConcat

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f"[ models/temporal ] Channel dimensions: {in_out}")

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )

        cond_dim = cond_dim + time_dim

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # Remember the property of the 1D convolution, [B, C_in, L_in] => [B, C_out, L_out]
        # L_out is dependent on the kernel size and stride, and L_in

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResidualTemporalMapBlock(
                            dim_in, dim_out, time_embed_dim=cond_dim, horizon=horizon
                        ),  # Feature dimension changes, no horizon changes
                        ResidualTemporalMapBlock(
                            dim_out, dim_out, time_embed_dim=cond_dim, horizon=horizon
                        ),
                        (
                            Downsample1d(dim_out) if not is_last else nn.Identity()
                        ),  # No feature dimension changes, but horizon changes
                    ]
                )
            )

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalMapBlock(
            mid_dim, mid_dim, time_embed_dim=cond_dim, horizon=horizon
        )
        self.mid_block2 = ResidualTemporalMapBlock(
            mid_dim, mid_dim, time_embed_dim=cond_dim, horizon=horizon
        )

        final_up_dim = None
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ResidualTemporalMapBlock(
                            dim_out * 2,
                            dim_in,
                            time_embed_dim=cond_dim,
                            horizon=horizon,
                        ),  # Feature dimension changes, no horizon changes
                        ResidualTemporalMapBlock(
                            dim_in, dim_in, time_embed_dim=cond_dim, horizon=horizon
                        ),
                        (
                            Upsample1d(dim_in) if not is_last else nn.Identity()
                        ),  # No feature dimension change, but horizon changes
                    ]
                )
            )
            final_up_dim = dim_in

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(final_up_dim, final_up_dim, kernel_size=5),
            nn.Conv1d(final_up_dim, output_dim, 1),
        )
        self.use_perceiver = use_perceiver

        if self.use_perceiver:
            print(f"[ models/temporal ] Using Perceiver")
            self.preceiver = FeaturePerceiver(
                transition_dim=transition_dim,
                condition_dim=cond_dim - time_dim,
                time_emb_dim=time_dim,
            )
            self.proj = nn.Linear(self.preceiver.last_dim, transition_dim)

    def forward(self, x, cond, time):
        """
        x : [ batch x horizon x transition ]
        cond: [ batch x cond_dim ]
        time: [ batch ]
        """
        t = self.time_mlp(time)

        if self.use_perceiver:
            x = self.preceiver(x, cond[:, None], t[:, None])
            x = self.proj(x)
        x = einops.rearrange(x, "b h t -> b t h")
        t = torch.cat([t, cond], dim=-1)  # [time+object+action+spatial]

        h = []
        for ii, (resnet, resnet2, downsample) in enumerate(self.downs):
            x = resnet(x, t)
            x = resnet2(x, t)

            h.append(x)
            x = downsample(
                x
            )  # Increase the feature dimension, reduce the horizon (consider the spatial resolution in image)
            # print("Downsample step {}, with shape {}".format(ii, x.shape)) # [B, C, H]
            # print(f"[ models/temporal ] Downsample step {ii}, with shape {x.shape}")

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)
        for ii, (resnet, resnet2, upsample) in enumerate(self.ups):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)  # Decrease the feature dimension, increase the horizon
            # print("Upsample step {}, with shape {}".format(ii, x.shape)) # [B, C, H]

        x = self.final_conv(x)
        x = einops.rearrange(x, "b t h -> b h t")
        return x


class TemporalMapUnet_v2(nn.Module):
    """
        remove the conv1d on the [batch, horizon, transition]
        use MLP on [batch, horizon * transition] transition
            [batch, horizon, transition] -> [batch, horizon * transition] -> [batch, horizon, transition]
    """
    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,  # additional dimension concatenated with the time dimension
        output_dim,
        dim=32,  # time_dimesion
        dim_mults=(1, 2, 4, 8),
        use_perceiver=False,
    ):
        super().__init__()

        PerceiverTemporalMapBlock = PerceiverBlock

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f"[ models/temporal ] Channel dimensions: {in_out}")

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )

        cond_dim = cond_dim

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # Remember the property of the 1D convolution, [B, C_in, L_in] => [B, C_out, L_out]
        # L_out is dependent on the kernel size and stride, and L_in

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        PerceiverTemporalMapBlock(
                            transition_dim = dim_in,
                            condition_dim = cond_dim,
                            time_emb_dim = time_dim,
                            decoder_q_input_channels = dim_out
                        ),  # Feature dimension changes, no horizon changes
                        PerceiverTemporalMapBlock(
                            transition_dim = dim_out,
                            condition_dim = cond_dim,
                            time_emb_dim = time_dim,
                            decoder_q_input_channels = dim_out
                        ),
                        (
                            DownsampleMLP(dim_out) if not is_last else nn.Identity()
                        ),  # No feature dimension changes, but horizon changes
                    ]
                )
            )

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = PerceiverTemporalMapBlock(
            transition_dim=mid_dim, condition_dim=cond_dim, time_emb_dim=time_dim, decoder_q_input_channels=mid_dim
        )
        self.mid_block2 = PerceiverTemporalMapBlock(
            transition_dim=mid_dim, condition_dim=cond_dim, time_emb_dim=time_dim, decoder_q_input_channels=mid_dim
        )

        final_up_dim = None
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        PerceiverBlock(
                            transition_dim=dim_out * 2,
                            condition_dim=cond_dim,
                            time_emb_dim=time_dim,
                            decoder_q_input_channels=dim_in,
                        ),  # Feature dimension changes, no horizon changes
                        PerceiverBlock(
                            transition_dim=dim_in,
                            condition_dim=cond_dim,
                            time_emb_dim=time_dim,
                            decoder_q_input_channels=dim_in,
                        ),
                        (
                            UpsampleMLP(dim_in) if not is_last else nn.Identity()
                        ),  # No feature dimension change, but horizon changes
                    ]
                )
            )
            final_up_dim = dim_in

            if not is_last:
                horizon = horizon * 2

        self.final_perceiver = FeaturePerceiver(
            transition_dim=dim_in,
            condition_dim=cond_dim,
            time_emb_dim=-1,
            encoder_num_heads=1,
            decoder_num_heads=1,
            decoder_q_input_channels=output_dim
        )
    
        self.use_perceiver = use_perceiver

        if self.use_perceiver:
            print(f"[ models/temporal ] Using Perceiver")
            self.perceiver = FeaturePerceiver(
                transition_dim=transition_dim,
                condition_dim=cond_dim,
                time_emb_dim=time_dim,
            )
            self.proj = nn.Linear(self.perceiver.last_dim, transition_dim)

    def forward(self, x, cond, time):
        """
        x : [ batch , horizon * transition ]
        cond: [ batch , cond_dim ]
        time: [ batch ]
        """
        B = x.size(0)
        t = self.time_mlp(time)

        if self.use_perceiver:
            x = self.perceiver(x, cond[:, None], t[:, None]) # [batch, horizon, 256]
            x = self.proj(x) # [batch, horizon, transition]
        #x = einops.rearrange(x, "b h t -> b t h") # [batch, transition, horizon]
        #t = torch.cat([t, cond], dim=-1)  # [time+object+action+spatial]

        h = []
        for ii, (resnet, resnet2, downsample) in enumerate(self.downs):
            x = resnet(x, cond[:,None], t[:, None]) # should be like [B, T, H], but [B, H, T]
            x = resnet2(x, cond[:, None], t[:, None]) # [B, 64, H]
            
            h.append(x) # [B, H, T] in h
            x = downsample(
                x
            )  # Increase the feature dimension, reduce the horizon (consider the spatial resolution in image)
            # print("Downsample step {}, with shape {}".format(ii, x.shape)) # [B, C, H]
            # print(f"[ models/temporal ] Downsample step {ii}, with shape {x.shape}")

        x = self.mid_block1(x, cond[:, None], t[:, None])
        x = self.mid_block2(x, cond[:, None], t[:, None])
        for ii, (resnet, resnet2, upsample) in enumerate(self.ups):
            x = torch.cat((x, h.pop()), dim=-1)
            x = resnet(x, cond[:, None],t[:, None])
            x = resnet2(x, cond[:, None], t[:, None])
            x = upsample(x)  # Decrease the feature dimension, increase the horizon
            # print("Upsample step {}, with shape {}".format(ii, x.shape)) # [B, C, H]
        B = x.size(0)
        H = x.size(1)
        x = self.final_perceiver(x, cond[:, None])
        return x



class TemporalMapUnet_v3(nn.Module):
    """
        remove the conv1d on the [batch, horizon, transition]
        use MLP on [batch, horizon * transition] transition
            [batch, horizon, transition] -> [batch, horizon * transition] -> [batch, horizon, transition]
    
        transfer [batch, horizon, transition ] -> [batch, h*t, 1]
        and without downsample and upsample
    """
    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,  # additional dimension concatenated with the time dimension
        output_dim,
        dim=32,  # time_dimesion
        dim_mults=(1, 2, 4, 8),
        use_perceiver=False,
    ):
        super().__init__()

        #ResidualTemporalMapBlock = ResidualTemporalMapBlockConcat
        PerceiverTemporalMapBlock = PerceiverBlock

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f"[ models/temporal ] Channel dimensions: {in_out}")

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )

        cond_dim = cond_dim

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # Remember the property of the 1D convolution, [B, C_in, L_in] => [B, C_out, L_out]
        # L_out is dependent on the kernel size and stride, and L_in

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        PerceiverTemporalMapBlock(
                            transition_dim = dim_in,
                            condition_dim = cond_dim,
                            time_emb_dim = time_dim,
                            decoder_q_input_channels = dim_out
                        ),  # Feature dimension changes, no horizon changes
                        PerceiverTemporalMapBlock(
                            transition_dim = dim_out,
                            condition_dim = cond_dim,
                            time_emb_dim = time_dim,
                            decoder_q_input_channels = dim_out
                        ),
                        #(
                        #    DownsampleMLP(dim_out) if not is_last else nn.Identity()
                        #),  # No feature dimension changes, but horizon changes
                    ]
                )
            )

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = PerceiverTemporalMapBlock(
            transition_dim=mid_dim, condition_dim=cond_dim, time_emb_dim=time_dim, decoder_q_input_channels=mid_dim
        )
        self.mid_block2 = PerceiverTemporalMapBlock(
            transition_dim=mid_dim, condition_dim=cond_dim, time_emb_dim=time_dim, decoder_q_input_channels=mid_dim
        )

        final_up_dim = None
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        PerceiverBlock(
                            transition_dim=dim_out * 2,
                            condition_dim=cond_dim,
                            time_emb_dim=time_dim,
                            decoder_q_input_channels=dim_in,
                        ),  # Feature dimension changes, no horizon changes
                        PerceiverBlock(
                            transition_dim=dim_in,
                            condition_dim=cond_dim,
                            time_emb_dim=time_dim,
                            decoder_q_input_channels=dim_in,
                        ),
                        #(
                        #    UpsampleMLP(dim_in) if not is_last else nn.Identity()
                        #),  # No feature dimension change, but horizon changes
                    ]
                )
            )
            final_up_dim = dim_in

            if not is_last:
                horizon = horizon * 2
        ### preivious version, using conv1d
        #self.final_conv = nn.Sequential(
        ##    Conv1dBlock(final_up_dim, final_up_dim, kernel_size=5),
        #    nn.Conv1d(final_up_dim, output_dim, 1),
        #)
        ### new version, using perceiver
        self.final_mlp = nn.Sequential(
            nn.Linear((final_up_dim) * transition_dim , (final_up_dim) * transition_dim  * 2),
            nn.Mish(),
            nn.Linear((final_up_dim) * transition_dim * 2, final_up_dim * transition_dim * 2),
            nn.Mish(),
            nn.Linear(final_up_dim * transition_dim  * 2, output_dim),
            
        )

        self.concat_mlp = nn.Sequential(
            nn.Linear(6, 6 * 2),
            nn.Mish(),
            nn.Linear(6 * 2, 3),
        )
    
        self.use_perceiver = use_perceiver

        if self.use_perceiver:
            print(f"[ models/temporal ] Using Perceiver")
            self.perceiver = FeaturePerceiver(
                transition_dim=1,
                condition_dim=cond_dim,
                time_emb_dim=time_dim,
            )
            self.proj = nn.Linear(self.perceiver.last_dim, transition_dim)

    def forward(self, x, cond, time):
        """
        x : [ batch ,  horizon * transition ]
        cond: [ batch ,  cond_dim ]
        time: [ batch ]
        """
        # transfer [batch, horizon, transition ] -> [batch, h*t, 1]
        B = x.size(0)
        H = x.size(1)

        x = x.view(x.size(0), -1, 1)
        t = self.time_mlp(time)


        if self.use_perceiver:
            x = self.perceiver(x, cond[:, None], t[:, None]) # [batch, horizon, 256]
            x = self.proj(x) # [batch, horizon, transition]
        #x = einops.rearrange(x, "b h t -> b t h") # [batch, transition, horizon]
        #t = torch.cat([t, cond], dim=-1)  # [time+object+action+spatial]

        h = []
        for ii, (resnet, resnet2) in enumerate(self.downs):
            x = resnet(x, cond[:,None], t[:, None]) # should be like [B, T, H], but [B, H, T]
            x = resnet2(x, cond[:, None], t[:, None]) # [B, 64, H]
            
            h.append(x) # [B, H, T] in h
            #x = downsample(x)  # Increase the feature dimension, reduce the horizon (consider the spatial resolution in image)
            # print("Downsample step {}, with shape {}".format(ii, x.shape)) # [B, C, H]
            # print(f"[ models/temporal ] Downsample step {ii}, with shape {x.shape}")

        x = self.mid_block1(x, cond[:, None], t[:, None])
        x = self.mid_block2(x, cond[:, None], t[:, None])
        for ii, (resnet, resnet2) in enumerate(self.ups):
            x = torch.cat((x, h.pop()), dim=-1)
            x = resnet(x, cond[:, None],t[:, None])
            x = resnet2(x, cond[:, None], t[:, None])
            #x = upsample(x)  # Decrease the feature dimension, increase the horizon
            # print("Upsample step {}, with shape {}".format(ii, x.shape)) # [B, C, H]

        x = x.view(B, H, -1)
        x = self.final_mlp(x)

        x = x.view(B, H, -1)
        cond = cond.view(B, -1, 3)
        x = torch.cat((x, cond), dim=-1)
        x = self.concat_mlp(x)

        x = x*0 + cond
        return x

class TemporalMapUnet_v4(nn.Module):
    """
        remove the conv1d on the [batch, horizon, transition]
        use MLP on [batch, horizon * transition] transition
            [batch, horizon, transition] -> [batch, horizon * transition] -> [batch, horizon, transition]
    
        transfer [batch, horizon, transition ] -> [batch, h*t, 1]
        and without downsample and upsample
    """
    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,  # additional dimension concatenated with the time dimension
        output_dim,
        dim=32,  # time_dimesion
        dim_mults=(1, 2, 4, 8),
        use_perceiver=False,
    ):
        super().__init__()

        PerceiverTemporalMapBlock = PerceiverBlock



        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )

        cond_dim = cond_dim
    
        self.use_perceiver = use_perceiver

        if self.use_perceiver:
            print(f"[ models/temporal ] Using Perceiver")
            self.perceiver = FeaturePerceiver(
                transition_dim=transition_dim,
                condition_dim=cond_dim,
                time_emb_dim=time_dim,
                decoder_num_heads=1,
                decoder_q_input_channels=output_dim
            )
            self.proj = nn.Linear(self.perceiver.last_dim, transition_dim)

    def forward(self, x, cond, time):
        """
        x : [ batch ,  horizon * transition ]
        cond: [ batch ,  cond_dim ]
        time: [ batch ]
        """
        # transfer [batch, horizon, transition ] -> [batch, h*t, 1]
        B = x.size(0)
        H = x.size(1)
        #x = x.view(x.size(0), -1, 1)
        t = self.time_mlp(time)


        x = self.perceiver(x, cond[:, None], t[:, None]) # [batch, horizon, 256]
        return x

class TemporalMapUnet_v5(nn.Module):
    """
        remove the conv1d on the [batch, horizon, transition]
        use MLP on [batch, horizon * transition] transition
            [batch, horizon, transition] -> [batch, horizon * transition] -> [batch, horizon, transition]
    
        transfer [batch, horizon, transition ] -> [batch, h*t, 1]
        and without downsample and upsample
    """
    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,  # additional dimension concatenated with the time dimension
        output_dim,
        dim=32,  # time_dimesion
        dim_mults=(1, 2, 4, 8),
        use_perceiver=False,
    ):
        super().__init__()

        PerceiverTemporalMapBlock = PerceiverBlock



        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )

        cond_dim = cond_dim
    
        self.use_perceiver = use_perceiver

        if self.use_perceiver:
            print(f"[ models/temporal ] Using Perceiver")
            self.perceiver = FeaturePerceiver(
                transition_dim=transition_dim,
                condition_dim=cond_dim,
                time_emb_dim=time_dim,
                decoder_num_heads=1,
                decoder_q_input_channels=output_dim
            )
        self.proj = nn.Sequential(
                nn.Linear(output_dim + 3, output_dim * 2),
                nn.Mish(),
                nn.Linear(output_dim * 2, output_dim),
        )


    def forward(self, x, cond, time):
        """
        x : [ batch ,  horizon * transition ]
        cond: [ batch ,  cond_dim ]
        time: [ batch ]
        """
        # transfer [batch, horizon, transition ] -> [batch, h*t, 1]
        B = x.size(0)
        H = x.size(1)
        #x = x.view(x.size(0), -1, 1)
        t = self.time_mlp(time)
        x = self.perceiver(x, cond[:, None], t[:, None]) # [batch, horizon, 3]
        x = torch.cat((x, cond.view(B, -1, 3)), dim=-1) # [batch, horizon, 6]
        x = self.proj(x)
        return x


class TemporalMapUnet_v6(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,  # additional dimension concatenated with the time dimension
        output_dim,
        dim=32,  # time_dimesion
        dim_mults=(1, 2, 4, 8),
        use_perceiver=False,
    ):
        super().__init__()

        PerceiverTemporalMapBlock = PerceiverBlock



        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )

        cond_dim = cond_dim
    
        self.use_perceiver = use_perceiver

        if self.use_perceiver:
            print(f"[ models/temporal ] Using Perceiver")
            self.perceiver = FeaturePerceiver(
                transition_dim=transition_dim,
                condition_dim=cond_dim,
                time_emb_dim=time_dim,
                decoder_num_heads=1,
                decoder_q_input_channels=output_dim
            )
        self.proj = nn.Sequential(
                nn.Linear(transition_dim + 3, output_dim * 2),
                nn.Mish(),
                nn.Linear(output_dim * 2, output_dim),
        )


    def forward(self, x, cond, time):
        """
        x : [ batch ,  horizon * transition ]
        cond: [ batch ,  cond_dim ]
        time: [ batch ]
        """
        # transfer [batch, horizon, transition ] -> [batch, h*t, 1]
        B = x.size(0)
        H = x.size(1)
        x = torch.cat((x, cond.view(B, -1, 3)), dim=-1) # [batch, horizon, 6]
        x = self.proj(x)
        return x

class TemporalMapUnet_v7(nn.Module):
    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,  # additional dimension concatenated with the time dimension
        output_dim,
        dim=32,  # time_dimesion
        dim_mults=(1, 2, 4, 8),
        use_perceiver=False,
    ):
        super().__init__()

        # Feature extraction layers for x
        self.x_feature_extractor = nn.Sequential(
            nn.Linear(4, 64),
            nn.Mish(),
            nn.Linear(64, 64),
            nn.Mish()
        )
        
        # Feature extraction layers for condition
        self.condition_feature_extractor = nn.Sequential(
            nn.Linear(3, 64),
            nn.Mish(),
            nn.Linear(64, 64),
            nn.Mish()
        )
        
        # Fusion and output layers
        self.fusion = nn.Sequential(
            nn.Linear(128, 128),
            nn.Mish(),
            nn.Linear(128, 64),
            nn.Mish(),
            nn.Linear(64, 3)  # Output shape matches the condition [batch, 80, 3]
        )

    def forward(self, x, condition, time):
        batch_size, seq_len, x_dim = x.size()
        cond_dim = 3


        # Flatten the inputs for processing
        x = x.view(-1, x_dim)  # [batch * 80, 4]
        condition = condition.view(-1, cond_dim)  # [batch * 80, 3]

        # Extract features
        x_features = self.x_feature_extractor(x)  # [batch * 80, 64]
        condition_features = self.condition_feature_extractor(condition)  # [batch * 80, 64]

        # Concatenate features
        combined = torch.cat([x_features, condition_features], dim=1)  # [batch * 80, 128]

        # Apply fusion and reshape to output shape
        output = self.fusion(combined)  # [batch * 80, 3]
        output = output.view(batch_size, seq_len, -1)  # [batch, 80, 3]
        output = output + condition.view(batch_size, -1, cond_dim)  # [batch, 80, 3]

        return output


if __name__ == "__main__":
    model = TemporalMapUnet_v2(
        horizon=80,  # time horizon
        transition_dim=3,  # dimension of the input trajectory
        cond_dim=240,  # dimension of the condition (from image, depth, text, etc.)
        output_dim=3,  # dimension of the output trajectory
        dim=32,  # base feature dimension
        dim_mults=(2, 4, 8),  # number of the layers
        use_perceiver=True,
    )
    x = torch.randn(2, 80, 3)
    cond = torch.randn(2, 240)
    time = torch.randn(2)
    print("Input shape: ", x.permute(0, 2, 1).shape)  # [B, input_dim, H]
    out = model(x, cond, time)  # [B, dim', H']
    print("Output shape: ", out.permute(0, 2, 1).shape)  # [B, outpu_dim, H]

    # Input shape:  torch.Size([2, 3, 20])
    # Downsample step 0, with shape torch.Size([2, 64, 10])
    # Downsample step 1, with shape torch.Size([2, 128, 5])
    # Downsample step 2, with shape torch.Size([2, 256, 5])
    # Upsample step 0, with shape torch.Size([2, 128, 10])
    # Upsample step 1, with shape torch.Size([2, 64, 20])
    # Output shape:  torch.Size([2, 3, 20])
