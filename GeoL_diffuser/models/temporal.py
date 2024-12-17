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
    Upsample1d,
    Conv1dBlock,
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
        ### preivious version, using conv1d
        self.final_conv = nn.Sequential(
            Conv1dBlock(final_up_dim, final_up_dim, kernel_size=5),
            nn.Conv1d(final_up_dim, output_dim, 1),
        )
        ### new version, using MLP
        self.final_mlp = nn.Sequential(
            nn.Linear((final_up_dim) * horizon, (final_up_dim) * horizon * 2),
            nn.Mish(),
            nn.Linear((final_up_dim) * horizon * 2, final_up_dim * horizon * 2),
            nn.Mish(),
            nn.Linear(final_up_dim * horizon * 2, output_dim * horizon),
            Rearrange("batch (output_dim horizon) -> batch output_dim horizon", output_dim=output_dim),
        )
    
        self.use_perceiver = use_perceiver

        if self.use_perceiver:
            print(f"[ models/temporal ] Using Perceiver")
            self.perceiver = FeaturePerceiver(
                transition_dim=transition_dim,
                condition_dim=cond_dim - time_dim,
                time_emb_dim=time_dim,
            )
            self.proj = nn.Linear(self.perceiver.last_dim, transition_dim)

    def forward(self, x, cond, time):
        """
        x : [ batch x horizon * transition ]
        cond: [ batch x cond_dim ]
        time: [ batch ]
        """
        t = self.time_mlp(time)


        if self.use_perceiver:
            x = self.perceiver(x, cond[:, None], t[:, None])
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
        B = x.size(0)
        H = x.size(2)
        x_viewed = x.view(x.size(0), -1) # [B, T*H]
        x = self.final_mlp(x_viewed).view(B, -1, H)
        x = einops.rearrange(x, "b t h -> b h t")
        return x

if __name__ == "__main__":
    model = TemporalMapUnet_v2(
        horizon=80,  # time horizon
        transition_dim=87,  # dimension of the input trajectory
        cond_dim=32,  # dimension of the condition (from image, depth, text, etc.)
        output_dim=3,  # dimension of the output trajectory
        dim=32,  # base feature dimension
        dim_mults=(2, 4, 8),  # number of the layers
        use_perceiver=True,
    )
    x = torch.randn(2, 80, 87)
    cond = torch.randn(2, 32)
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
