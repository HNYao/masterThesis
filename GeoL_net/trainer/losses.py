import torch
import torch.nn as nn
from omegaconf import DictConfig

from GeoL_net.core.registry import registry
from torch.nn import functional as F


@registry.register_loss_fn(name="focal")
class FocalLoss(nn.Module):
    def __init__(self, config:DictConfig, alpha=0.25, gamma=5.0):
        super().__init__()
        self.config = config
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction="none")(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


@registry.register_loss_fn(name="soft_dice")
class SoftDiceLoss(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

    def forward(self, inputs, targets, dims=(1, 2)):
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum(dims)
        print("shape of inputs:", inputs.shape)
        print("shape of targets:", targets.shape)
        union = inputs.sum(dims) + targets.sum(dims)
        dice = (2 * intersection) / (union + self.config.eps)
        return 1 - dice.mean()

@registry.register_loss_fn(name="binary_ce")
class BinaryCELoss(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

    def forward(self, inputs, targets):
        # inputs as logits
        loss = F.binary_cross_entropy_with_logits(inputs, targets)
        return loss