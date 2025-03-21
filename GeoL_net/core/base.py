import os
import random
from abc import abstractmethod
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchvision import transforms

import wandb
from GeoL_net.core.logger import logger
from GeoL_net.utils.ddp_utils import (
    get_distrib_size,
    init_distrib_slurm,
    rank0_only,
)


class BaseTrainer:
    def __init__(
        self,
        cfg: DictConfig,
        dataset_dir: str,
        checkpoint_dir: str,
        log_dir: str,
    ) -> None:
        self.dataset_dir = dataset_dir

        self.log_dir: str = log_dir
        self.checkpoint_dir: str = checkpoint_dir
        self.cfg = cfg

        self.batch_size: int = self.cfg.training.batch_size
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.input_shape = (3, 480, 640)
        self.target_input_shape = (3, 128, 128)
        self.eps = 1e-7
        self.local_rank = 0
        self.activation = nn.Sigmoid()

        self.log_interval = self.cfg.training.log_interval

        self.init_distrib()
        self.init_model()
        self.init_dataset()
        self.init_dirs()
        self.init_wandb()

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def init_distrib(self):
        self.n_gpus = get_distrib_size()[2]
        self._is_distributed = self.n_gpus > 1
        logger.info(
            "Number of GPUs:{} - {}".format(self.n_gpus, self._is_distributed)
        )
        if self._is_distributed:
            self.local_rank, self.tcp_store = init_distrib_slurm(
                self.cfg.distributed.backend
            )
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.device)
        self.seed(self.cfg.seed)

    def init_wandb(self) -> None:
        if rank0_only():
            wandb.init(
                project=self.cfg.wandb.project,
                entity=self.cfg.wandb.entity,
                name=self.cfg.wandb.name,
                job_type=self.cfg.wandb.job_type,
            )

    def init_dirs(self) -> None:
        if self.cfg.visualization.visualize:
            os.makedirs(self.cfg.visualization.output_dir, exist_ok=True)

        if os.path.isfile(self.checkpoint_dir):
            return
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def log(self, step, metrics):
        wandb.log({**metrics, "step": step})

    def log_img(self, step, img_pil, cls,  phrase="None", file_name="None"):
        image = wandb.Image(img_pil, caption=f"{file_name} -- {phrase}--STEP: {step}")
        wandb.log({f"{cls}": image})

    def log_img_batch(self, step, img_batch, phrase="None", file_name="None"):
        wandb.log({"batch_images": [wandb.Image(img) for img in img_batch]})
    
    def log_table(self, step, predictions, batch):
        table = wandb.Table(columns=["Pred","GT", "min", "max"])
        for i, pred in enumerate(predictions):
            table.add_data(predictions[i], batch["gt_pose_xyR"][i], batch["gt_pose_xyR_min_bound"][i], batch["gt_pose_xyR_max_bound"][i])
        wandb.log({"Prediction training dataset": table})

    def log_img_table(self, step, img_batch, gt_batch, phrase="None", file_name="None"):
        if phrase != "None":
            wandb_Images = [wandb.Image(img, caption=f"{phrase[i]}-step:{step}") for i, img in enumerate(img_batch)]
            wandb_Images_gt = [wandb.Image(img, caption=f"{gt_batch[i]}-step:{step}") for i, img in enumerate(gt_batch)]
            table = wandb.Table(columns=["Pred","GT", "phrase", "file_name"])
            for i, img in enumerate(wandb_Images):
                table.add_data(img,wandb_Images_gt[i],phrase[i], file_name[i])
            wandb.log({"Image Grid with Captions": table})
        else:
            wandb_Images = [wandb.Image(img, caption=f"step:{step}") for img in img_batch]
            wandb_Images_gt = [wandb.Image(img, caption=f"step:{step}") for img in gt_batch]
            table = wandb.Table(columns=["Pred","GT"])
            for i, img in enumerate(wandb_Images):
                table.add_data(img,wandb_Images_gt[i])
            wandb.log({"Image Grid with Captions": table})

    @abstractmethod
    def init_dataset(self) -> None:
        pass

    @abstractmethod
    def init_model(self) -> None:
        pass

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def eval(self, path: str) -> None:
        pass


class BaseTransform:
    def __init__(self) -> None:
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def __call__(
        self,
        x: Union[np.ndarray, torch.Tensor],
        masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Any:
        return self.apply(x, masks)

    def apply(
        self,
        x: Union[np.ndarray, torch.Tensor],
        masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        if masks is None:
            return self.transforms(x)
        if isinstance(x, torch.Tensor):
            return x / 255.0, masks
        return self.transforms(x), masks
