import glob
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import cv2
import hydra
import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from GeoL_net.core.base import BaseTrainer
from GeoL_net.core.logger import logger
from GeoL_net.core.registry import registry
from GeoL_net.utils.ddp_utils import rank0_only


@registry.register_trainer(name="GeoL_net_trainer")
class GeometryLanguageTrainer(BaseTrainer):
    def __init__(
        self,
        cfg: DictConfig,
        dataset_dir: str,
        checkpoint_dir: str,
        log_dir: str,
    ) -> None:
        super().__init__(cfg, dataset_dir, checkpoint_dir, log_dir)

    def init_dataset(self) -> None:
        # Create datasets for training & validation, download if necessary
        dataset_cls = registry.get_dataset(self.cfg.dataset.name)

        print("dataset name:",self.cfg.dataset.name)
        #self.train_dataset = dataset_cls(split="train",
        #                                 root_dir=self.dataset_dir)
        self.dataset = dataset_cls(split="train",
                                         root_dir=self.dataset_dir)
        #Subset random.sample(range(2001), self.cfg.dataset.size)
        subset_indice = random.sample(range(2001), self.cfg.dataset.size)
        #subset_indice = [100,400,600]
        #subset_indice = [70, 120, 220, 320] # for testing prediction
        self.dataset = Subset(self.dataset, subset_indice)
        train_size = int(0.9 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])

        # Initialize data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.cfg.training.dataset.num_workers,
            drop_last=True,
            pin_memory=True,
        )

        self.validation_loader = DataLoader(
            self.val_dataset,
            batch_size=2, # batschsize =2 for validation
            num_workers=self.cfg.training.dataset.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def init_model(self) -> None:
        model_cls = registry.get_affordance_model(self.cfg.model.name)
        self.model = model_cls(
            input_shape=self.input_shape,
            target_input_shape=self.target_input_shape,
        ).to(self.device)
        self.pretrained_state = defaultdict(int)

        logger.info("Is model init distrib: {}".format(self._is_distributed))
        if self._is_distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.device],
                output_device=self.device,
                find_unused_parameters=False,
            )

        if self.cfg.training.pretrained:
            path = self.cfg.training.pretrained_checkpoint
            logger.info(
                "Initializing using pretrained weights from {}".format(path)
            )
            self.pretrained_state = self.load_state(path, ckpt_only=True)

        if self.cfg.training.optimizer == "Adam":
            trainable_params = [
                p for p in self.model.parameters() if p.requires_grad
            ]
            self.optimizer = torch.optim.Adam(
                trainable_params,
                lr=self.cfg.training.lr,
                eps=1e-3
            )
            logger.info(
                "Total trainable parameters: {}/{}".format(
                    sum([p.numel() for p in trainable_params]),
                    sum([p.numel() for p in self.model.parameters()]),
                )
            )
            logger.info("Initializing using Adam optimizer")
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.cfg.training.lr,
                momentum=0.9,
                weight_decay=0.0005,
            )
            logger.info("Initializing using SGD optimizer")
        self.scheduler = StepLR(
            self.optimizer,
            step_size=self.cfg.training.lr_scheduler.step_decay,
            gamma=self.cfg.training.lr_scheduler.gamma,
        )

    def _all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        r"""All reduce helper method that moves things to the correct
        device and only runs if distributed
        """
        if not self._is_distributed:
            return t

        orig_device = t.device
        t = t.to(device=self.device)
        torch.distributed.all_reduce(t)

        return t.to(device=orig_device)

    def save_state(self, epoch):
        state_dict = {
            "epoch": epoch,
            "ckpt_dict": self.model.state_dict(),
            "optim_dict": self.optimizer.state_dict(),
            "lr_scheduler": self.scheduler.state_dict(),
        }
        model_path = os.path.join(
            self.checkpoint_dir, "ckpt_{}.pth".format(epoch)
        )
        torch.save(state_dict, model_path)

    def load_state(self, path, ckpt_only: bool = False):
        state_dict = torch.load(path, map_location="cpu")

        if "optim_dict" in state_dict and not ckpt_only:
            self.optimizer.load_state_dict(state_dict["optim_dict"])

        # To handle case when model is saved before commit 862850571316b82613dad67525f8c1bf643b4f10
        ckpt_dict = (
            state_dict["ckpt_dict"] if "ckpt_dict" in state_dict else state_dict
        )
        if not self._is_distributed:
            missing_keys = self.model.load_state_dict(ckpt_dict)
            #missing_keys = self.model.load_state_dict(
            #    {k.replace("module.", ""): v for k, v in ckpt_dict.items()}
            #)
        else:
            missing_keys = self.model.load_state_dict(ckpt_dict)
        logger.info("Missing keys: {}".format(missing_keys))
        return {
            "epoch": (
                state_dict["epoch"]
                if "epoch" in state_dict and not ckpt_only
                else 0
            ),
        }

    def observations_batch_from_batch(
        self, batch: Dict[str, torch.Tensor], preds: torch.Tensor
    ) -> List[Dict]:
        None

    def metrics(
        self,
        preds: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        mode: str = "val",
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        return 0,0 # fake metrics

    def model_metrics(self):
        None

    def apply_transforms(
        self, batch: Dict[str, torch.Tensor], split: str = "train"
    ) -> Any:
        return batch

    def train_one_epoch(
        self, epoch_index: int
    ) -> Tuple[float, Dict[str, float]]:
        total_loss = 0.0
        running_loss = 0.0
        last_loss = 0.0

        running_metrics = defaultdict(float)
        avg_metrics = defaultdict(float)

        num_batches = len(self.train_loader)

        #if self._is_distributed:
        #    self.train_loader.sampler.set_epoch(epoch_index)

        loss_fn = registry.get_loss_fn(self.cfg.training.loss.name)(
            self.cfg.training.loss[self.cfg.training.loss.name]
        )
        logger.info(
            "[Worker {}] Loss fn: {} - {}, Rank 0 - {}".format(
                self.local_rank,
                self.cfg.training.loss.name,
                loss_fn,
                rank0_only(),
            )
        )
        self.scheduler.step()

        batch_load_time = 0
        batch_aug_time = 0
        batch_proc_time = 0
        batch_update_time = 0
        start_time = time.time()
        for i, batch in enumerate(self.train_loader):
            # logger.info(
            #     "[Process: {}] Step: {} done".format(self.local_rank, i)
            # )
            for key, val in batch.items():
                if type(val) == list:
                    continue
                batch[key] = val.float().to(self.device)
            batch_load_time += time.time() - start_time
            start_time = time.time()
            batch = self.apply_transforms(batch, split="train")
            batch_aug_time += time.time() - start_time

            update_start_time = time.time()

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(batch=batch)["affordance"].squeeze(1)
            #print("outputs shape:", outputs.shape)
            #print("mask shape:", batch["mask"].shape)
            # Compute the loss and its gradients
            loss = loss_fn(outputs, batch["mask"].permute(0,2,1))
            loss.backward()

            metrics, _ = self.metrics(outputs, batch, mode="train")

            if torch.isnan(loss):
                #o_inputs = torch.sigmoid(outputs)
                logger.info(
                    "[Process: {}] Step: {}\t Loss: {}\t Metrics: {}\t Loss pre: {}\t P  inp: {} - {}".format(
                        self.local_rank,
                        i,
                        loss.item(),
                        metrics,
                        loss.item(),
                        #o_inputs.sum((1, 2)),
                        batch["image"].min(),
                        batch["image"].max(),
                    )
                )

                sys.exit(1)

            # Adjust learning weights
            self.optimizer.step()

            batch_update_time += time.time() - update_start_time
            aggregate_start_time = time.time()

            if self._is_distributed:
                loss = (
                    self._all_reduce(loss) / torch.distributed.get_world_size()
                )

                metrics_order = sorted(metrics.keys())
                stats = torch.stack([metrics[k] for k in metrics_order])
                stats = self._all_reduce(stats)

                for k, v in zip(metrics_order, stats):
                    metrics[k] = v / torch.distributed.get_world_size()

#            for k, v in metrics.items():
#                running_metrics[k] += v.cpu().item()
#                avg_metrics[k] += v.cpu().item() / num_batches

            # Gather data and report
            total_loss += loss.item()
            running_loss += loss.item()

            batch_proc_time += time.time() - aggregate_start_time

            if rank0_only() and i % self.log_interval == 0 and i > 0:
                last_loss = running_loss / (self.log_interval + 1)
                tb_x = epoch_index * len(self.train_loader) + i + 1

                self.log(
                    tb_x,
                    {
                        "train_per_batch/loss": last_loss,
                        "train_per_batch/learning_rate": self.scheduler.get_last_lr()[
                            0
                        ],
                    },
                )
                self.log(
                    tb_x,
                    {
                        f"train_per_batch/{k}": v / (self.log_interval + 1)
                        for k, v in running_metrics.items()
                    },
                )

                logger.info(
                    "[Process: {}] Step: {}\t Update time: {}\t Metrics time: {}".format(
                        self.local_rank,
                        i,
                        batch_update_time / (self.log_interval + 1),
                        batch_proc_time / (self.log_interval + 1),
                    )
                )
                logger.info(
                    "[Process: {}] Step: {}\t Load time: {}\t Augment time: {}".format(
                        self.local_rank,
                        i,
                        batch_load_time / (self.log_interval + 1),
                        batch_aug_time / (self.log_interval + 1),
                    )
                )
                logger.info(
                    "[Process: {}] Step: {}\t Loss:: {}\t Metrics: {}".format(
                        self.local_rank,
                        i,
                        running_loss / (self.log_interval + 1),
                        {
                            k: v / (self.log_interval + 1)
                            for k, v in running_metrics.items()
                        },
                    )
                )

                batch_load_time = 0
                batch_proc_time = 0
                batch_update_time = 0
                batch_aug_time = 0

                running_loss = 0.0
                running_metrics = defaultdict(float)

            start_time = time.time()

        avg_loss = total_loss / num_batches
        return avg_loss

    def evaluate(
        self
    ) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        running_loss = 0.0
        num_batches = len(self.validation_loader) 
        loss_fn = registry.get_loss_fn(self.cfg.training.loss.name)(
            self.cfg.training.loss[self.cfg.training.loss.name]
            )
        
        for i, batch in enumerate(self.validation_loader):
            # logger.info(
            #     "[Process: {}] Step: {} done".format(self.local_rank, i)
            # )
            for key, val in batch.items():
                if type(val) == list:
                    continue
                batch[key] = val.float().to(self.device)

            # Make predictions for this batch
            outputs = self.model(batch=batch)["affordance"].squeeze(1)


            eval_loss = loss_fn(outputs, batch["mask"].permute(0,2,1))
            total_loss += eval_loss.item()
            running_loss += eval_loss.item()
        avg_loss = total_loss / num_batches
        return avg_loss

    def train(self):
        EPOCHS = self.cfg.training.epochs
        self.model.train()
        logger.info(
            "[Worker {}] Is distributed: {}".format(
                self.local_rank, self._is_distributed
            )
        )
        
        # Synchronize all processes
        if self._is_distributed:
            torch.distributed.barrier()

        logger.info("[Process: {}] Starting training".format(self.local_rank))
        for epoch in range(self.pretrained_state["epoch"], EPOCHS):
            logger.info(
                "[Process: {}] EPOCH {}:".format(self.local_rank, epoch + 1)
            )

            # Train model
            self.model.train()
            avg_loss = self.train_one_epoch(epoch)
            if epoch % 50 == 0:
                self.save_state(epoch + 1)
            
            if epoch% 5 ==0:
                self.model.generate_heatmap(epoch) # debug
            

            logger.info(
                "[Process: {}] Synchronize training processes".format(
                    self.local_rank
                )
            )
            # Synchronize all processes
            if self._is_distributed:
                torch.distributed.barrier()

            logger.info("[Process: {}] Evaluating...".format(self.local_rank))
            # Evaluate model
            self.model.eval()
            eval_loss = self.evaluate()

                #image_pil, phrase, file_name = self.model.inference_4cls()
                #self.log_img(epoch + 1, image_pil, phrase, file_name)

            if rank0_only():
                #logger.info(
                #    "[Epoch {}] Train loss: {}, metrics: {}".format(
                #        epoch, avg_loss, avg_metrics
                #    )
                #)
                

                self.log(
                    epoch + 1,
                    {
                        "train/loss_per_epoch": avg_loss,
                        "train/learning_rate": self.scheduler.get_last_lr()[0],
                        "train/val_loss_per_epoch": eval_loss,
                    },
                )

                
            # Synchronize all processes
            if self._is_distributed:
                torch.distributed.barrier()

    def eval(self):
        with torch.no_grad():
            for i, batch in enumerate(self.train_loader):
                for key, val in batch.items():
                    if type(val) == list:
                        continue
                    batch[key] = val.float().to(self.device)
                batch = self.apply_transforms(batch, split="train")

                outputs = self.model(batch=batch)["affordance"].squeeze(1)
                


    def visualize(
        self,
        input_img: List[np.ndarray],
        targets: List[np.ndarray],
        preds: List[np.ndarray],
        target_query: List[np.ndarray],
        epoch: int,
        split: str,
    ):
        None


