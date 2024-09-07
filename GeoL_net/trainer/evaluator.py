import glob
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from GeoL_net.core.logger import logger
from GeoL_net.core.registry import registry
from GeoL_net.dataset.dataset import collate_fn
from GeoL_net.dataset.transform_utils import TTAWrapper
from GeoL_net.utils.ddp_utils import rank0_only
from GeoL_net.utils.utils import write_json
from GeoL_net.trainer.trainer import GeometryLanguageTrainer, BaseTrainer

@registry.register_trainer(name="GeoL_evaluator")
class GeometryLanguageEvaluator(BaseTrainer):
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
        self.train_dataset = dataset_cls(split="train",
                                         root_dir=self.dataset_dir)
        
        #Subset
        subset_indice = list(range(self.cfg.dataset.size))
        self.train_dataset = Subset(self.train_dataset, subset_indice)

        # Initialize data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
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

    def load_state(self, path, ckpt_only: bool = False):
        state_dict = torch.load(path, map_location="cpu")

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

            # Make predictions for this batch
            outputs = self.model(batch=batch)["affordance"].squeeze(1)
            #print("outputs shape:", outputs.shape)
            #print("mask shape:", batch["mask"].shape)
            # Compute the loss and its gradients
            loss = loss_fn(outputs, batch["mask"].permute(0,2,1))
            loss.backward()

            metrics, _ = self.metrics(outputs, batch, mode="train")

            if torch.isnan(loss):
                o_inputs = torch.sigmoid(outputs)
                logger.info(
                    "[Process: {}] Step: {}\t Loss: {}\t Metrics: {}\t Loss pre: {}\t P Mask: {} inp: {} - {}".format(
                        self.local_rank,
                        i,
                        loss.item(),
                        metrics,
                        loss.item(),
                        o_inputs.sum((1, 2)),
                        batch["image"].min(),
                        batch["image"].max(),
                    )
                )

                sys.exit(1)



            batch_update_time += time.time() - update_start_time
            aggregate_start_time = time.time()


            start_time = time.time()

        avg_loss = total_loss / num_batches
        return avg_loss


    def train(self):
        EPOCHS = self.cfg.training.epochs

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
            if epoch % 1 == 0:
                self.model.inference_4cls(epoch) # debug
                self.model.inference_heatmap_4cls(epoch)


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
