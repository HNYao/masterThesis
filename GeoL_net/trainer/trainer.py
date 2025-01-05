import glob
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple
from matplotlib import cm

import open3d as o3d
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
from GeoL_net.models.modules import pcdheatmap2img
from GeoL_net.core.base import BaseTrainer
from GeoL_net.core.logger import logger
from GeoL_net.core.registry import registry
from GeoL_net.utils.ddp_utils import rank0_only
from GeoL_net.models.modules import ProjectColorOntoImage_v3, ProjectColorOntoImage
import torchvision.transforms as T
from PIL import Image
from GeoL_net.dataset_gen.RGBD2PC import backproject, visualize_points
from scipy.spatial.distance import cdist


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
        self.dataset = dataset_cls(split="train",
                                         root_dir=self.dataset_dir)
        #Subset random.sample(range(2001), self.cfg.dataset.size)
        subset_indice = random.sample(range(self.cfg.dataset.size+1), self.cfg.dataset.size)
        self.dataset = Subset(self.dataset, subset_indice)
        train_size = int(0.9 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])

        # data agumentation initialization
        transform_args = self.cfg.dataset.transform_args
        original_size = self.input_shape
        if (
            transform_args["random_resize_crop_prob"] > 0
            or transform_args["resize_prob"] > 0
        ):
            self.input_shape = [
                int(i) for i in transform_args["resized_resolution"]
            ]
        self.train_transforms = registry.get_transforms(
            self.cfg.dataset.train_transforms
        )(**transform_args, **{"original_size": original_size})

        logger.info(
            "Train transfoms: {} ".format(
                self.cfg.dataset.train_transforms
            )
        )

        if self._is_distributed:
            self.train_sampler = DistributedSampler(self.train_dataset)

        # Initialize data loaders
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            sampler=self.train_sampler if self._is_distributed else None,
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

        # Report split sizes
        logger.info(
            "Training set has {} instances".format(len(self.train_dataset))
        )
        logger.info(
            "Validation set has {} instances".format(len(self.val_dataset))
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
        # randomly apply transforms, p=0.5
        if random.random() > 0.5:       
            transforms = self.train_transforms
            batch['image'] = transforms(batch['image']) # only agumente the image
        
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

        if self._is_distributed:
            self.train_loader.sampler.set_epoch(epoch_index)

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
            outputs = self.model(batch=batch)["affordance"]

            # Compute the loss and its gradients
            loss = loss_fn(outputs, batch["mask"])
            loss.backward()

            metrics, _ = self.metrics(outputs, batch, mode="train")

            if torch.isnan(loss):
                logger.info(
                    "[Process: {}] Step: {}\t Loss: {}\t Metrics: {}\t Loss pre: {}\t P  inp: {} - {}".format(
                        self.local_rank,
                        i,
                        loss.item(),
                        metrics,
                        loss.item(),
                        batch["image"].min(),
                        batch["image"].max(),
                    )
                )

                sys.exit(1)
            logger.info(f"batch min sigmoid: {outputs.sigmoid().min()} -- max sigmoid: {outputs.sigmoid().max()}")

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
                logger.info(
                    "[Process: {}] Step: {}\t Max:: {}\t Min: {}".format(
                        self.local_rank,
                        i,
                        torch.max(outputs),
                        torch.min(outputs),
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
        return avg_loss, outputs, batch

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
            # logger.info(v2_1
            #     "[Process: {}] Step: {} done".format(self.local_rank, i)
            # )
            for key, val in batch.items():
                if type(val) == list:
                    continue
                batch[key] = val.float().to(self.device)
                logger.info("validation batch key: {}".format(batch['file_path']))
            

            # Make predictions for this batch
            #batch = self.apply_transforms(batch, split="train")
            outputs = self.model(batch=batch)["affordance"]


            eval_loss = loss_fn(outputs, batch["mask"])
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
            avg_loss, model_pred, last_batch  = self.train_one_epoch(epoch)
            if epoch % 200 == 0:
                self.save_state(epoch + 1)
            
            if epoch% 1 ==0:
                #self.model.generate_heatmap(epoch)
                #img_rgb_list, file_name_list, phrase_list = self.model.pcdheatmap2img()
                #img_rgb_list, img_gt_list, file_name_list, phrase_list = self.generate_heatmap(last_batch, model_pred)
                
                img_rgb_list, img_gt_list, file_name_list, phrase_list = self.generate_heatmap_target_point(last_batch, model_pred)
                self.log_img_table(epoch, img_rgb_list, img_gt_list, phrase_list, file_name_list)

                for i in range(len(img_rgb_list)):
                    self.log_img(epoch, img_rgb_list[i], cls="Prediction", phrase = phrase_list[i])
                    self.log_img(epoch, img_gt_list[i], cls="Ground Truth", phrase = phrase_list[i])
                    

                #self.model.generate_heatmap(epoch) # debug
            

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
        
    ):
        pass
    
    def generate_heatmap(self, batch, model_pred):
        """
        Generate heatmap for the model prediction and groud truth mask

        Parameters:
        batch: dict
            batch of data
        model_pred: torch.tensor (default: None) [b, num_points, 1]
            model prediction
        
        Returns:
        img_pred_list: list
            list of PIL images of the model prediction
        img_gt_list: list
            list of PIL images of the ground truth mask
        file_path: list
            list of file path
        phrase: list
            list of phrase
        """
    # camera intrinsc matrix kinect    
        intrinsics = np.array([[607.09912/2 ,   0.     , 636.85083/2  ],
                [  0.     , 607.05212/2, 367.35952/2],
                [  0.     ,   0.     ,   1.     ]])

        img_pred_list = []
        img_gt_list = []
        img_rgb_list = batch["image"].cpu() # the image of the scene [b,c, h, w]
  
        normalized_gt_feat = batch['mask']
 
        feat = model_pred.sigmoid()
        min_feat = feat.min(dim=1, keepdim=True)[0]
        max_feat = feat.max(dim=1, keepdim=True)[0]
        normalized_pred_feat = (feat - min_feat) / (max_feat - min_feat + 1e-6)
        
        pcs = batch['fps_points_scene'].cpu()
        
        turbo_colormap = cm.get_cmap('turbo', 256)
        normialized_gt_feat_np = normalized_gt_feat.cpu().detach().numpy()
        normalized_pred_feat_np = normalized_pred_feat.cpu().detach().numpy()
        color_gt_maps = turbo_colormap(normialized_gt_feat_np)[:, :, :, :3]
        color_gt_maps = torch.from_numpy(color_gt_maps).squeeze(2).cpu()
        color_pred_maps = turbo_colormap(normalized_pred_feat_np)[:, :, :, :3] # [b, num_points, 3] ignore alpha
        color_pred_maps = torch.from_numpy(color_pred_maps).squeeze(2).cpu()
        # Project color onto image
        projector = ProjectColorOntoImage()
        output_pred_img = projector(image_grid = img_rgb_list,
                               query_points = pcs,
                               query_colors = color_pred_maps,
                               intrinsics = intrinsics)
        
        output_gt_img = projector(image_grid = img_rgb_list,
                                 query_points = pcs,
                                 query_colors = color_gt_maps,
                                 intrinsics = intrinsics)
        
        # merge the image and heatmap of prediction
        for i, pred_img in enumerate(output_pred_img):
            color_image = T.ToPILImage()(img_rgb_list[i].cpu())
            pil_img = T.ToPILImage()(pred_img.cpu())

            image_np = np.clip(pil_img, 0, 255)

            color_image_np = np.floor(color_image)
            color_image_np = np.clip(color_image_np, 0, 255)
            color_image_np = np.uint8(color_image_np)

            image_np = cv2.addWeighted(image_np, 0.3, color_image_np, 0.7, 0.0)
            pil_image = Image.fromarray(np.uint8(image_np))
            img_pred_list.append(pil_image)

        # merge the image and heatmap of ground truth
        for i, gt_img in enumerate(output_gt_img):
            color_image = T.ToPILImage()(img_rgb_list[i].cpu())
            pil_img = T.ToPILImage()(gt_img.cpu())

            image_np = np.clip(pil_img, 0, 255)

            color_image_np = np.floor(color_image)
            color_image_np = np.clip(color_image_np, 0, 255)
            color_image_np = np.uint8(color_image_np)

            image_np = cv2.addWeighted(image_np, 0.3, color_image_np, 0.7, 0.0)
            pil_image = Image.fromarray(np.uint8(image_np))
            img_gt_list.append(pil_image)
   
        return img_pred_list, img_gt_list, batch['file_path'], batch['phrase']
    
    def generate_heatmap_target_point(self, batch, model_pred):
        """
        Generate heatmap for the model prediction and groud truth mask

        Parameters:
        batch: dict
            batch of data
        model_pred: torch.tensor (default: None) [b, num_points, 1]
            model prediction
        
        Returns:
        img_pred_list: list
            list of PIL images of the model prediction
        img_gt_list: list
            list of PIL images of the ground truth mask
        file_path: list
            list of file path
        phrase: list
            list of phrase
        """
        # camera intrinsc matrix kinect
        intrinsics =  np.array([[607.09912/2 ,   0.     , 636.85083/2  ],
                [  0.     , 607.05212/2, 367.35952/2],
                [  0.     ,   0.     ,   1.     ]])

        img_pred_list = []
        img_gt_list = []
        img_rgb_list = batch["image"].cpu() # the image of the scene [b,c, h, w]

        normalized_gt_feat = batch['mask']
        feat = model_pred.sigmoid()
        min_feat = feat.min(dim=1, keepdim=True)[0]
        max_feat = feat.max(dim=1, keepdim=True)[0]
        normalized_pred_feat = (feat - min_feat) / (max_feat - min_feat + 1e-6)

        turbo_colormap = cm.get_cmap('turbo', 256) # get the color map for the prediction and ground truth

        # normalize the prediction and ground truth
        normialized_gt_feat_np = normalized_gt_feat.cpu().detach().numpy()
        normalized_pred_feat_np = normalized_pred_feat.cpu().detach().numpy()

        # get the color map for the prediction and ground truth [b, num_points, 3]
        color_gt_maps = turbo_colormap(normialized_gt_feat_np)[:, :, :, :3]
        color_gt_maps = torch.from_numpy(color_gt_maps).squeeze(2).cpu()
        color_pred_maps = turbo_colormap(normalized_pred_feat_np)[:, :, :, :3] # [b, num_points, 3] ignore alpha
        color_pred_maps = torch.from_numpy(color_pred_maps).squeeze(2).cpu()
        
        projector = ProjectColorOntoImage()


        pcs = []
        color_gt_list = []
        color_pred_list = []
        for i in range(batch['fps_points_scene'].shape[0]):
            depth = batch['depth'][i].cpu().numpy()
            fps_points_scene = batch['fps_points_scene'][i].cpu().numpy()
            fps_colors = batch['fps_colors_scene'][i].cpu().numpy()
            points_scene, _ = backproject(depth, intrinsics, np.logical_and(depth > 0, depth > 0), NOCS_convention=False)
            pcs.append(points_scene)

            distance_pred= cdist(points_scene, fps_points_scene)
            nearest_pred_idx = np.argmin(distance_pred, axis=1)
            color_pred_map = color_pred_maps[i]
            color_pred_scene = color_pred_map[nearest_pred_idx]
            color_pred_list.append(color_pred_scene)
            
            color_gt_map = color_gt_maps[i]
            color_gt_scene = color_gt_map[nearest_pred_idx]
            color_gt_list.append(color_gt_scene)


        #pcs = torch.tensor(pcs, dtype=torch.float32) # list to tensor
        output_pred_img_list = []
        output_gt_img_list = []
        for i in range(len(pcs)):
            output_pred_img = projector(image_grid = img_rgb_list[i],
                                query_points = torch.tensor(pcs[i]),
                                query_colors = color_pred_list[i],
                                intrinsics = intrinsics)
            output_pred_img_list.append(output_pred_img)
            output_gt_img = projector(image_grid = img_rgb_list[i],
                                    query_points = torch.tensor(pcs[i]),
                                    query_colors = color_gt_list[i],
                                    intrinsics = intrinsics)
            output_gt_img_list.append(output_gt_img)
        # merge the image and heatmap of prediction
        for i, pred_img in enumerate(output_pred_img_list):
            color_image = T.ToPILImage()(img_rgb_list[i].cpu())
            pil_img = T.ToPILImage()(pred_img.squeeze(0).cpu())

            image_np = np.clip(pil_img, 0, 255)

            color_image_np = np.floor(color_image)
            color_image_np = np.clip(color_image_np, 0, 255)
            color_image_np = np.uint8(color_image_np)

            image_np = cv2.addWeighted(image_np, 0.3, color_image_np, 0.7, 0.0)
            pil_image = Image.fromarray(np.uint8(image_np))
            img_pred_list.append(pil_image)

        # merge the image and heatmap of ground truth
        for i, gt_img in enumerate(output_gt_img_list):
            color_image = T.ToPILImage()(img_rgb_list[i].cpu())
            pil_img = T.ToPILImage()(gt_img.squeeze(0).cpu())

            image_np = np.clip(pil_img, 0, 255)

            color_image_np = np.floor(color_image)
            color_image_np = np.clip(color_image_np, 0, 255)
            color_image_np = np.uint8(color_image_np)

            image_np = cv2.addWeighted(image_np, 0.3, color_image_np,0.7, 0.0)
            pil_image = Image.fromarray(np.uint8(image_np))
            img_gt_list.append(pil_image)
   
        return img_pred_list, img_gt_list, batch['file_path'], batch['phrase']

            



            







