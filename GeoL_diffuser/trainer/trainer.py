import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple
import numpy as np

import cv2
import random
import torchvision.transforms as T
from PIL import Image

import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset, random_split

from GeoL_net.core.base import BaseTrainer
from GeoL_net.core.logger import logger
from GeoL_net.core.registry import registry
from GeoL_net.utils.ddp_utils import rank0_only
from GeoL_net.models.modules import ProjectColorOntoImage
from GeoL_net.dataset_gen.RGBD2PC import backproject, project_3d
from GeoL_diffuser.algos.pose_algos import PoseDiffusionModel
from clip.model import tokenize



@registry.register_trainer(name="GeoL_diffusion_trainer")
class PoseDiffuserTrainer(BaseTrainer):
    def __init__(self, cfg, dataset_dir, checkpoint_dir, log_dir):
        super().__init__(cfg, dataset_dir, checkpoint_dir, log_dir)

    def init_dataset(self):
        dataset_cls = registry.get_dataset(self.cfg.dataset.name)

        print("dataset name:", self.cfg.dataset.name)
        self.dataset = dataset_cls(
            split="train",
            affordance_threshold=self.cfg.dataset.affordance_threshold,
            gt_pose_samples = self.cfg.dataset.gt_pose_samples,
            root_dir=self.dataset_dir,
        )

        subset_indice = random.sample(
            range(self.cfg.dataset.size * 5), self.cfg.dataset.size
        )
        self.dataset = Subset(self.dataset, subset_indice)
        train_size = int(0.9 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_size, val_size]
        )

        # data agumentation initialization

        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=None,
            num_workers=self.cfg.training.dataset.num_workers,
            drop_last=True,
            pin_memory=True,
        )

        self.validation_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.cfg.training.dataset.num_workers,
            drop_last=False,
            pin_memory=True,
        )

        logger.info(
            "Train dataset size: {}, Validation dataset size: {}".format(
                len(self.train_dataset), len(self.val_dataset)
            )
        )

    def init_model(self):
        # initialize model
        model_cls = PoseDiffusionModel
        self.model = model_cls(self.cfg.model).to(self.device) 
        self.pretrained_state = defaultdict(int)

        # load pretrained model
        if self.cfg.training.pretrained:
            path = self.cfg.training.pretrained_checkpoint
            logger.info(f"Loading pretrained model from {path}")
            self.pretrained_state = self.load_state(path, ckpt_only=True)

        if self.cfg.training.optimizer == "Adam":
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.Adam(
                trainable_params, lr=self.cfg.training.lr, eps=1e-3
            )
            logger.info("Initializing Adam optimizer...")
        else:
            print(" Not using Adam optimizer ...")
        self.scheduler = StepLR(
            self.optimizer,
            step_size=self.cfg.training.lr_scheduler.step_decay,
            gamma=self.cfg.training.lr_scheduler.gamma,
        )

    def save_state(self, epoch):
        state_dict = {
            "epoch": epoch,
            "ckpt_dict": self.model.state_dict(),
            "optim_dict": self.optimizer.state_dict(),
            "lr_scheduler": self.scheduler.state_dict(),
        }
        model_path = os.path.join(self.checkpoint_dir, "ckpt_{}.pth".format(epoch))
        torch.save(state_dict, model_path)

    def load_state(self, path, ckpt_only: bool = False):
        state_dict = torch.load(path, map_location="cpu")

        if "optim_dict" in state_dict and not ckpt_only:
            self.optimizer.load_state_dict(state_dict["optim_dict"])

        # To handle case when model is saved before commit 862850571316b82613dad67525f8c1bf643b4f10
        ckpt_dict = state_dict["ckpt_dict"] if "ckpt_dict" in state_dict else state_dict

        missing_keys = self.model.load_state_dict(ckpt_dict)

        logger.info("Missing keys: {}".format(missing_keys))
        return {
            "epoch": (
                state_dict["epoch"] if "epoch" in state_dict and not ckpt_only else 0
            ),
        }

    def metrics(
        self,
        preds: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        model: str = "val",
    ):
        return 0.0  # TODO: add metrics

    def model_metrics(self):
        None

    def apply_transforms(self, batch, split="train"):
        return batch  # TODO: add transforms

    def train_one_epoch(self, epoch_index):
        total_loss = 0.0
        running_loss = 0.0
        last_loss = 0.0

        running_metrics = defaultdict(float)
        avg_metrics = defaultdict(float)

        num_batches = len(self.train_loader)

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

            update_start_time = time.time()
            self.optimizer.zero_grad()

            batch_size = batch["image"].shape[0]
            #cond_fill_val = -1
            
            #drop_mask_cond_position = torch.rand(len(batch["pc_position"])) < self.cfg.model.training.cond_drop_pc_position_p
            drop_mask_cond_affordance = torch.rand(len(batch["affordance"])) < self.cfg.model.training.cond_drop_affordance_p
            #drop_mask_cond_obj_pc_position = torch.rand(len(batch["object_pc_position"])) < self.cfg.model.training.cond_drop_object_pc_position_p

            #random_mask = torch.rand_like(batch["affordance"])
            #batch["affordance"][drop_mask_cond_affordance] = batch['affordance_for_non_cond'][drop_mask_cond_affordance]
            #batch['pc_position'][drop_mask_cond_position] = cond_fill_val
            #batch["object_pc_position"][drop_mask_cond_obj_pc_position] = cond_fill_val
            batch['gt_pose_xyz'][drop_mask_cond_affordance] = batch['gt_pose_xyz_for_non_cond'][drop_mask_cond_affordance]
            batch['gt_pose_xy'][drop_mask_cond_affordance] = batch['gt_pose_xy_for_non_cond'][drop_mask_cond_affordance] # if drop the affordance, chage the gt
            #batch['gt_pose_xyz'][drop_mask_cond_position] = batch['gt_pose_xyz_for_non_cond'][drop_mask_cond_position] # if drop the affordance, chage the gt
            #batch['gt_pose_xyz'][drop_mask_cond_obj_pc_position] = batch['gt_pose_xyz_for_non_cond'][drop_mask_cond_obj_pc_position] # if drop the affordance, chage the gt
            

            #batch["affordance"][drop_mask_cond_affordance] = cond_fill_val
            #batch["object_pc_position"][drop_mask_cond_obj_pc_position] = cond_fill_val
            
            outputs = self.model(data_batch=batch)["pose_xy_pred"]

            # compute the loss and its gradients
            loss = self.model.get_loss(batch, i)["loss"]
            loss.backward()

            if torch.isnan(loss):
                logger.info(
                    "[Process: {}] Step: {}\t Loss: {}".format(
                        self.local_rank,
                        i,
                        loss.item(),
                    )
                )
                sys.exit(1)

            # Adjust learning weights
            self.optimizer.step()

            batch_update_time += time.time() - update_start_time
            aggregate_start_time = time.time()

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
        self,
    ):
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

            # Make predictions for this batch
            #batch = self.apply_transforms(batch, split="train")
            
            eval_loss = self.model.get_loss(batch, i)["loss"]
            total_loss += eval_loss.item()
            running_loss += eval_loss.item()
        avg_loss = total_loss / num_batches
        return avg_loss

    def train(self):
        EPOCHS = self.cfg.training.epochs
        self.model.train()

        logger.info("[Process: {}] Starting training".format(self.local_rank))
        for epoch in range(self.pretrained_state["epoch"], EPOCHS):
            logger.info("[Process: {}] EPOCH {}:".format(self.local_rank, epoch + 1))

            # Train model
            self.model.train()
            avg_loss, model_pred, last_batch = self.train_one_epoch(epoch)
            if epoch % 30 == 0:
                self.save_state(epoch + 1)

            if epoch % 1 == 0:
                pose_xyz_pred = self.get_xyz_pose_pred(
                    model_pred, last_batch
                )  # [batch_size, 8, 3]
                img_pred_list, img_gt_list = self.visualize_prediction_with_bound(
                    pose_xyz_pred, last_batch
                )
                self.log_img_table(epoch, img_pred_list, img_gt_list)
                self.log_img(epoch, img_pred_list[0], "pred")

                logger.info("Pose_4d_pred: {}".format(pose_xyz_pred[0]))
                logger.info("Ground truth: {}".format(last_batch["gt_pose_xyz"][0][0]))
                logger.info("Min: {}".format(last_batch["gt_pose_xyz_min_bound"][0]))
                logger.info("Max: {}".format(last_batch["gt_pose_xyz_max_bound"][0]))

            logger.info(
                "[Process: {}] Synchronize training processes".format(self.local_rank)
            )
            # Synchronize all processes
            if self._is_distributed:
                torch.distributed.barrier()

            logger.info("[Process: {}] Evaluating...".format(self.local_rank))
            # Evaluate model
            self.model.eval()
            eval_loss = self.evaluate()
    

            if rank0_only():

                self.log(
                    epoch + 1,
                    {
                        "train/loss_per_epoch": avg_loss,
                        "train/learning_rate": self.scheduler.get_last_lr()[0],
                        "train/val_loss_per_epoch": eval_loss,
                    },
                )

    def get_xyz_pose_pred(self, prediction, data_batch):
        """
        Get xyz pose prediction from the model xy prediction

        Args:
            prediction: model prediction, (B,2)
            data_batch: data batch
        """
        batch_size = prediction.shape[0]
        xy_pred = prediction.squeeze(1)
        scene_pc_position = data_batch["pc_position"]
        affordance = data_batch["affordance"]

        # find the nearest point in the scene point cloud
        scene_xy = scene_pc_position[:, :, :2]

        distances = torch.norm(scene_xy[:, None, :] - xy_pred[:, :, None, :], dim=-1)
        min_indices = torch.argmin(distances, dim=-1)

        nearest_points = scene_pc_position[
            torch.arange(batch_size)[:, None], min_indices
        ]

        return nearest_points



    def visualize_prediction_pose(self, pose_4d_pred, batch):
        """
        Visualize the prediction of the model
        pose_4d_pred: 4D pose prediction, [batch_size, 8, 3]
        data_batch: data batch
        """
        intrinsics = np.array(
            [
                [607.09912 / 2, 0.0, 636.85083 / 2],
                [0.0, 607.05212 / 2, 367.35952 / 2],
                [0.0, 0.0, 1.0],
            ]
        )

        # transform
        points = pose_4d_pred[:, :, :3]
        R = batch["T_plane"][:, :3, :3]
        t = batch["T_plane"][:, :3, 3]
        R_inv = R.transpose(1, 2)
        transformed_points = torch.bmm(points, R_inv) - t.unsqueeze(1)
        pose_4d_pred[:, :, :3] = transformed_points

        gt_points = batch["gt_pose_xyz"][:, :, :3]
        gt_transformed_points = torch.bmm(gt_points, R_inv) - t.unsqueeze(1)
        batch["gt_pose_xyz"][:, :, :3] = gt_transformed_points

        pcs = []
        img_pred_list = []
        img_gt_list = []
        img_rgb_list = batch["image"].cpu()
        output_pred_img_list = []
        output_gt_img_list = []

        projector = ProjectColorOntoImage()

        distance_threshold = 10
        for i in range(batch["depth"].shape[0]):
            depth = batch["depth"][i].cpu().numpy()
            points_scene, _ = backproject(
                depth,
                intrinsics,
                np.logical_and(depth > 0, depth > 0),
                NOCS_convention=False,
            )
            points_scene = torch.tensor(points_scene).to(self.device)
            pcs.append(points_scene)

            distances = torch.norm(
                points_scene[None, :, :] - pose_4d_pred[i][:, None, :], dim=-1
            )
            is_near_pose = (distances < distance_threshold).any(dim=0)
            colors = torch.zeros_like(points_scene, dtype=torch.float32)
            colors[is_near_pose] = torch.tensor(
                [1, 0, 0], dtype=torch.float32, device=self.device
            )
            output_pred_img = projector(
                image_grid=img_rgb_list[i],
                query_points=points_scene.cpu(),
                query_colors=colors.cpu(),
                intrinsics=intrinsics,
            )
            output_pred_img_list.append(output_pred_img)

            colors_gt = torch.zeros_like(points_scene, dtype=torch.float32)
            distances_gt = torch.norm(
                points_scene[None, :, :] - batch["gt_pose_xyz"][i][:, :3][:, None, :],
                dim=-1,
            )
            is_near_pose_gt = (distances_gt < distance_threshold).any(dim=0)
            colors_gt[is_near_pose_gt] = torch.tensor(
                [1, 0, 0], dtype=torch.float32, device=self.device
            )
            output_gt_img = projector(
                image_grid=img_rgb_list[i],
                query_points=points_scene.cpu(),
                query_colors=colors_gt.cpu(),
                intrinsics=intrinsics,
            )
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

            image_np = cv2.addWeighted(image_np, 0.3, color_image_np, 0.7, 0.0)
            pil_image = Image.fromarray(np.uint8(image_np))
            img_gt_list.append(pil_image)

        return img_pred_list, img_gt_list
    
    def visualize_prediction_with_bound(self, pred_xyz_pose, batch):
        """
        Visualize the prediction of the model

        Args:
            pred_xyz_pose: xyz pose prediction, [batch_size, 80, 3]
            batch: data batch

        Returns:
            img_pred_list: list of images of the prediction
            img_gt_list: list of images of the ground truth
        """
        img_gt_list = []
        img_pred_list = []
        batch_size = pred_xyz_pose.shape[0]

        intrinsics = np.array(
                [
                    [607.09912 / 2, 0.0, 636.85083 / 2],
                    [0.0, 607.05212 / 2, 367.35952 / 2],
                    [0.0, 0.0, 1.0],
                ]
            )
        ############################################## Visualize the groundtruth ##############################################
        for i in range(batch_size):
            gt_pose_xyz = batch['gt_pose_xyz'].cpu().numpy()[i] 
            gt_pose_xyz_min_bound = batch['gt_pose_xyz_min_bound'].cpu().numpy()[i, :3]
            gt_pose_xyz_max_bound = batch['gt_pose_xyz_max_bound'].cpu().numpy()[i, :3]

            image = batch['image'].cpu().numpy()[i] # [3, H, W]
            image = (np.transpose(image, (1, 2, 0)) * 255).astype(np.uint8)
            depth = batch['depth'].cpu().numpy()[i] # [H, W]
            points_scene, _ = backproject(depth, intrinsics, depth>0) # [num_points, 3]

            # Measure the pairwise distance between the points
            distances = np.sqrt(
                ((points_scene[:, :2][:, None, :] - gt_pose_xyz[:, :2]) ** 2).sum(axis=2)
            )

            # Find the topk scene points that are closest to the anchor points (gt_pose_xyR)
            scenepts_to_anchor_dist = np.min(distances, axis=1)  # [num_points]
            scenepts_to_anchor_id = np.argmin(distances, axis=1)  # [num_points]
            topk_points_id = np.argsort(scenepts_to_anchor_dist, axis=0)[: gt_pose_xyz.shape[0]]
            tokk_points_id_corr_anchor = scenepts_to_anchor_id[topk_points_id]
            points_for_place = points_scene[topk_points_id]
            points_for_place_bound = np.stack([gt_pose_xyz_min_bound, gt_pose_xyz_max_bound], axis=0)
            # points_for_place = gt_pose_xyR.copy()
            # points_for_place[:, 2] = 1.0

            # Visualize the image
            uv_for_place = project_3d(points_for_place, intrinsics)
            uv_for_place_bound = project_3d(points_for_place_bound, intrinsics)


            image_gt = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            for i in range(uv_for_place.shape[0]):
                uv_color = np.random.randint(0, 255, 3).astype(np.uint8)
                cv2.circle(image_gt, 
                        (int(uv_for_place[i][0]), int(uv_for_place[i][1])), 
                        5, 
                        (int(uv_color[0]), int(uv_color[1]), int(uv_color[2])), -1)
            cv2.circle(image_gt, 
                        (int(uv_for_place_bound[0, 0]), int(uv_for_place_bound[0, 1])), 
                        8, 
                        (255, 0, 0), -1)
            cv2.circle(image_gt, 
                        (int(uv_for_place_bound[1, 0]), int(uv_for_place_bound[1, 1])), 
                        8, 
                        (0, 0, 255), -1)
            cv2.rectangle(image_gt, 
                        (int(uv_for_place_bound[0, 0]), int(uv_for_place_bound[0, 1])), 
                        (int(uv_for_place_bound[1, 0]), int(uv_for_place_bound[1, 1])),
                        (0, 255, 0), 2)

            img_gt_list.append(image_gt)
        ############################################## Visualize the prediction ##############################################
        for i in range(batch_size):
            pred_pose_xyz = pred_xyz_pose.cpu().numpy()[i] 
            gt_pose_xyz_min_bound = batch['gt_pose_xyz_min_bound'].cpu().numpy()[i, :3]
            gt_pose_xyz_max_bound = batch['gt_pose_xyz_max_bound'].cpu().numpy()[i, :3]

            image = batch['image'].cpu().numpy()[i] # [3, H, W]
            image = (np.transpose(image, (1, 2, 0)) * 255).astype(np.uint8)
            depth = batch['depth'].cpu().numpy()[i] # [H, W]
            points_scene, _ = backproject(depth, intrinsics, depth>0) # [num_points, 3]

            # Measure the pairwise distance between the points
            distances = np.sqrt(
                ((points_scene[:, :2][:, None, :] - pred_pose_xyz[:, :2]) ** 2).sum(axis=2)
            )

            # Find the topk scene points that are closest to the anchor points (gt_pose_xyR)
            scenepts_to_anchor_dist = np.min(distances, axis=1)  # [num_points]
            scenepts_to_anchor_id = np.argmin(distances, axis=1)  # [num_points]
            topk_points_id = np.argsort(scenepts_to_anchor_dist, axis=0)[: pred_pose_xyz.shape[0]]
            tokk_points_id_corr_anchor = scenepts_to_anchor_id[topk_points_id]
            points_for_place = points_scene[topk_points_id]
            points_for_place_bound = np.stack([gt_pose_xyz_min_bound, gt_pose_xyz_max_bound], axis=0)
            # points_for_place = gt_pose_xyR.copy()
            # points_for_place[:, 2] = 1.0

            # Visualize the image
            uv_for_place = project_3d(points_for_place, intrinsics)
            uv_for_place_bound = project_3d(points_for_place_bound, intrinsics)


            image_pred = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            for i in range(uv_for_place.shape[0]):
                uv_color = np.random.randint(0, 255, 3).astype(np.uint8)
                cv2.circle(image_pred, 
                        (int(uv_for_place[i][0]), int(uv_for_place[i][1])), 
                        5, 
                        (int(uv_color[0]), int(uv_color[1]), int(uv_color[2])), -1)
            cv2.circle(image_pred, 
                        (int(uv_for_place_bound[0, 0]), int(uv_for_place_bound[0, 1])), 
                        8, 
                        (255, 0, 0), -1)
            cv2.circle(image_pred, 
                        (int(uv_for_place_bound[1, 0]), int(uv_for_place_bound[1, 1])), 
                        8, 
                        (0, 0, 255), -1)
            cv2.rectangle(image_pred, 
                        (int(uv_for_place_bound[0, 0]), int(uv_for_place_bound[0, 1])), 
                        (int(uv_for_place_bound[1, 0]), int(uv_for_place_bound[1, 1])),
                        (0, 255, 0), 2)

            img_pred_list.append(image_pred)
        return img_pred_list, img_gt_list


if __name__ == "__main__":
    batch_size = 2
    num_point = 2048
    xy_pred = torch.rand(2, 8, 2)
    scene_pc_position = torch.rand(2, 2048, 3)
    scene_xy = scene_pc_position[:, :, :2]
    distances = torch.norm(scene_xy[:, None, :] - xy_pred[:, :, None, :], dim=-1)
    min_indices = torch.argmin(distances, dim=-1)
    nearest_points = scene_pc_position[torch.arange(batch_size)[:, None], min_indices]
    print(xy_pred)
    print(nearest_points)
