import sys
import os
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import options
from GeoL_diffuser.algos.pose_algos import PoseDiffusionModel
from GeoL_diffuser.dataset.dataset import PoseDataset, PoseDataset_overfit, PoseDataset_overfit_pl
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

def find_lastest_checkpoint(cfg):
    ckpt_dir = os.path.join(cfg.log_dir, cfg.name)
    ckpt_files = os.listdir(ckpt_dir)
    ckpt_files = [f for f in ckpt_files if f.endswith(".ckpt")]
    if "last.ckpt" in ckpt_files:
        ckpt_file_path = os.path.join(ckpt_dir, "last.ckpt")
    else:
        ckpt_files = sorted(ckpt_files)
        ckpt_it2files = {int(f.split("iter")[-1].split(".")[0]): f for f in ckpt_files}
        latest_iter = max(ckpt_it2files.keys())
        ckpt_file_path = os.path.join(ckpt_dir, ckpt_it2files[latest_iter])
    return ckpt_file_path

def main(cfg):
    pl.seed_everything(cfg.seed)

    # Dataset
    datamodule = PoseDataset_overfit_pl(cfg.DATA, cfg.TRAIN)
    datamodule.setup()

    # Model
    model = PoseDiffusionModel(cfg.ALGORITHM)

    # Checkpointer
    train_callbacks = []
    ckpt_dir = os.path.join(cfg.log_dir, cfg.name)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_fixed_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="iter{step}",
        save_top_k=-1,
        monitor=None,
        every_n_train_steps=cfg.TRAIN.save.every_n_steps,
        verbose=True,
        save_last=True,
        auto_insert_metric_name=False,
    )
    train_callbacks.append(ckpt_fixed_callback)

    # Logger
    wandb.login()
    logger = WandbLogger(
        name=cfg.name,
        project=cfg.TRAIN.logging.wandb_project_name,
    )
    logger.experiment.config.update(cfg)
    logger.watch(model=model)

    # Resume
    if cfg.resume:
        ckpt_path = find_lastest_checkpoint(cfg)
        if ckpt_path is None:
            print("No checkpoint found for resume")
            return
        print(f"... Resuming from checkpoint: {ckpt_path}")
    else:
        ckpt_path = None
    
    # Trainer
    trainer = pl.Trainer(
        default_root_dir=cfg.log_dir,
        # checkpointing
        enable_checkpointing=cfg.TRAIN.save.enabled,
        # logging
        logger=logger,
        log_every_n_steps=cfg.TRAIN.logging.every_n_steps,
        # training
        max_epochs=-1,
        max_steps=cfg.TRAIN.training.num_steps,
        # validation
        val_check_interval= None,
        limit_val_batches=cfg.TRAIN.validation.num_val_batches,  # First N batches from a dataset
        # all callbacks
        check_val_every_n_epoch=cfg.TRAIN.validation.every_n_epochs,
        callbacks=train_callbacks,
    )
    
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    print(sys.argv[1:])
    opt_cmd = options.parse_arguments(sys.argv[1:])
    print(opt_cmd)
    opt = options.set(opt_cmd=opt_cmd)
    options.print_options(opt)
    options.save_options_file(opt)

    main(opt)