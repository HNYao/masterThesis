import os

import hydra
from omegaconf import DictConfig

#from GeoL_net.core.base import BaseTrainer
#from GeoL_net.core.registry import registry
from thirdpart.seeingunseen.seeing_unseen.core.base import BaseTrainer
from thirdpart.seeingunseen.seeing_unseen.core.registry import registry
import torch.multiprocessing as mp

@hydra.main(
    config_path=os.path.join(os.getcwd(), "config/baseline/"),
    config_name="clip_unet_metrics",
)
def main(cfg: DictConfig):
    trainer: BaseTrainer = registry.get_evaluator(cfg.training.trainer)(
        cfg=cfg,
        dataset_dir=cfg.dataset.root_dir,
        checkpoint_dir=cfg.checkpoint_dir,
        log_dir=cfg.log_dir,
    )

    trainer.eval(cfg.training.pretrained_checkpoint)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()