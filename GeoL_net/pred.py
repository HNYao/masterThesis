import os

import hydra
from omegaconf import DictConfig

from GeoL_net.core.base import BaseTrainer
from GeoL_net.core.registry import registry

from torch.multiprocessing import Pool, Process, set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass



@hydra.main(
    config_path=os.path.join(os.getcwd(), "config/baseline/"),
    config_name="GeoL_net_pred"
)
def main(cfg: DictConfig):
    print("trainer name:", cfg.training.trainer)
    trainer = registry.get_trainer(cfg.training.trainer)
    trainer: BaseTrainer = registry.get_trainer(cfg.training.trainer)(
        cfg=cfg,
        dataset_dir=cfg.dataset.root_dir,
        checkpoint_dir=cfg.checkpoint_dir,
        log_dir=cfg.log_dir,
    )
    if cfg.run_type == "train":
        trainer.train()
    else:
        trainer.eval(cfg.checkpoint_dir)


if __name__ == "__main__":
    main()
