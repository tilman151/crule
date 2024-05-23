from typing import Any, Dict

import torch
from pytorch_lightning import Trainer
from torch.utils.data import ConcatDataset, DataLoader

import rul_adapt
from rul_adapt.approach.pseudo_labels import get_max_rul
from crule.run import common


def pseudo_labels(config: Dict[str, Any]):
    dm = common.get_adaption_datamodule(config)
    best_pretrained = common.run_pretraining(config, dm)
    approach = common.get_approach(config)
    approach.set_model(best_pretrained.feature_extractor, best_pretrained.regressor)

    trainer = common.get_trainer(config)
    if common.is_wandb_logger(trainer.logger):
        trainer.logger.experiment.define_metric(
            "val/loss", summary="best", goal="minimize"
        )
        trainer.logger.experiment.config["approach"] = "PseudoLabelsApproach"
        trainer.logger.log_hyperparams(dm.hparams)  # log manually as dm isn't used

    best_val_score = float("inf")
    while True:
        combined_dl = _get_adaption_dataloader(approach, dm)
        trainer.fit(
            approach,
            train_dataloaders=combined_dl,
            val_dataloaders=dm.target.val_dataloader(),
        )
        val_score = trainer.checkpoint_callback.best_model_score
        if val_score >= best_val_score or abs(val_score - best_val_score) < 0.1:
            break
        best_val_score = val_score
        best_checkpoint = trainer.checkpoint_callback.best_model_path
        approach = type(approach).load_from_checkpoint(best_checkpoint)
        dm = common.get_adaption_datamodule(config)
        _reset_trainer(trainer)

    result = common.get_result(config, trainer, dm)
    if common.is_wandb_logger(trainer.logger):
        trainer.logger.experiment.finish()

    return result


def _reset_trainer(trainer: Trainer):
    trainer.should_stop = False
    trainer.early_stopping_callback.best_score = torch.tensor(float("inf"))
    trainer.early_stopping_callback.wait_count = 0


def _get_adaption_dataloader(approach, dm):
    approach.cpu()
    if not hasattr(dm.source, "_data"):  # check if setup was already performed
        dm.setup()
    pseudo_rul = rul_adapt.approach.generate_pseudo_labels(
        dm.target, approach, dm.inductive
    )
    max_rul = get_max_rul(dm.target.reader)
    pseudo_rul = [min(max_rul, max(0.0, pr)) for pr in pseudo_rul]
    rul_adapt.approach.patch_pseudo_labels(dm.target, pseudo_rul, dm.inductive)

    source_data = dm.source.to_dataset("dev")
    target_data = dm.target.to_dataset("test" if dm.inductive else "dev", alias="dev")
    combined_data = ConcatDataset([source_data, target_data])
    combined_dl = DataLoader(combined_data, dm.source.batch_size, shuffle=True)

    return combined_dl
