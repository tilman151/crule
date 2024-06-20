import uuid
from typing import Dict, Any

import hydra
from omegaconf import DictConfig, OmegaConf

from crule.run import common


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(config: DictConfig):
    replications = config["replications"]
    if replications > 1:
        config["logger"]["group"] = str(uuid.uuid4())
    run_config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    for _ in range(replications):
        no_adaption(run_config)


def no_adaption(config: Dict[str, Any]):
    dm = common.get_adaption_datamodule(config)
    approach = common.get_approach(config)
    approach.set_model(*common.get_models(config))
    trainer = common.get_trainer(config)
    if common.is_wandb_logger(trainer.logger):
        trainer.logger.experiment.define_metric(
            "val/loss", summary="best", goal="minimize"
        )
        trainer.logger.experiment.config["approach"] = "NoAdaptionApproach"
        trainer.logger.log_hyperparams(
            dm.source.hparams
        )  # log manually as dm isn't used
    dm.prepare_data()  # needs to be called because only dataloaders are used
    dm.setup()
    trainer.fit(
        approach,
        train_dataloaders=dm.source.train_dataloader(),
        val_dataloaders=dm.source.val_dataloader(),
    )
    common.get_result(config, trainer, dm)
    if common.is_wandb_logger(trainer.logger):
        trainer.logger.experiment.finish()


if __name__ == "__main__":
    main()
