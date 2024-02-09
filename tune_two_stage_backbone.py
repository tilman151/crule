import functools
import os
import random
import uuid
from datetime import datetime
from typing import Optional

import hydra.utils
import pytorch_lightning as pl
import ray
import rul_adapt
import wandb
from ray import tune

from crule.run.utils import XjtuSyWindowExtractor

FIXED_HPARAMS = ["_target_", "input_channels", "seq_len"]
BATCH_SIZE = 128

FEMTO_FTTP = {
    1: [407, 544, 521, 840, 2306, 479, 995],
    2: [819, 192, 257, 248, 252, 213, 163],
    3: [132, 116, 306],
}
XJTU_SY_FTTP = {
    1: [28, 32, 61, 52, 36],
    2: [238, 65, 128, 4, 121],
    3: [749, 614, 342, 1418, 74],
}


def _max_lower_layers(config):
    kernel_size = config["model"]["lower_stage"]["kernel_size"]
    seq_len = config["model"]["lower_stage"]["seq_len"]
    dilation = config["model"]["lower_stage"]["dilation"]
    stride = config["model"]["lower_stage"]["stride"]
    cut_off = dilation * (kernel_size - 1)
    max_cnn_layers = -1
    while seq_len > 0:
        seq_len = (seq_len - cut_off - 1) // stride + 1
        max_cnn_layers += 1

    return max_cnn_layers


def _max_upper_layers(config):
    kernel_size = config["model"]["upper_stage"]["kernel_size"]
    seq_len = config["model"]["upper_stage"]["seq_len"]
    cut_off = kernel_size
    max_cnn_layers = -1
    while seq_len > 0:
        seq_len = seq_len - cut_off
        max_cnn_layers += 1

    return max_cnn_layers


COMMON_SEARCH_SPACE = {
    "evaluate_degraded_only": True,
    "lr": tune.qloguniform(1e-5, 1e-2, 1e-5),  # quantized log uniform
    "lower_window_size": 2560,
    "upper_window_size": tune.choice([5, 10, 20]),
}
LOWER_CNN_SEARCH_SPACE = {
    "_target_": "rul_adapt.model.CnnExtractor",
    "input_channels": 2,
    "seq_len": tune.sample_from(lambda config: config["lower_window_size"]),
    "kernel_size": tune.choice([3, 5, 7]),
    "dropout": tune.quniform(0.0, 0.5, 0.1),  # quantized uniform
    "units": tune.sample_from(
        lambda config: [random.choice([16, 32, 64])]
        * random.randint(1, min(10, _max_lower_layers(config)))
    ),
    "fc_units": tune.choice([16, 32, 64, 128]),
    "dilation": 1,
    "stride": 2,
}
UPPER_CNN_SEARCH_SPACE = {
    "_target_": "rul_adapt.model.CnnExtractor",
    "input_channels": tune.sample_from(
        lambda config: config["model"]["lower_stage"]["fc_units"]
    ),
    "kernel_size": 3,
    "seq_len": tune.sample_from(lambda config: config["upper_window_size"]),
    "dropout": tune.sample_from(
        lambda config: config["model"]["lower_stage"]["dropout"]  # same as lower
    ),
    "units": tune.sample_from(
        lambda config: [config["model"]["lower_stage"]["fc_units"]]
        * random.randint(1, min(10, _max_upper_layers(config)))
    ),
    "fc_units": tune.choice([16, 32, 64, 128]),
}


def tune_backbone(
    dataset: str,
    gpu: bool,
    entity: str,
    project: str,
    sweep_name: Optional[str],
    save_dir: Optional[str],
):
    sweep_uuid = (
        str(uuid.uuid4()) if sweep_name is None else f"{sweep_name}-{datetime.now()}"
    )
    save_dir = save_dir or os.path.join("./", sweep_uuid)
    save_dir = f"file://{os.path.abspath(save_dir)}"
    search_space = {
        **COMMON_SEARCH_SPACE,
        "model": {
            "_target_": "rul_adapt.model.TwoStageExtractor",
            "lower_stage": {**LOWER_CNN_SEARCH_SPACE},
            "upper_stage": {**UPPER_CNN_SEARCH_SPACE},
        },
    }

    if dataset == "femto":
        source_config = {
            "_target_": "rul_datasets.RulDataModule",
            "reader": {"_target_": "rul_datasets.FemtoReader", "norm_rul": True},
            "batch_size": BATCH_SIZE,
        }
        fds = list(range(1, 4))
        resources = {"gpu": 0.5}
        fttp = FEMTO_FTTP
    elif dataset == "xjtu-sy":
        source_config = {
            "_target_": "rul_datasets.RulDataModule",
            "reader": {"_target_": "rul_datasets.XjtuSyReader", "norm_rul": True},
            "batch_size": BATCH_SIZE,
        }
        fds = list(range(1, 4))
        resources = {"gpu": 0.5}
        fttp = XJTU_SY_FTTP
    else:
        raise ValueError(f"Unknown dataset {dataset}.")

    metric_columns = ["avg_rmse"] + [f"rmse_{i}" for i in fds]
    scheduler = tune.schedulers.FIFOScheduler()  # runs trials sequentially
    parameter_cols = [k for k in search_space.keys() if k not in FIXED_HPARAMS]
    reporter = tune.CLIReporter(  # prints progress to console
        parameter_columns=parameter_cols,
        metric_columns=metric_columns,
        max_column_length=15,
    )

    # set arguments constant for all trials and run tuning
    tune_func = functools.partial(
        run_training,
        source_config=source_config,
        fds=fds,
        sweep_uuid=sweep_uuid,
        entity=entity,
        project=project,
        gpu=gpu,
        fttp=fttp,
    )
    analysis = tune.run(
        tune_func,
        name=f"tune-{dataset}-two-stage-supervised",
        metric="avg_rmse",  # monitor this metric
        mode="min",  # minimize the metric
        num_samples=100,
        resources_per_trial=resources if gpu else {"cpu": 16},
        scheduler=scheduler,
        config=search_space,
        progress_reporter=reporter,
        fail_fast=True,  # stop on first error
        storage_path=save_dir,
    )

    wandb.init(
        project=project,
        entity=entity,
        job_type="analysis",
        tags=[sweep_uuid],
    )
    analysis_table = wandb.Table(dataframe=analysis.dataframe())
    wandb.log({"tune_analysis": analysis_table})

    print("Best hyperparameters found were: ", analysis.best_config)


def run_training(config, source_config, fds, sweep_uuid, entity, project, gpu, fttp):
    trial_uuid = uuid.uuid4()
    results = []
    for fd in fds:
        source_config["reader"]["fd"] = fd
        if fttp is not None:
            source_config["reader"]["first_time_to_predict"] = fttp[fd]
        dm = hydra.utils.instantiate(
            source_config,
            feature_extractor=XjtuSyWindowExtractor(
                config["upper_window_size"], config["lower_window_size"]
            ),
        )

        backbone = hydra.utils.instantiate(config["model"])
        regressor = rul_adapt.model.FullyConnectedHead(
            config["model"]["upper_stage"]["fc_units"],
            [1],
            act_func_on_last_layer=False,
        )
        approach = rul_adapt.approach.SupervisedApproach(
            loss_type="rmse",
            optim_type="adam",
            lr=config["lr"],
            evaluate_degraded_only=config["evaluate_degraded_only"],
        )
        approach.set_model(backbone, regressor)

        logger = pl.loggers.WandbLogger(
            project=project,
            entity=entity,
            group=str(trial_uuid),
            tags=[sweep_uuid],
        )
        logger.experiment.define_metric("val/loss", summary="best", goal="minimize")
        callbacks = [
            pl.callbacks.EarlyStopping(monitor="val/loss", patience=20),
            pl.callbacks.ModelCheckpoint(
                monitor="val/loss", save_top_k=1, dirpath="./checkpoints"
            ),
        ]
        trainer = pl.Trainer(
            accelerator="gpu" if gpu else "cpu",
            max_epochs=100,
            logger=logger,
            callbacks=callbacks,
            num_sanity_val_steps=0,
        )

        trainer.fit(approach, dm)
        results.append(trainer.checkpoint_callback.best_model_score.item())
        wandb.finish()

    # report average RMSE and RMSE for each FD
    results = {
        "avg_rmse": sum(results) / len(results),
        **{f"rmse_{i}": r for i, r in enumerate(results, start=1)},
    }
    ray.train.report(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="xjtu-sy")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--entity", type=str, default="rul-adapt")
    parser.add_argument("--project", type=str, default="backbone-tuning")
    parser.add_argument("--sweep_name", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    opt = parser.parse_args()

    ray.init(log_to_driver=False)
    tune_backbone(
        opt.dataset, opt.gpu, opt.entity, opt.project, opt.sweep_name, opt.save_dir
    )
