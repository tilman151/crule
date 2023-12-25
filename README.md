# CRULE - Cross-Domain Remaining Useful Life Evaluation Suite

**This is an anonymized version of this repository and does not support PDF. Please finde the appendix [here](https://filebin.net/tbeqx8kbgfevcnk6/Supplementary_Material.pdf). The project depends on two python packages that may give a hint to the authors' identities. For this reason, we anonymized references to these dependencies, too. You can find anonymized versions of these dependencies [here](https://anonymous.4open.science/r/approaches-752D) and [here](https://anonymous.4open.science/r/datasets-E963).**

[![Master](https://github.com/tilman151/crule/actions/workflows/on_push.yaml/badge.svg)](https://github.com/tilman151/rul-adapt/actions/workflows/on_push.yaml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository accompanies the paper "From Inconsistency to Unity: A Benchmarking
Framework for RUL Domain Adaptation" currently under review.
It contains a benchmark suite for domain adaptation approaches for remaining useful life estimation, including hyperparameter search.

Please refer to [rul-datasets](https://www.github.com/tilman151/rul-adapt) for the included datasets and [rul-adapt](https://www.github.com/tilman151/rul-adapt) for the included approaches.

## Installation

This project is set up with [Poetry](https://python-poetry.org/).
It is the easiest to install Poetry via pipx:

```bash
pipx install poetry
```

To install the dependencies, run:

```bash
poetry install
```

If you are running this command on a server, you may need to deactivate the keyring backend.
To do so, run:

```bash
PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring
```

Poetry will generate a new virtual environment for you to use.
To activate it, run:

```bash
poetry shell
```

or prefix your commands with `poetry run`.

## Hyperparameter Search

To run a hyperparameter search for a specific approach on a GPU, run:

```bash
poetry run python tune_adaption.py --dataset <Dataset> --backbone <Backbone> --approach <Approach> --gpu --sweep_name <Name_for_your_sweep> --entity <WandB_Entity>
```

All results will be logged to WandB in the specified entity and project.
By default, CMAPSS runs with four parallel trials and the remaining datasets with one.
To change this, go to line 53 or 56 respectively and set the value for `"gpu"`.
If you want five parallel trials, set it to 0.2.
How many trials can be run in parallel depends on the GPU memory.

Each trial will be logged, and after all of them are finished, an additional summary run will be created.
This run contains the analysis dataframe of the search.
To get the best hyperparameters, you can run:

```python
from crule.evaluation import get_best_tune_run

best_hparams = get_best_tune_run("<WandB/Summary/Run/Path>")
```

The returned dictionary contains the best hyperparameters.

## Reproduction

To run the experiments from the paper, run:

```bash
chmode +x run_cmapss.sh
poetry run ./run_cmapss.sh
```

for CMAPSS or:

```bash
chmode +x run_bearing.sh
chmode +x ./run_bearing.sh
```

for FEMTO and XJTU-SY. If you want to run a specific experiment, execute `run.py` directly:

```bash
poetry run python train.py --multirun \
       hydra/launcher=ray \  # omit to run without ray
       +hydra.launcher.num_gpus=<GPU per Run> \  # use num_cpus when running on CPU
       +task=<Task Name> \  # e.g., three2one
       +approach=<Approach Name> \  # e.g., dann
       +feature_extractor=<Feature Extractor Name> \  # either cnn or lstm
       +dataset=<Dataset Name> \  # either cmapss, femto or xjtu-sy
       test=True \  # default is False
       logger.entity=<WandB Entity Name> \
       logger.project=<WandB Project Name> \
       +logger.tags=<List of Tags as String> \  # e.g., "[a,b,c]"
       replications=<Number of Replications> \
       accelerator=<Accelerator Name>  # either cpu or gpu, default is gpu
```

To export all runs from WandB to a data frame, run:

```python
from crule.evaluation import load_runs

runs = load_runs("<WandB/Project/Path>", exclude_tags=["pretraining"])
```

## Extension

This project uses [rul-datasets](https://www.github.com/tilman151/rul-adapt) for loading the benchmark datasets and [rul-adapt](https://www.github.com/tilman151/rul-adapt) for adaptation approaches.
If you want to add an approach or dataset, please propose them to these packages.

For configuration, this project uses [Hydra](https://hydra.cc/).
All configuration files are located in `conf`.
To use a different logger, e.g. MlFlow, add a configuration YAML to `conf/logger` and use the `logger=mlflow` override when calling `train.py`:

```yaml
# config/logger/mlflow.yaml

_target_: pytorch_lightning.loggers.MLFlowLogger
experiment_name: my_experiment
run_name: my_run
tracking_uri: file://.mlruns
```

Any logger compatible with PyTorch Lightning may be used.
Similarly, other parts (feature extractor, regressor, etc.) of the suite can be replaced like this.
