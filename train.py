import uuid

import hydra
from omegaconf import DictConfig, OmegaConf

from hydra_plugins.ray_hydra_launcher.ray_launcher import register_ray_launcher
from rul_adapt import utils


register_ray_launcher()


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(config: DictConfig):
    replications = config["replications"]
    if replications > 1:
        config["logger"]["group"] = str(uuid.uuid4())
    runner = utils.str2callable(config["runner"], restriction="crule.run")
    run_config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    for _ in range(replications):
        runner(run_config)


if __name__ == "__main__":
    main()
