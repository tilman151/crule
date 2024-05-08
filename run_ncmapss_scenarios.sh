ENTITY=tkrokotsch
PROJECT=data-scenarios
APPROACHES=("conditional_dann" "latent_align" "pseudo_labels")
PERCENT_BROKEN=("1.0" "0.8" "0.6" "0.4" "0.2")
PERCENT_FAILED=("1.0" "0.8" "0.6" "0.4" "0.2")
REPLICATIONS=5

for APPROACH in "${APPROACHES[@]}"; do
for BROKEN in "${PERCENT_BROKEN[@]}"; do
for FAILED in "${PERCENT_FAILED[@]}"; do
poetry run python train.py \
        --multirun hydra/launcher=ray \
        +hydra.launcher.num_gpus=0.25 \
        +task="glob(*)" \
        +approach="$APPROACH" \
        +feature_extractor=cnn \
        +dataset=ncmapss \
        ++target.reader.percent_broken="$BROKEN" \
        ++target.reader.percent_fail_runs="$FAILED" \
        test=True \
        logger.entity=$ENTITY \
        logger.project=$PROJECT \
       +logger.tags="[ncmapss,broken-$BROKEN,failed-$FAILED]" \
        replications=$REPLICATIONS
done
done
done

poetry run python train.py \
        --multirun hydra/launcher=ray \
        +hydra.launcher.num_gpus=0.2 \
        +task="glob(*)" \
        +approach="no_adaption" \
        +feature_extractor=cnn \
        +dataset=ncmapss \
        test=True \
        logger.entity=$ENTITY \
        logger.project=$PROJECT \
       +logger.tags="[ncmapss,baseline]" \
        replications=$REPLICATIONS