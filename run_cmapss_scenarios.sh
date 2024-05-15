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
        +task="one2two,one2three,one2four,two2one,two2three,two2four,three2one,three2two,three2four,four2one,four2two,four2three" \
        +approach="$APPROACH" \
        +feature_extractor=cnn \
        +dataset=cmapss \
        ++target.reader.percent_broken="$BROKEN" \
        ++target.reader.percent_fail_runs="$FAILED" \
        test=True \
        logger.entity=$ENTITY \
        logger.project=$PROJECT \
       +logger.tags="[cmapss,broken-$BROKEN,failed-$FAILED]" \
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
        +dataset=cmapss \
        test=True \
        logger.entity=$ENTITY \
        logger.project=$PROJECT \
       +logger.tags="[cmapss,baseline]" \
        replications=$REPLICATIONS