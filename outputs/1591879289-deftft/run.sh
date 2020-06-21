#!/bin/bash
outname="outputs/$(date +'%s')-deftft"
# run the learner
python conceptual_copy/runner_2.py \
    --rounds 20 \
    --episodes 2000 \
    --alpha 0.0001 \
    --beta 0.0003 \
    --seed 10 \
    --encoder-fc 30 \
    --latent-dims 2 \
    --a2c-fc 5 \
    --output $outname \
    'tit for tat' \
    defector
# plot the results
python conceptual_copy/plotter.py \
    --output "$outname/results.png" \
    "$outname/scores.json"
# copy the runner so we know the config
cp ./run.sh $outname/