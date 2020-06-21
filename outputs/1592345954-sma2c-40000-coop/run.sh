#!/bin/bash
outname="outputs/$(date +'%s')-sma2c-40000-coop"
mkdir -p $outname
cp ./run.sh $outname/
python final/runner_2.py \
    --rounds 25 \
    --episodes 40000 \
    --alpha 0.00003 \
    --beta 0.0001 \
    --seed 1232 \
    --encoder-fc 20 \
    --encoder-fc 30 \
    --latent-dims 2 \
    --a2c-fc 10 \
    --a2c-fc 10 \
    --output $outname \
    'tit for tat' \
    'cooperator' \
    defector

python final/plotter.py \
    --output "$outname/results.png" \
    "$outname/scores.json"

