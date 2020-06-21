#!/bin/bash
outname="outputs/$(date +'%s')-sma2c-4000-smaller"
mkdir -p $outname
cp ./run.sh $outname/
python final/runner_2.py \
    --rounds 25 \
    --episodes 4000 \
    --alpha 0.00003 \
    --beta 0.0001 \
    --seed 1232 \
    --encoder-fc 10 \
    --encoder-fc 5 \
    --latent-dims 2 \
    --a2c-fc 5 \
    --a2c-fc 3 \
    --output $outname \
    'tit for tat' \
    defector

python final/plotter.py \
    --output "$outname/results.png" \
    "$outname/scores.json"

