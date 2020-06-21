#!/bin/bash
outname="outputs/$(date +'%s')"
mkdir -p $outname
cp ./run.sh $outname/
cp ./run.sh $outname/
python conceptual_copy/runner_2.py \
    --rounds 25 \
    --episodes 4000 \
    --alpha 0.00001 \
    --beta 0.00003 \
    --seed 123 \
    --encoder-fc 20 \
    --encoder-fc 30 \
    --latent-dims 2 \
    --a2c-fc 20 \
    --pure-a2c \
    --a2c-fc 80 \
    --output $outname \
    'tit for tat' \
    defector

python conceptual_copy/plotter.py \
    --output "$outname/results.png" \
    "$outname/scores.json"

