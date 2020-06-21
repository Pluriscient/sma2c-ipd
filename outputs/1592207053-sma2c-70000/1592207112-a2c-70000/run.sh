#!/bin/bash
outname="outputs/$(date +'%s')-a2c-70000"
mkdir -p $outname
cp ./run.sh $outname/
python conceptual_copy/runner_2.py \
    --rounds 25 \
    --episodes 70000 \
    --alpha 0.00003 \
    --beta 0.0001 \
    --seed 123 \
    --encoder-fc 20 \
    --encoder-fc 30 \
    --latent-dims 2 \
    --pure-a2c \
    --a2c-fc 10 \
    --a2c-fc 10 \
    --output $outname \
    'tit for tat' \
    defector

python conceptual_copy/plotter.py \
    --output "$outname/results.png" \
    "$outname/scores.json"

