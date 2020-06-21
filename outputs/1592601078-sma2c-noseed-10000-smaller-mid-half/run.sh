#!/bin/bash
outname="outputs/$(date +'%s')-sma2c-noseed-10000-smaller-mid-half"
mkdir -p $outname
cp ./run.sh $outname/
python final/runner_2.py \
    --rounds 25 \
    --episodes 10000 \
    --alpha 0.00075 \
    --beta 0.0025 \
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
    "$outname/real_scores.json"

# for i in {1..5}; do sleep 5; ./run.sh & done; wait