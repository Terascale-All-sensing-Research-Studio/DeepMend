# !/bin/bash
# Perform inference using deepmend
#

EXPERIMENT="$1"     # Path to an experiment to evaluate
CHK="2000"
UNIFORM_RATIO="0.2"
NUMITS="1600"
LREG="0.0001"
LR="0.005"
LAMBDANER="0.00001"
LAMBDAPROX="0.005"
NME="deepmend"

echo "Loading from ""$1"
echo "Reconstructing $NME"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

python python/reconstruct.py \
    -e "$EXPERIMENT" \
    -c "$CHK" \
    --name "$NME" \
    --threads 2 \
    --num_iters "$NUMITS" \
    --lambda_reg "$LREG" \
    --learning_rate "$LR" \
    --render_threads 5 \
    --uniform_ratio "$UNIFORM_RATIO" \
    --lambda_ner "$LAMBDANER" \
    --lambda_prox "$LAMBDAPROX"