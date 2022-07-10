# !/bin/bash
# Automate training of ORGAN-3D
#

EXPDIR="$1"
if [ -z "$1" ]; then
    echo "Must pass: EXPDIR"; exit
fi
BATCHSIZE="$2"
if [ -z "$2" ]; then
    BATCHSIZE=140
fi

python -m reconstruction \
    -d "$EXPDIR" \
    -o "$EXPDIR" \
    --epochs 400 \
    --workers 10 \
    --overwrite \
    --bs "$BATCHSIZE"