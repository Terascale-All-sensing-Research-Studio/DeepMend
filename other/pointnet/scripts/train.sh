# !/bin/bash
#  Train pointnet
#

INPUT_DATA="$1"         # whatever you put for SAVE_PATH_PKL
INPUT_BATCH_SIZE="12"
INPUT_FRAC_WEIGHT="0.9"
INPUT_NPOINTS="16384"
INPUT_EPOCHS="250"

python train.py \
    --train_pkl "$INPUT_DATA"_train.pkl \
    --test_pkl "$INPUT_DATA"_test.pkl \
    --normal \
    --model pointnet_part_seg \
    --batch_size "$INPUT_BATCH_SIZE" \
    --frac_weight "$INPUT_FRAC_WEIGHT" \
    --npoint "$INPUT_NPOINTS" \
    --epoch "$INPUT_EPOCHS" \
    --log_dir fracture_classifier
