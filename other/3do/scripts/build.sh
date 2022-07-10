# !/bin/bash
# Build a new data file for ORGAN-3D from existing pkl files
#

ROOTDIR="$1"        # Path to the root of a fractured shapenet dataset
SPLITSFILE="$1"     # Path to train/test pkl files

# Build the train and test files
python build.py \
    --root_dir "$ROOTDIR" \
    --train_splits "$SPLITSFILE"_train.pkl \
    --test_splits "$SPLITSFILE"_test.pkl \
    --out "$SPLITSFILE".npy