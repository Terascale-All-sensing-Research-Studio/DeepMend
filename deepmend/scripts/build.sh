# !/bin/bash
# Build pkl files for training and testing
#

ROOTDIR="$1"    # Path to the root of a fractured shapenet dataset
SPLITSFILE="$2" # Path to a splits file
OUTFILE="$3"    # Path to an output file (no extension)
NUMBREAKS="$4"  # Number of break to load

# Build the pkl files
python python/build.py \
    "$ROOTDIR" \
    "$SPLITSFILE" \
    --train_out "$OUTFILE"_train.pkl \
    --test_out "$OUTFILE"_test.pkl \
    --breaks "$NUMBREAKS" \
    --load_models \
    --use_spline 