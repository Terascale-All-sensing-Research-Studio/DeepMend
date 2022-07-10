# !/bin/bash
# Perform inference using deepsdf with fracture classifier
#

EXPERIMENT="$1"     # Path to an experiment to evaluate
CHK="2000"
NME="deepsdf_S"

echo "Loading from ""$1"
echo "Reconstructing $NME"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

python python/reconstruct.py \
	-e "$EXPERIMENT" \
	-c "$CHK" \
	--name "$NME" \
	--threads 2 \
	--render_threads 5 \
	--slippage 0.0 \
	--save_values \
	--slippage_method classifier