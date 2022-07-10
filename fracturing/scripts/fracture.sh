# !/bin/bash
# Fracture the ShapeNet dataset
#

# Set the input arguments
ROOTDIR="$1" 
if [ -z "$1" ]; then
    echo "Must pass: ROOTDIR"; exit
fi
SPLITSFILE="$2" 
if [ -z "$2" ]; then
    echo "Must pass: SPLITSFILE"; exit
fi
OPERATION="$3"
if [ -z "$3" ]; then
    echo "Must pass: OPERATION"; exit
fi
NUMBREAKS="$4" 
if [ -z "$4" ]; then
    NUMBREAKS="10"
fi
SUBSAMPLE="6000"
FRACTURETHREADS=6

# Run 
case $OPERATION in
	"1")
        # Waterproof
        python -m processor.process \
            "$ROOTDIR" \
            "$SPLITSFILE" \
            WATERPROOF \
            --instance_subsample "$SUBSAMPLE" \
            --outoforder \
            -t 6
        ;;
    
	"2")
        # Clean
        python -m processor.process \
            "$ROOTDIR" \
            "$SPLITSFILE" \
            CLEAN \
            --outoforder \
            -t "$FRACTURETHREADS"
        ;;

	"3")
        # Break
        python -m processor.process \
            "$ROOTDIR" \
            "$SPLITSFILE" \
            BREAK \
            --breaks "$NUMBREAKS" \
            --break_all \
            --min_break 0.05 \
            --max_break 0.20 \
            --break_method surface-area \
            --outoforder \
            -t "$FRACTURETHREADS"
        ;;

	"4")
        # Compute the sample points 
        python -m processor.process \
            "$ROOTDIR" \
            "$SPLITSFILE" \
            SAMPLE \
            --breaks "$NUMBREAKS" \
            --outoforder \
            -t "$FRACTURETHREADS"
        ;;

	"5")
        # Compute SDF
        python -m processor.process \
            "$ROOTDIR" \
            "$SPLITSFILE" \
            SDF PARTIAL_SDF \
            --breaks "$NUMBREAKS" \
            --outoforder \
            -t 3
        ;;

	"6")
        # Compute Fracture Spline Fit
        python -m processor.process \
            "$ROOTDIR" \
            "$SPLITSFILE" \
            SPLINE \
            --breaks "$NUMBREAKS" \
            --outoforder \
            -t "$FRACTURETHREADS" --debug
        ;;

	"7")
        # Compute Voxels
        python -m processor.process \
            "$ROOTDIR" \
            "$SPLITSFILE" \
            VOXEL_32 \
            --breaks "$NUMBREAKS" \
            --outoforder \
            -t "$FRACTURETHREADS"
        ;;

	"8")
        # Compute Uniform Occupancy
        python -m processor.process \
            "$ROOTDIR" \
            "$SPLITSFILE" \
            UNIFORM_OCC \
            --breaks "$NUMBREAKS" \
            --outoforder \
            -t 3
        ;;
esac