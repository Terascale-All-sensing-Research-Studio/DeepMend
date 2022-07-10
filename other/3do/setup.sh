conda activate 3do-gpu
echo "The primary data directory is located at: "$DATADIR

unset PYTHONPATH

# Add local libraries to pythonpath
export PYTHONPATH=$PYTHONPATH:`pwd`/../fracturing
export PYTHONPATH=$PYTHONPATH:`pwd`/../deepmend/python

# Add dependancies to pythonpath
if [ -d "libs" ]; then
    export PYTHONPATH=$PYTHONPATH:`pwd`/libs/mesh-fusion
    export PYTHONPATH=$PYTHONPATH:`pwd`/libs/inside_mesh
    export LIBRENDERPATH=`pwd`/libs/mesh-fusion/librender
else
    echo "Library directory does not exist, cannot add libraries"
fi
