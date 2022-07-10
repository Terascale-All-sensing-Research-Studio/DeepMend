
from pointnet.data_utils.ShapeNetDataLoader import build_data

DEFAULT_SPLIT_LIST = []     # A list of split files
ROOTS = []                  # A list of root directories to fractured shapenet
SAVE_PATH_PKL = ""          # Template output file as a pkl
SAVE_PATH_JSON = ""         # Template output file as a json
NUM_INSTACES = 400          # Number of instances to pick from each class

for key in ["train", "test"]:
    build_data(
        roots=ROOTS, # Root dir
        save_path_pkl=SAVE_PATH_PKL.format(key), # pkl save path
        save_path_json=SAVE_PATH_JSON.format(key), # json save path
        split_list=DEFAULT_SPLIT_LIST, # List of splits files
        split=key, # Either train or test
        ninstances=NUM_INSTACES, # Number of instances to select from each split
        rand_seed=1,
    )