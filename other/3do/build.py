import os
import logging
import argparse
import random
from copy import deepcopy

import tqdm
import numpy as np

import core


def get_all_voxels(file_list, d=32):
    """
    Given a list of files, load voxels from each file and return.
    """
    voxel_accumulator = []

    for f in tqdm.tqdm(file_list):
        if not os.path.exists(f):
            print("skipping ", f)
            continue

        # Load
        voxel = np.load(f)["occ"]
        voxel = voxel.reshape((d, d, d))

        # Stack voxel on the 1st dimension
        voxel_accumulator.append(np.expand_dims(voxel, axis=0))

    print("Loaded {} samples".format(len(voxel_accumulator)))
    return np.concatenate(voxel_accumulator, axis=0)


def main(
    root_dir,
    train_splits_meta, 
    test_splits_meta, 
    out, 
):
    if not isinstance(train_splits_meta, list):
        train_splits_meta = [train_splits_meta]
    if not isinstance(test_splits_meta, list):
        test_splits_meta = [test_splits_meta]

    # Check that the paths are valid
    logging.info("Will save data to: {}".format(out))
    assert os.path.splitext(out)[-1] == ".npy"
    assert os.path.isdir(os.path.dirname(out)), "Save directory {} does not exist".format(out)

    # Load the training data
    train_file_list, train_labels = [], []
    train_file_list_broken = []
    for train_splits in train_splits_meta:

        logging.info("Loading saved data from train splits file {}".format(train_splits))
        try:
            if os.path.exists(core.quick_load_path(train_splits)):
                train_splits = core.quick_load_path(train_splits)
            data_handler = core.data.SamplesDataset(
                train_splits,
                root=root_dir,
                load_values=False,
            )
            
            # Get all of the paths to the stored voxel files
            index_list = range(len(data_handler))
            for idx in index_list:
                break_idx = data_handler.get_broken_index(idx)
                obj = data_handler.get_object(idx)
                path_complete = obj.path_c_voxel()
                path_broken = obj.path_b_voxel(break_idx)
                if os.path.exists(path_complete) and os.path.exists(path_broken):
                    train_file_list.append(path_complete)
                    train_labels.append(obj.class_id)
                    train_file_list_broken.append(path_broken)
                else:
                    raise RuntimeError("missing voxel sample {}, {}".format(
                        path_complete, path_broken
                    ))

            # Load the voxels
            logging.info("Loading voxels from train list...")
            train_data = get_all_voxels(train_file_list)
            train_data_complete = deepcopy(train_data)
            train_data_broken = get_all_voxels(train_file_list_broken)
        except IndexError:
            train_file_list, train_labels, train_data = [], [], []
        
    # Load the training data
    test_file_list, test_labels = [], []
    test_file_list_complete = []
    for test_splits in test_splits_meta:

        logging.info("Loading saved data from test splits file {}".format(test_splits))

        if os.path.exists(core.quick_load_path(test_splits)):
            test_splits = core.quick_load_path(test_splits)
        data_handler = core.data.SamplesDataset(
            test_splits,
            root=root_dir,
        )
        
        # Get all of the paths to the stored voxel files
        index_list = range(len(data_handler))
        for idx in index_list:
            break_idx = data_handler.get_broken_index(idx)
            obj = data_handler.get_object(idx)
            path_complete = obj.path_c_voxel()
            path_broken = obj.path_b_voxel(break_idx)
            if os.path.exists(path_complete) and os.path.exists(path_broken):
                test_file_list.append(path_broken)
                test_labels.append(obj.class_id)
                test_file_list_complete.append(path_complete)
            else:
                raise RuntimeError("missing voxel sample {}, {}".format(
                    path_complete, path_broken
                ))

        # Load the voxels
        logging.info("Loading voxels from test list...")
        test_data = get_all_voxels(test_file_list)
        test_data_complete = get_all_voxels(test_file_list_complete)
        test_data_broken = deepcopy(test_data)

    # Build the dictionary
    logging.info("Saving data ...")
    np.save(
        out, 
        {
            "train" : {
                "file_list": train_file_list,
                "labels": train_labels, 
                "data": train_data,
                "data_broken": train_data_complete,
                "data_complete": train_data_broken,
                "errors": [],
            },
            "test" : {
                "file_list": test_file_list,
                "labels": test_labels,
                "data": test_data,
                "data_broken": test_data_broken,
                "data_complete": test_data_complete,
                "errors": [],
            },
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "--root_dir",
        default=None,
        type=str,
        help="Data root directory. Typically the top level directory of shapenet.",
    )
    parser.add_argument(
        "--train_splits",
        default=None,
        type=str,
        nargs="+",
        help="Where to load the train database file. Should be a .pkl",
    )
    parser.add_argument(
        "--test_splits",
        default=None,
        type=str,
        nargs="+",
        help="Where to load the train database file. Should be a .pkl",
    )
    parser.add_argument(
        "--out",
        default=None,
        type=str,
        help="Where to save the resulting train database file. Should be a .npy",
    )
    core.add_common_args(parser)
    args = parser.parse_args()
    core.configure_logging(args)

    main(
        root_dir=args.root_dir,
        train_splits_meta=args.train_splits,
        test_splits_meta=args.test_splits,
        out=args.out,
    )