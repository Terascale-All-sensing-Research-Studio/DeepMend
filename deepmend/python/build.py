import os
import argparse
import logging
import json
import pickle

import tqdm
import trimesh
import zipfile
import numpy as np

import core
from processor.shapenet import ShapeNetObject


def main(
    root_dir,
    train_out,
    train_c_out,
    train_check,
    test_out,
    splits_file,
    num_breaks,
    load_models,
    use_spline,
    use_occ,
    use_pointer,
    z_rot,
):
    logging.info("Loading saved data from splits file {}".format(splits_file))

    object_id_dict = json.load(open(splits_file, "r"))
    id_train_list = [
        ShapeNetObject(root_dir, o[0], o[1]) for o in object_id_dict["id_train_list"]
    ]
    id_test_list = [
        ShapeNetObject(root_dir, o[0], o[1]) for o in object_id_dict["id_test_list"]
    ]

    if train_c_out is not None:
        assert not use_occ

    if (train_out is not None) or (train_c_out is not None):
        # System to double check that two train databases are the same
        if train_check is not None:
            logging.info("Loading saved data from check file {}".format(train_check))
            check_data = pickle.load(open(train_check, "rb"))
            check_indexer = check_data["indexer"]
            check_data.clear()
            del check_data            

        logging.info("Will save train data to: {}".format(train_out))
        sdf_list = []
        occ_value_list = []
        index_list = []
        complete_index_list = []
        flipped_list = []
        logging.info("Loading train list")
        for obj_idx, obj in tqdm.tqdm(enumerate(id_train_list)):

            breaks_loaded = []
            for break_idx in range(num_breaks):
                
                # Here's all the data we're going to load
                path_list = [
                    obj.path_sampled(break_idx),
                    obj.path_c_sdf(break_idx),
                    obj.path_b_sdf(break_idx),
                    obj.path_r_sdf(break_idx)
                ]
                if use_spline:
                    path_list.append(
                        obj.path_spline_sdf(break_idx)
                    )
                else:
                    path_list.append(
                        obj.path_tool_sdf(break_idx)
                    )

                print([os.path.exists(p) for p in path_list])

                if (
                    all([os.path.exists(p) for p in path_list])
                ):
                    try:
                        if use_spline:
                            pts, c, b, r, t = (
                                np.load(path_list[0])["xyz"],
                                np.expand_dims(np.load(path_list[1])["sdf"], axis=1),
                                np.expand_dims(np.load(path_list[2])["sdf"], axis=1),
                                np.expand_dims(np.load(path_list[3])["sdf"], axis=1),
                                np.expand_dims(np.load(path_list[4])["occ"], axis=1).astype(int),
                            )
                            sdf_sample = np.hstack((
                                pts, c, b, r, t
                            ))
                        else:
                            pts, c, b, r, t = (
                                np.load(path_list[0])["xyz"],
                                np.expand_dims(np.load(path_list[1])["sdf"], axis=1),
                                np.expand_dims(np.load(path_list[2])["sdf"], axis=1),
                                np.expand_dims(np.load(path_list[3])["sdf"], axis=1),
                                np.expand_dims(np.load(path_list[4])["sdf"], axis=1),
                            )
                            sdf_sample = np.hstack((
                                pts, c, b, r, t
                            ))
                    except (zipfile.BadZipFile):
                        logging.warning(
                            "Sample ({}, {}) is corrupted, skipping".format(
                                obj_idx, break_idx
                            )
                        )
                        continue

                    index_tuple = (obj_idx, break_idx)

                    # Data has format:
                    # [xyz, c, b, r, t]
                    # [012, 3, 4, 5, 6]

                    if z_rot:
                        # Flip a coin
                        if np.random.randint(0, 2):
                            sdf_sample[:, :3] = core.points_transform(
                                sdf_sample[:, :3], 
                                trimesh.transformations.rotation_matrix(
                                    angle=np.radians(90), direction=[0, 0, 1], point=(0, 0, 0)
                            ))
                            flipped_list.append(True)
                        else:
                            flipped_list.append(False)
                    else:
                        flipped_list.append(False)
                    
                    if use_spline:
                        # If using spline, convert to sdf(ish)
                        sdf_sample[sdf_sample[:, -1] == 1, -1] = -1
                        sdf_sample[sdf_sample[:, -1] == 0, -1] = 1

                    occ_sample = core.data.sdf_to_occ(
                        sdf_sample.astype(np.float16), skip_cols=3
                    )

                    # This is a sanity check
                    if (
                        not (
                            occ_sample[:, 3] == occ_sample[:, 4] + occ_sample[:, 5]
                        ).all()
                        or not (
                            occ_sample[:, 4]
                            == np.clip(occ_sample[:, 3] - occ_sample[:, 6], 0, 1)
                        ).all()
                        or not (
                            occ_sample[:, 5]
                            == np.clip(occ_sample[:, 3] + occ_sample[:, 6], 1, 2) - 1
                        ).all()
                    ):
                        logging.warning(
                            "Sample ({}, {}) is invalid, skipping.".format(
                                *index_tuple
                            )
                        )
                        continue
                    
                    # Double check against the input train check
                    if train_check is not None:
                        if not index_tuple in check_indexer:
                            logging.info("Sample ({}, {}) not in train check, skipping.".format(*index_tuple))
                            continue
                    
                    # Add all the data to the corresponding lists
                    if use_occ:
                        sdf_sample = occ_sample
                        occ_value_list.append(occ_sample.astype(bool))
                        sdf_list.append(sdf_sample[:, :3].astype(np.float16))
                    else:
                        sdf_list.append(sdf_sample.astype(np.float16))
                            
                    breaks_loaded.append(len(sdf_list))
                    index_list.append(
                        index_tuple
                    )

            if len(breaks_loaded) > 0:
                complete_index_list.append(breaks_loaded)

            # Make sure the cache is empty
            obj._cache = {}

        if use_occ:
            logging.info("num samples loaded: {}".format(len(occ_value_list)))
        else:
            logging.info("num samples loaded: {}".format(len(sdf_list)))
        logging.info("Saving data ...")

        # Double check against the input train check
        if train_check is not None:
            logging.info("Checking that all samples were added")
            for index_tuple in check_indexer:
                if index_tuple not in index_list:
                    logging.info("Sample ({}, {}) not added.".format(*index_tuple))
            logging.info("Loaded samples: {}".format(len(index_list)))
            logging.info("Check samples:  {}".format(len(check_indexer)))

        if train_out is not None:
            if os.path.exists(train_out):
                input(
                    "File: {} already exists, are you sure you want to overwrite?".format(
                        train_out
                    )
                )
            data_dict = {
                "indexer": index_list,
                "complete_indexer": complete_index_list,
                "objects": id_train_list,
                "use_occ": use_occ,
                "train": True,
                "zrot90": flipped_list,
            }
            tr_out, _ = os.path.splitext(train_out)

            if use_pointer:
                sdf_path = tr_out + "_sdf.npz"
                logging.info("Saving sdf values to: {}".format(sdf_path))

                data_dict["sdf"] = sdf_path
                np.savez(sdf_path, sdf=sdf_list)
            else:
                data_dict["sdf"] = sdf_list

            if use_occ:
                data_dict["occ_values"] = occ_value_list

            logging.info("Saving data_dict to: {}".format(train_out))
            pickle.dump(
                data_dict,
                open(train_out, "wb"),
                pickle.HIGHEST_PROTOCOL,
            )
        if train_c_out is not None:
            if os.path.exists(train_c_out):
                input(
                    "File: {} already exists, are you sure you want to overwrite?".format(
                        train_c_out
                    )
                )
            for idx in range(len(sdf_list)):
                sdf_list[idx] = sdf_list[idx][:, :4]
            pickle.dump(
                {
                    "sdf": sdf_list,
                    "indexer": index_list,
                    "complete_indexer": complete_index_list,
                    "objects": id_train_list,
                    "use_occ": use_occ,
                    "train": True,
                    "zrot90": flipped_list,
                },
                open(train_c_out, "wb"),
                pickle.HIGHEST_PROTOCOL,
            )

    if test_out is not None:
        logging.info("Will save test data to: {}".format(test_out))
        sdf_list = []
        index_list = []
        complete_index_list = []
        logging.info("Loading test list")
        for obj_idx, obj in tqdm.tqdm(enumerate(id_test_list)):

            breaks_loaded = []
            for break_idx in range(num_breaks):
                if os.path.exists(obj.path_b_partial_sdf(break_idx)):
                    try:
                        sdf_sample = obj.load(
                            obj.path_b_partial_sdf(break_idx), skip_cache=True
                        )
                    except (zipfile.BadZipFile):
                        logging.warning(
                            "Sample ({}, {}) is corrupted, skipping".format(
                                obj_idx, break_idx
                            )
                        )
                        continue

                    breaks_loaded.append(len(sdf_list))
                    sdf_list.append(sdf_sample)
                    index_list.append(
                        (
                            obj_idx,
                            break_idx,
                        )
                    )

                    if load_models:
                        obj.load(obj.path_b(break_idx))
                        obj.load(obj.path_r(break_idx))

            if len(breaks_loaded) > 0:
                complete_index_list.append(breaks_loaded)
                if load_models:
                    obj.load(obj.path_c())

        if os.path.exists(test_out):
            input(
                "File: {} already exists, are you sure you want to overwrite?".format(
                    test_out
                )
            )
        logging.info("num samples loaded: {}".format(len(sdf_list)))
        logging.info("Saving data ...")
        pickle.dump(
            {
                "sdf": sdf_list,
                "indexer": index_list,
                "complete_indexer": complete_index_list,
                "objects": id_test_list,
                "use_occ": False,
                "train": False,
            },
            open(test_out, "wb"),
            pickle.HIGHEST_PROTOCOL,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a train and test pkl file from a splits file. The "
        + "pkl files will only contain valid samples. Optionally preload "
        + "upsampled points and models to accelerate evaluation."
    )
    parser.add_argument(
        dest="input",
        type=str,
        help="Location of the database. Pass the top level directory. For "
        + "ShapeNet this would be ShapeNet.v2",
    )
    parser.add_argument(
        dest="splits",
        type=str,
        help=".json file path, this file will be created and will store the "
        + "ids of all objects in the training and testing split.",
    )
    parser.add_argument(
        "--train_check",
        default=None,
        type=str,
        help="Train database file to check against. Should be a .pkl",
    )
    parser.add_argument(
        "--train_out",
        default=None,
        type=str,
        help="Where to save the resulting train database file. Should be a .pkl",
    )
    parser.add_argument(
        "--train_c_out",
        default=None,
        type=str,
        help="Where to save the resulting train database file. Should be a .pkl. "
        + "This file is a simplified version of the training database file that "
        + "only contains the complete shape."
    )
    parser.add_argument(
        "--test_out",
        default=None,
        type=str,
        help="Where to save the resulting test database file. Should be a .pkl",
    )
    parser.add_argument(
        "--breaks",
        "-b",
        type=int,
        default=1,
        help="Number of breaks to generate for each object. This will only be "
        + "used if BREAK is passed.",
    )
    parser.add_argument(
        "--load_models",
        action="store_true",
        default=False,
        help="If passed, will preload object models. Note this is only applicable " 
        + "to the test database file."
    )
    parser.add_argument(
        "--use_spline",
        action="store_true",
        default=False,
        help="If passed, will use the spline approximations."
    )
    parser.add_argument(
        "--use_occ",
        action="store_true",
        default=False,
        help="If passed, will load data in occ mode."
    )
    parser.add_argument(
        "--use_pointer",
        action="store_true",
        default=False,
        help="If passed, will save a pointer to the file."
    )
    parser.add_argument(
        "--z_rot",
        action="store_true",
        default=False,
        help="If passed, will randomly rotate half of the training points "
        + "around the z axis by 90 degrees."
    )
    core.add_common_args(parser)
    args = parser.parse_args()
    core.configure_logging(args)

    main(
        args.input,
        args.train_out,
        args.train_c_out,
        args.train_check,
        args.test_out,
        args.splits,
        args.breaks,
        args.load_models,
        args.use_spline,
        args.use_occ,
        args.use_pointer,
        args.z_rot,
    )
