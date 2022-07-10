import os
import logging

import pickle
import trimesh
import numpy as np
from PIL import Image
import torch


def loader(f_in):
    """Multipurpose loader used for all file types"""
    extension = os.path.splitext(f_in)[-1]
    logging.debug("Attempting to load file {}".format(f_in))
    if extension == ".pkl":
        with open(f_in, "rb") as f:
            return pickle.load(f)
    elif extension == ".obj":
        return trimesh.load(f_in, force=True)
    elif extension == ".npz":
        return dict(np.load(f_in, allow_pickle=True))
    elif extension == ".npy":
        return np.load(f_in, allow_pickle=True)
    elif extension == ".png":
        return np.array(Image.open(f_in))
    elif extension == ".pth":
        return torch.load(f_in, map_location=torch.device("cpu"))
    else:
        raise RuntimeError("Loader: Unhandled file type: {}".format(f_in))


def saver(f_out, data):
    """Multipurpose saver used for all file types"""
    logging.debug("Saving file {}".format(f_out))
    extension = os.path.splitext(f_out)[-1]
    if extension == ".obj":
        data.export(f_out)
    elif extension == ".png":
        Image.fromarray(data).save(f_out)
    elif extension == ".pth":
        torch.save(data, f_out)
    elif extension == ".npz":
        np.savez(f_out, data)
    elif extension == ".npy":
        np.save(f_out, data)
    else:
        raise RuntimeError("Saver: Unhandled file type: {}".format(f_out))


def save(path, lst):
    """Save a list of database objects"""
    pickle.dump(
        lst,
        open(path, "wb"),
        protocol=pickle.HIGHEST_PROTOCOL,
    )


def load(path):
    """Load a list of database objects"""
    return pickle.load(open(path, "rb"))


class DatabaseObject:
    def __init__(self, root_dir, class_id, instance_id):
        self._root_dir = root_dir
        self._class_id = class_id
        self._instance_id = instance_id

        self._cache = {}

    def __repr__(self):
        return (
            "DatabaseObject("
            + self._root_dir
            + ", "
            + self._class_id
            + ", "
            + self._instance_id
            + ")"
        )

    def load(self, f_in, skip_cache=False):
        if f_in in self._cache and not skip_cache:
            logging.debug("Pulling file from cache: {}".format(f_in))
            return self._cache[f_in]

        data = loader(f_in)

        if not skip_cache:
            self._cache[f_in] = data
        return data

    def __hash__(self):
        return hash(str(self))

    @property
    def root_dir(self):
        return self._root_dir

    @property
    def class_id(self):
        return self._class_id

    @property
    def instance_id(self):
        return self._instance_id

    def path(self, *args, **kwargs):
        raise NotImplementedError

    def build_dirs(self):
        raise NotImplementedError


class ThreeDNetObject(DatabaseObject):

    def path_normalized(self):
        return os.path.join(
            self._root_dir,
            self._class_id,
            self._instance_id + ".ply",
        )
    
    def path_waterproofed(self):
        return os.path.join(
            self._root_dir,
            self._class_id,
            self._instance_id + "_model_waterproofed.ply",
        )
    
    def path_c(self):
        return os.path.join(
            self._root_dir,
            self._class_id,
            self._instance_id + "_model_c.ply",
        )
    
    def path_random_reorient(self, idx):
        return os.path.join(
            self._root_dir,
            self._class_id,
            self._instance_id + "model_reoriented_{}.ply".format(idx),
        )
    
    def path_samples_reorient(self, idx):
        return os.path.join(
            self._root_dir,
            self._class_id,
            self._instance_id + "samples_reoriented_{}.npz".format(idx),
        )
    
    def path_sdf_reorient(self, idx):
        return os.path.join(
            self._root_dir,
            self._class_id,
            self._instance_id + "sdf_reoriented_{}.npz".format(idx),
        )

    def path_ptcld_complete(self):
        return os.path.join(
            self._root_dir,
            self._class_id,
            self._instance_id + "_ptcld_complete.ply",
        )

    def path_depth_map(self, index):
        return os.path.join(
            self._root_dir,
            self._class_id,
            self._instance_id + "_depth_map_{}.png".format(index),
        )

    def path_depth_map_trans_mat(self):
        return os.path.join(
            self._root_dir,
            self._class_id,
            self._instance_id + "_depth_map_trans_map.json",
        )

    def path_ptcld_partial(self, index):
        return os.path.join(
            self._root_dir,
            self._class_id,
            self._instance_id + "_ptcld_partial_{}.ply".format(index),
        )
    
    def build_dirs(self):
        pass


class HiRestObject(DatabaseObject):

    def build_dirs(self):
        """Build any required subdirectories"""
        # List of all the directories to build
        dir_list = [
            os.path.join(self._root_dir, self._class_id, self._instance_id),
            os.path.join(self._root_dir, self._class_id, self._instance_id, "models"),
            os.path.join(self._root_dir, self._class_id, self._instance_id, "renders"),
        ]

        # Build the directories
        for d in dir_list:
            if not os.path.exists(d):
                os.mkdir(d)

    def path(self, subdir, basename):
        """Return the path to a subdir + basename"""
        return os.path.join(
            self._root_dir,
            self._class_id,
            self._instance_id,
            subdir,
            basename,
        )

    def path_normalized(self):
        return self.path("models", "model_normalized.obj")

    def path_waterproofed(self):
        return self.path("models", "model_waterproofed.obj")
    
    def path_ptcld_complete(self):
        return self.path("models", "ptcld_complete.obj")

    def path_depth_map(self, index):
        return self.path("models", "depth_map_{}.png".format(index))

    def path_depth_map_trans_mat(self):
        return self.path("models", "depth_map_trans_map.json")

    def path_ptcld_partial(self, index):
        return self.path("models", "ptcld_partial_{}.ply".format(index))

    def build_dirs(self):
        pass