# *_*coding:utf-8 *_*
import os
import json
import time
import warnings
import random
import numpy as np
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

import pickle
import trimesh
import tqdm
from scipy.spatial import cKDTree as KDTree


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def get_faces_from_vertices(vertex_mask, edges):
    """ Get faces containting vertices """
    vertex_index = np.nonzero(vertex_mask)[0]
    _, face_indices1, _ = np.intersect1d(edges[:, 0], vertex_index, return_indices=True)
    face_indices1 = face_indices1 // 3
    _, face_indices2, _ = np.intersect1d(edges[:, 1], vertex_index, return_indices=True)
    face_indices2 = face_indices2 // 3
    face_indices = np.hstack((face_indices1, face_indices2))
    num_faces = edges.shape[0] // 3
    face_mask = np.zeros((num_faces,))
    face_mask[face_indices] = 1
    return face_mask.astype(bool)


def get_faces_from_vertices2(vertex_mask, faces):
    """ Get faces containting vertices """
    vertex_index = set(np.nonzero(vertex_mask)[0])
    face_mask = np.zeros((faces.shape[0],))
    for idx, f in enumerate(faces):
        if f[0] in vertex_index or f[1] in vertex_index or f[2] in vertex_index:
            face_mask[idx] = 1
    return face_mask.astype(bool)


def get_fracture_from_models(model_c, model_b):
    """ Return a mask corresponding to the fracture vertices """
    # Get the fracture verts (this will work because B comes from C)
    d, _ = KDTree(model_c.vertices).query(model_b.vertices)
    fracture_vert_mask = d > 0.001
    assert fracture_vert_mask.mean() != 0
    assert fracture_vert_mask.mean() != 1
    return fracture_vert_mask


def get_model_dir(root, class_id, instance_id):
    return os.path.join(
        root, 
        class_id, 
        instance_id, 
        "models",
    )


def get_paths_from_model_dir(model_dir, break_num):
    """ Return the complete and broken paths """
    return (
        os.path.join(model_dir, "model_c.obj"), 
        os.path.join(model_dir, "model_b_{}.obj".format(break_num))
    )


def get_r_path(root, class_id, instance_id, break_num):
    """ Return the complete and broken paths """
    model_dir = get_model_dir(root, class_id, instance_id)
    return os.path.join(model_dir, "model_r_{}.obj".format(break_num))


def load_data_from_model_dir(model_dir, break_num, sample=True, count=100000):
    """ Return the broken vertices, the broken normals, and a mask of the fracture """
    path_c, path_b = get_paths_from_model_dir(model_dir, break_num)
    model_c = trimesh.load(path_c)
    model_b = trimesh.load(path_b)
    model_b.fix_normals() # Just in case

    if sample:
        # Densely sample the surface of the mesh
        vertices, face_indices = model_b.sample(count=count, return_index=True)
        normals = model_b.face_normals[face_indices, :]

        # Fet a mask of the fracture vertices
        fracture_vertex_mask = get_fracture_from_models(model_c, model_b)

        # Convert to a mask of the fracture faces
        fracture_face_mask = get_faces_from_vertices2(fracture_vertex_mask, model_b.faces)
        sample_mask = fracture_face_mask[face_indices]
    else:
        vertices = model_b.vertices
        normals = model_b.vertex_normals
        sample_mask = get_fracture_from_models(model_c, model_b)

    # Return: [vertices, normals, mask] 
    # where mask is 1 on the fracture region, 0 otherwise
    return np.hstack((
        vertices,
        normals,
        np.expand_dims(
            sample_mask, 
            axis=1,
        )
    ))


def build_data(
    roots, # Root dir
    save_path_pkl, # pkl save path
    save_path_json, # json save path
    split_list, # List of splits files
    split, # Either train or test
    ninstances, # Number of instances to select from each split
    rand_seed=1,
):

    # Load splits
    split_list = [
        json.load(open(p, "r")) 
        for p in split_list
    ]

    # Training and testing splits have different keys other than train and test
    key = "id_test_list"        
    if split == "train":
        key = "id_train_list"

    # Randomly (statically) sample from the input splits 
    random.seed(rand_seed)
    data = list()
    pbar = tqdm.tqdm(desc="Checking {} data".format(split))
    for dataset_split, root in zip(split_list, roots):

        class_split_list = dataset_split[key]
        class_split_list = [[c[0], c[1], br] for c in class_split_list for br in range(10)]
        random.shuffle(class_split_list)

        class_instances_collected = 0
        for ss in class_split_list:
            class_id, instance_id, break_num = ss

            # Only append to datalist if all relevant paths exist
            path_c, path_b = get_paths_from_model_dir(
                model_dir=get_model_dir(root, class_id, instance_id),
                break_num=break_num,
            )

            if os.path.exists(path_c) and os.path.exists(path_b):
                data.append(
                    [class_id, instance_id, break_num]
                )
                class_instances_collected += 1
                pbar.update(1)

            # Break when we've collected enough data
            if class_instances_collected >= ninstances:
                break
    pbar.close()

    print("Loaded {} samples".format(len(data)))
    if os.path.exists(save_path_json):
        input("file: {} exists, are you sure you want to overwrite?".format(save_path_json))
    print("Saving split to {}".format(save_path_json))
    json.dump(
        data,
        open(save_path_json, "w")
    )

    cache = {}
    for idx in tqdm.tqdm(range(len(data)),desc="Loading {} data".format(split)):
        class_id, instance_id, break_num = data[idx]
        cache[idx] = load_data_from_model_dir(
                model_dir=get_model_dir(root, class_id, instance_id),
                break_num=break_num,
            ).astype(np.float32)

    if os.path.exists(save_path_pkl):
        input("file: {} exists, are you sure you want to overwrite?".format(save_path_pkl))
    print("Saving data to {}".format(save_path_pkl))
    pickle.dump(
        [cache, data],
        open(save_path_pkl, "wb"),
        protocol=pickle.HIGHEST_PROTOCOL,
    )


class PartNormalDataset(Dataset):
    def __init__(
        self,
        data_file,
        root=None,
        npoints=2500,
        split='train',
        normal_channel=False,
    ):
        """
        Inputs: 
            list of training split files,
            root dir to shapenet,
        """

        assert split in ["train", "test"]
        self.npoints = npoints
        self.normal_channel = normal_channel
        # self.root = root
        self.cache_size = 20000
        self.cache = {} # This is a static cache that stores subsampled data
        
        data_file = data_file.replace("DATADIR", os.environ["DATADIR"]) 

        print("Loading saved data from: {}".format(data_file))
        self.main_cache, self.data = pickle.load(open(data_file, "rb"))

        # Remap classes to integers
        self.classes = {}
        class_list = list([d[0] for d in self.data])
        classes = sorted(list(set(class_list)))
        for class_idx, c in enumerate(classes):
            self.classes[c] = class_idx
            # print("{} instances of class {}".format(class_list.count(c), c))
        
    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_parts(self):
        return 2

    @property
    def seg_classes(self):
        if not hasattr(self, "_seg_classes"):
            self._seg_classes = {
                c : [0, 1] # All objects have fractured and non-fractured components
                for c in self.classes.keys()
            }
            # for index, d in enumerate(self.data):
            #     self._seg_classes[d[0]].append(index)
        return self._seg_classes

    def get_mesh(self, index):
        class_id, instance_id, break_num = self.data[index]
        _, path_b = get_paths_from_model_dir(
            model_dir=get_model_dir(self.root, class_id, instance_id),
            break_num=break_num,
        )
        return trimesh.load(path_b)

    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            class_id, instance_id, break_num = self.data[index]
            data = self.main_cache[index]

            cls = self.classes[class_id]
            cls = np.array([cls]).astype(np.int32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, cls, seg

    def __len__(self):
        return len(self.data)



