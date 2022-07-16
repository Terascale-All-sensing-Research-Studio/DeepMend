import os
import argparse
import logging

import trimesh
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree as KDTree

import processor.logger as logger
import processor.errors as errors

import core


def points_maximal_orient(points):
    """ Return the transformation matrix that orients a point set by its maximal dimensions """
    pca = PCA(n_components=3)
    pca.fit(points)
    matrix = pca.components_
    return np.vstack((
        np.hstack((
            np.expand_dims(matrix[2, :], axis=1),
            np.expand_dims(matrix[1, :], axis=1),
            np.expand_dims(matrix[0, :], axis=1),
            np.zeros((3, 1))
        )),
        np.array([0, 0, 0, 1])
    )).T


def smooth(f_in, f_out, lamb=0.5, iterations=10, overwrite=False):
    """Perform laplacian smoothing on a mesh"""
    # Load mesh
    mesh = trimesh.load(f_in)
    if len(mesh.vertices) > 1000000:
        raise errors.MeshSizeError
    if not mesh.is_watertight:
        raise errors.MeshNotClosedError
    mesh = trimesh.smoothing.filter_laplacian(mesh, lamb, iterations)

    # mesh.vertex_normals
    if overwrite or not os.path.exists(f_out):
        logging.debug("Saving to: {}".format(f_out))
        mesh.export(f_out)


def normalize_unit_cube(mesh):
    """Normalize a mesh so that it occupies a unit cube"""

    # Get the overall size of the object
    mesh = mesh.copy()
    mesh_min, mesh_max = np.min(mesh.vertices, axis=0), np.max(mesh.vertices, axis=0)
    size = mesh_max - mesh_min

    # Center the object
    mesh.vertices = mesh.vertices - ((size / 2.0) + mesh_min)

    # Normalize scale of the object
    mesh.vertices = mesh.vertices * (1.0 / np.max(size))
    return mesh


def normalize(f_in, f_out, skip_check=False, overwrite=False, reorient=False):
    """Translate and rescale a mesh so that it is centered inside a unit cube"""
    # Load mesh
    mesh = trimesh.load(f_in)
    if not skip_check:
        if len(mesh.vertices) > 1000000:
            raise errors.MeshSizeError
        if not mesh.is_watertight:
            mesh = core.repair_self_intersection(mesh)
        if not mesh.is_watertight:
            mesh = core.repair_watertight(mesh)
            mesh = core.repair_self_intersection(mesh)
        if not mesh.is_watertight:
            raise errors.MeshNotClosedError

    mesh = normalize_unit_cube(mesh)
    if reorient:
        mesh.apply_transform(
            points_maximal_orient(mesh.vertices)
        )  
        normalize_unit_cube(mesh)

    # Save
    if overwrite or not os.path.exists(f_out):
        logging.debug("Saving to: {}".format(f_out))
        mesh.export(f_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply laplacian smoothing or unit cube normalization. Will "
        + "fail if the mesh is not waterproof."
    )
    parser.add_argument(dest="input", type=str, help="Path to the input file.")
    parser.add_argument(dest="output", type=str, help="Path to the output file.")
    parser.add_argument(
        "--smooth",
        action="store_true",
        default=False,
        help="If passed, will smooth the mesh instead of performing unit cube "
        + "normalization.",
    )
    parser.add_argument(
        "--lamb", type=float, default=0.5, help="Lambda value for laplacian smoothing."
    )
    parser.add_argument(
        "--iterations", type=int, default=10, help="Iterations for laplacian smoothing."
    )
    parser.add_argument(
        "--skip_check", action="store_true", default=False, help="Skip size check."
    )
    logger.add_logger_args(parser)
    args = parser.parse_args()
    logger.configure_logging(args)

    # Process
    if args.smooth:
        smooth(
            f_in=args.input,
            f_out=args.output,
            lamb=args.lamb,
            iterations=args.iterations,
        )
    else:
        normalize(f_in=args.input, f_out=args.output, skip_check=args.skip_check)
