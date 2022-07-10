import os, argparse
import logging

# Set the available threads to (1/3)
os.environ["OMP_NUM_THREADS"] = str(int(os.cpu_count() * (1 / 3)))

import trimesh
import numpy as np
from pykdtree.kdtree import KDTree

from libmesh import check_mesh_contains

import processor.logger as logger
import processor.errors as errors
from processor.utils_3d import repair_self_intersection
import processor.process_sample as sampler


def compute_sdf(
    mesh,
    points=None,
    surface_points=10000000,
):

    # Get distances
    logging.debug("Computing SDF approximation using {} points".format(surface_points))
    sdf, _ = KDTree(mesh.sample(count=surface_points).astype(np.float64)).query(
        points.astype(np.float64)
    )

    # Apply sign using check mesh contains
    logging.debug("Computing sign")
    sign = check_mesh_contains(mesh, points).astype(int)
    sdf[sign == 1] = -np.abs(sdf[sign == 1])
    sdf[sign != 1] = np.abs(sdf[sign != 1])
    return sdf


def process(
    f_in,
    f_out,
    f_samp=None,
    n_points=500000,
    surface_points=10000000,
    uniform_ratio=0.5,
    padding=0.1,
    sigma=0.01,
    min_percent=0.02,
    overwrite=False,
):
    # Load meshes
    mesh = trimesh.load(f_in)
    if not mesh.is_watertight:
        raise errors.MeshNotClosedError

    if f_samp is None:
        # Get sample points
        points = sampler.sample_points(
            mesh=mesh,
            n_points=n_points,
            uniform_ratio=uniform_ratio,
            padding=padding,
            sigma=sigma,
        )
    else:
        logging.debug("Loading sample points from {}".format(f_samp))
        points = np.load(f_samp)["xyz"]
        assert (
            points.shape[0] == n_points
        ), "Loaded sample points were the wrong size {} vs {}".format(
            points.shape[0], n_points
        )

    # Compute sdf
    sdf = compute_sdf(
        mesh=mesh,
        points=points,
        surface_points=surface_points,
    )

    # Must have at least this many points, else is a bad sample
    logging.debug(
        "Mesh had {}/{} interior points".format((sdf < 0).sum(), sdf.shape[0])
    )
    if (sdf < 0).sum() < (n_points * min_percent):
        raise errors.MeshEmptyError

    # Compress
    if overwrite or not os.path.exists(f_out):
        logging.debug("Saving to: {}".format(f_out))
        np.savez(f_out, sdf=sdf.astype(np.float16))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computes the occupancy values for samples points on and "
        + "around an object. Accepts the arguments common for sampling."
    )
    parser.add_argument(dest="input", type=str, help="Path to the input file.")
    parser.add_argument(dest="output", type=str, help="Path to the output file.")
    parser.add_argument(
        "--samples",
        type=str,
        help="Input file that stores sample points to use (.npz).",
    )
    parser.add_argument(
        "--uniform",
        "-r",
        type=float,
        default=0.5,
        help="Uniform ratio. eg 1.0 = all uniform points, no surface points.",
    )
    parser.add_argument(
        "--n_points",
        "-n",
        type=int,
        default=500000,
        help="Total number of sample points.",
    )
    parser.add_argument(
        "--padding",
        "-p",
        type=float,
        default=0.1,
        help="Extra padding to add when performing uniform sampling. eg 0 = "
        + "uniform sampling is done in unit cube.",
    )
    parser.add_argument(
        "--sigma",
        "-s",
        type=float,
        default=0.01,
        help="Sigma used to compute surface points perturbation.",
    )
    logger.add_logger_args(parser)
    args = parser.parse_args()
    logger.configure_logging(args)

    process(
        f_in=args.input,
        f_out=args.output,
        f_samp=args.samples,
        n_points=args.n_points,
        uniform_ratio=args.uniform,
        padding=args.padding,
        sigma=args.sigma,
    )
