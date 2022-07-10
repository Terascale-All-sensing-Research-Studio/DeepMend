import os
import argparse
import logging

import trimesh
import numpy as np

from libmesh import check_mesh_contains

import processor.errors as errors
import processor.process_sample as sampler


def process(
    f_in,
    f_out,
    f_samp=None,
    n_points=500000,
    uniform_ratio=0.5,
    padding=0.2,
    sigma=0.01,
    min_percent=0.02,
    overwrite=False,
):
    # Load the mesh
    mesh = trimesh.load(f_in)

    # Mesh must be watertight
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
        points = np.load(f_samp)["xyz"]
    assert points.shape[0] == n_points, "Loaded sample points were the wrong size"

    # Get occupancies
    occupancies = check_mesh_contains(mesh, points)

    # Must have at least this many points, else is a bad sample
    logging.debug(
        "Mesh had {}/{} interior points".format(
            occupancies.astype(int).sum(), occupancies.shape[0]
        )
    )
    if occupancies.astype(int).sum() < (n_points * min_percent):
        raise errors.MeshEmptyError

    # Save as boolean values
    if overwrite or not os.path.exists(f_out):
        logging.debug("Saving to: {}".format(f_out))
        np.savez(f_out, occ=occupancies.astype(bool))


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
        help="Input file that stores sample points to use (.ply).",
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
        default=100000,
        help="Total number of sample points.",
    )
    parser.add_argument(
        "--padding",
        "-p",
        type=float,
        default=0.2,
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
    parser.add_argument(
        "--min_percent",
        "-m",
        type=float,
        default=0.02,
        help="Minimum number of positive points or will raise error.",
    )
    args = parser.parse_args()

    process(
        f_in=args.input,
        f_out=args.output,
        f_samp=args.samples,
        n_points=args.n_points,
        uniform_ratio=args.uniform,
        padding=args.padding,
        sigma=args.sigma,
        min_percent=args.min_percent,
    )
