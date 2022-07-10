import os
import argparse
import logging
import subprocess

import trimesh
import numpy as np

from libmesh import check_mesh_contains

import processor.errors as errors
import processor.process_sample as sampler


def process(
    f_in,
    f_out,
    dim=256,
    padding=0.1,
    overwrite=False,
):
    # Load the mesh
    mesh = trimesh.load(f_in)

    # Mesh must be watertight
    if not mesh.is_watertight:
        raise errors.MeshNotClosedError

    # Get uniform points
    points = sampler.uniform_sample_points(dim=dim, padding=padding)

    # Get occupancies
    occupancies = check_mesh_contains(mesh, points)
    logging.debug(
        "Mesh had {}/{} interior points".format(
            occupancies.astype(int).sum(), occupancies.shape[0]
        )
    )

    # Save as boolean values
    if overwrite or not os.path.exists(f_out):
        logging.debug("Saving to: {}".format(f_out))
        np.savez(f_out, occ=occupancies.astype(bool))


def handsoff(*args, **kwargs):

    cmd = ["python " + os.path.abspath(__file__)]

    # Badness, but prevents segfault
    for a in args:
        cmd[0] += " " + str(a)
    for k, v in kwargs.items():
        cmd[0] += " --" + str(k) + " " + str(v)

    logging.debug("Executing command in the shell: \n{}".format(cmd))

    if subprocess.call(cmd, shell=True) != 0:
        raise RuntimeError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computes the occupancy values for samples points on and "
        + "around an object. Accepts the arguments common for sampling."
    )
    parser.add_argument(dest="f_in", type=str, help="Path to the input file.")
    parser.add_argument(dest="f_out", type=str, help="Path to the output file.")
    parser.add_argument(
        "--dim",
        "-d",
        type=int,
        default=256,
        help="Dimension of point samples.",
    )
    parser.add_argument(
        "--padding",
        "-p",
        type=float,
        default=0.1,
        help="Extra padding to add when performing uniform sampling. eg 0 = "
        + "uniform sampling is done in unit cube.",
    )
    args = parser.parse_args()

    process(
        f_in=args.f_in,
        f_out=args.f_out,
        dim=args.dim,
        padding=args.padding,
    )
