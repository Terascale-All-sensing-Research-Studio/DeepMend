import os, argparse
import logging

import trimesh
import numpy as np

import processor.errors as errors
import processor.process_sample as sampler
import processor.process_sdf as compute_sdf


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

    # Get sdf
    sdf = compute_sdf.compute_sdf(
        mesh,
        points=points,
        surface_points=10000000,
    )
    logging.debug(
        "Mesh had {}/{} interior points".format((sdf < 0).sum(), sdf.shape[0])
    )

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
        "--padding",
        "-p",
        type=float,
        default=0.2,
        help="Extra padding to add when performing uniform sampling. eg 0 = "
        + "uniform sampling is done in unit cube.",
    )
    parser.add_argument(
        "--dim",
        "-d",
        type=int,
        default=256,
        help="Dimension of point samples.",
    )
    args = parser.parse_args()

    process(
        f_in=args.input,
        f_out=args.output,
        padding=args.padding,
        dim=args.dim,
    )
