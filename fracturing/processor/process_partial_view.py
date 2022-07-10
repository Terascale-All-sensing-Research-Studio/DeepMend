import os, argparse
import logging

# Set the available threads to (1/3)
os.environ["OMP_NUM_THREADS"] = str(int(os.cpu_count() * (1 / 3)))

import trimesh
import numpy as np
from pykdtree.kdtree import KDTree

from processor.process_sample import sample_points
from processor.process_sdf import compute_sdf
import processor.errors as errors
import processor.logger as logger


def mesh_partial_view(
    broken_mesh, restoration_mesh, pts, values=None, thresh=0.02, use_occ=True
):
    """
    Return a mask that identifies points not in the fracture region.
    This simulates high accuracy fracture removal on
    the broken shape. Uses knn search to determine which points to keep.
    This corresponds to the optimal partial view of the broken shape.

    Args:
        broken_mesh: broken trimesh object.
        restoration_mesh: restoration trimesh object.
        pts: Points in n dimensional space
        values: Value at each point. The number of columns gives the number
            of shapes (three shapes are required).
        thresh: Threshold at which a point in the restoration shape is
            considered part of the fracture region.
    """

    # Sample 1m points
    b_pts = broken_mesh.sample(1000000)
    r_pts = restoration_mesh.sample(1000000)

    # Extract points very close to the fracture
    dist, _ = KDTree(r_pts).query(b_pts, k=1)

    # Indicies of the broken object that are very close to the fracture
    b_fracture_pts = b_pts[(dist < thresh), :]
    b_not_fracture_pts = b_pts[np.logical_not(dist < thresh), :]

    # Reorganize the vertices such that the fracture points come last
    b_pts = np.vstack((b_not_fracture_pts, b_fracture_pts))

    # Find points that are closer to the fracture than the rest of the broken shape
    _, inds = KDTree(b_pts).query(pts, k=1)
    pts_not_fracture_mask = inds < b_not_fracture_pts.shape[0]

    # All of the points inside of the broken object should be included
    if values is not None:
        if use_occ:
            pts_not_fracture_mask[values.squeeze() == 1.0] = True
        else:
            pts_not_fracture_mask[values.squeeze() < 0.0] = True
    return pts_not_fracture_mask


def process(
    f_in,
    f_rest,
    f_out,
    n_points=500000,
    uniform_ratio=0.5,
    padding=0.2,
    sigma=0.01,
    overwrite=False
):

    # Load meshes
    broken_mesh = trimesh.load(f_in)
    if not broken_mesh.is_watertight:
        raise errors.MeshNotClosedError
    restoration_mesh = trimesh.load(f_rest)
    if not restoration_mesh.is_watertight:
        raise errors.MeshNotClosedError

    # Get sample points
    logging.debug("Sampling points on broken")
    pts = sample_points(
        mesh=broken_mesh,
        n_points=n_points,
        uniform_ratio=uniform_ratio,
        padding=padding,
        sigma=sigma,
    )

    # Compute sdf
    logging.debug("Computing sdf for sample points")
    sdf = compute_sdf(mesh=broken_mesh, points=pts)
    sdf = np.expand_dims(sdf, axis=1)

    # Compute partial view
    logging.debug("Computing partial view")
    mask = mesh_partial_view(
        broken_mesh, restoration_mesh, pts, sdf, thresh=0.01, use_occ=False
    )

    # Save
    if overwrite or not os.path.exists(f_out):
        logging.debug("Saving to: {}".format(f_out))
        np.savez(
            f_out,
            xyz=pts.astype(np.float16),
            sdf=sdf.astype(np.float16),
            mask=mask.astype(bool),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computes the sdf values for samples points on and "
        + "around an object. Accepts the arguments common for sampling. Also "
        + "computes a mask corresponding to points NOT in the fracture region."
    )
    parser.add_argument(dest="input", type=str, help="Path to the input file.")
    parser.add_argument(dest="rest", type=str, help="Path to the restoration file.")
    parser.add_argument(dest="output", type=str, help="Path to the output file.")
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
    logger.add_logger_args(parser)
    args = parser.parse_args()
    logger.configure_logging(args)

    process(
        f_in=args.input,
        f_rest=args.rest,
        f_out=args.output,
        n_points=args.n_points,
        uniform_ratio=args.uniform,
        padding=args.padding,
        sigma=args.sigma,
    )
