import argparse, os
import logging

import pymesh
import trimesh
import numpy as np
from pykdtree.kdtree import KDTree

import processor.errors as errors
import processor.logger as logger


def paint_mesh(mesh_from, mesh_to, vertex_inds):
    """Transfer vertex colors from one mesh to another"""
    # Vertices to transfer color to
    vertices = mesh_to.vertices[vertex_inds, :]

    # Find their nearest neighbor on source mesh
    _, v_idx = KDTree(mesh_from.vertices).query(vertices)

    # Standin color is white, opaque
    mesh_to.visual.vertex_colors = (
        np.ones((mesh_to.vertices.shape[0], 4)).astype(np.uint8) * 255
    )

    # Transfer the colors
    mesh_to.visual.vertex_colors[vertex_inds, :] = mesh_from.visual.vertex_colors[
        v_idx, :
    ]


def intersect_mesh(a, b, sig=5):
    """get mask of vertices in a occurring in both a and b, to apply to a"""
    av = [frozenset(np.round(v, sig)) for v in a]
    bv = set([frozenset(np.round(v, sig)) for v in b])
    return np.asarray(list(map(lambda v: v in bv, av)))


def compute_break_percent(gt_restoration, gt_complete, method="volume"):
    """Compute the percent of an object removed by a break"""

    def num_intersecting(a, b, thresh=1e-8):
        """Return number of intersecting vertices"""
        d, _ = KDTree(a).query(b)
        return (d < thresh).sum()

    if method == "volume":
        return gt_restoration.volume / gt_complete.volume
    elif method == "surface_area":
        return (
            num_intersecting(gt_complete.vertices, gt_restoration.vertices)
            / gt_complete.vertices.shape[0]
        )
    else:
        raise RuntimeError("Unknown method {}".format(method))


def break_mesh(
    mesh,
    offset=0.0,
    rand_translation=0.1,
    noise=0.005,
    replicator=None,
    return_tool=False,
):
    """
    Break an object and return the broken and restoration objects.
    Returns a dictionary that can be used to replicate the break exactly.
    """

    if replicator is None:
        replicator = {}

    tool_type = replicator.setdefault("tool_type", np.random.randint(1, high=5))
    if tool_type == 1:
        tool = pymesh.generate_box_mesh(
            box_min=[-0.5, -0.5, -0.5], box_max=[0.5, 0.5, 0.5], subdiv_order=6
        )
    else:
        if tool_type == 2:
            tool = pymesh.generate_icosphere(0.5, [0.0, 0.0, 0.0], refinement_order=0)
        elif tool_type == 3:
            tool = pymesh.generate_icosphere(0.5, [0.0, 0.0, 0.0], refinement_order=1)
        elif tool_type == 4:
            tool = pymesh.generate_icosphere(0.5, [0.0, 0.0, 0.0], refinement_order=2)

        # Disjoint the vertices so that the icosphere isn't regular
        random_disjoint = replicator.setdefault(
            "random_disjoint", np.random.random(tool.vertices.shape)
        )
        tool = pymesh.form_mesh(
            tool.vertices + (random_disjoint * (0.1) - (0.1 / 2)), tool.faces
        )

        # Subdivide the mesh
        tool, __ = pymesh.split_long_edges(tool, noise * 5)
    vertices = tool.vertices

    # Scale to half size
    # vertices = vertices * 0.25
    vertices = vertices * 0.7

    # Offset the tool so that the break is roughly in the center
    set_offset = replicator.setdefault("set_offset", np.array([-0.5 + offset, 0, 0]))
    vertices = vertices + set_offset

    # Add random noise to simulate fracture geometry
    noise = np.asarray([noise, noise, noise])
    random_noise = replicator.setdefault(
        "random_noise", np.random.random(vertices.shape)
    )
    vertices = vertices + (random_noise * (noise) - (noise / 2))

    # Add a random rotation
    # http://planning.cs.uiuc.edu/node198.html
    y_range = 20
    z_range = 90
    _, rand_y_rot, rand_z_rot = replicator.setdefault(
        "random_rotation", 
        [
            None, 
            np.radians((np.random.random() * y_range) - (y_range/2)), 
            np.radians((np.random.random() * z_range) - (z_range/2)),
        ]
    )
    mat = np.dot(
        trimesh.transformations.rotation_matrix(rand_y_rot, [0, 1, 0]),
        trimesh.transformations.rotation_matrix(rand_z_rot, [0, 0, 1]),
    )

    # u, v, w = replicator.setdefault(
    #     "random_rotation", 
    #     np.random.random(3)
    # )
    # q = [
    #     np.sqrt(1 - u) * np.sin(2 * np.pi * v),
    #     np.sqrt(1 - u) * np.cos(2 * np.pi * v),
    #     np.sqrt(u) * np.sin(2 * np.pi * w),
    #     np.sqrt(u) * np.cos(2 * np.pi * v),
    # ]
    # mat = pymesh.Quaternion(q).to_matrix()
    vertices = np.dot(mat[:3, :3], vertices.T).T

    # Add a small random translation
    random_translation = replicator.setdefault(
        "random_translation", np.random.random(3)
    )
    vertices += random_translation * (rand_translation) - (rand_translation / 2.0)

    # Add a warp
    warp = lambda vs: np.asarray([(v ** 3) for v in vs])
    vertices += np.apply_along_axis(warp, 1, vertices)

    # Now make sure it doesnt hit the handle
    mug_width = np.max(mesh.vertices[:, 2]) * 2
    mug_xmax = np.max(mesh.vertices[:, 0])
    tool_xmax = np.max(vertices[:, 0])
    target_xmax = mug_xmax - mug_width
    xoffset = 0
    if tool_xmax > target_xmax:
        xoffset = target_xmax - tool_xmax  
        logging.debug("Touching a non-handle region, adjusting tool by {}".format(xoffset))
    vertices[:, 0] = vertices[:, 0] + xoffset

    # Break
    tool = pymesh.form_mesh(vertices, tool.faces)
    broken = pymesh.boolean(mesh, tool, "difference")
    restoration = pymesh.boolean(
        mesh, pymesh.form_mesh(vertices, tool.faces), "intersection"
    )

    if return_tool:
        return broken, restoration, replicator, tool
    return broken, restoration, replicator


def process(
    f_in,
    f_out,
    f_restoration=False,
    f_tool=False,
    export_color=False,
    export_normals=False,
    validate=True,
    save_meta=False,
    max_break=0.5,
    min_break=0.3,
    num_components=1,
    max_overall_retries=5,
    max_single_retries=5,
    break_method="surface-area",
    overwrite=False,
):

    assert max_break > min_break
    assert break_method in ["surface-area", "volume", "combined"]

    # Break parameters
    offset = 0.2 # offset into the mesh. A higher value will remove more
    refinement_offset = 0.05
    refinement_decay = 0.95
    noise=0.005

    # Load the mesh
    tri_mesh_in = trimesh.load(f_in)
    tri_mesh_in.fix_normals()
    mesh_in = pymesh.form_mesh(tri_mesh_in.vertices, tri_mesh_in.faces)

    # Make sure mesh is closed
    if (not mesh_in.is_manifold()) or (not mesh_in.is_closed()):
        raise errors.MeshNotClosedError

    cur_retry = 0
    while cur_retry < max_overall_retries:
        logging.debug("== Restarting fracture ... ==")

        # Break for the first time
        mesh_out, rmesh_out, replicator, mesh_tool = break_mesh(
            mesh_in, replicator=None, offset=offset, return_tool=True, noise=noise,
        )

        # Check to make sure enough of the object was removed
        for itr in range(max_single_retries):
            amount_removed_vol = compute_break_percent(
                rmesh_out, mesh_in, method="volume"
            )
            amount_removed_sa = compute_break_percent(
                rmesh_out, mesh_in, method="surface_area"
            )
            mesh_num_components = mesh_out.num_surface_components
            rmesh_num_components = rmesh_out.num_surface_components

            # Print debug information
            logging.debug("Removed {}%% volume".format(round(amount_removed_vol, 3)))
            logging.debug(
                "Removed {}%% surface_area".format(round(amount_removed_sa, 3))
            )
            logging.debug("Broken has {} components".format(mesh_num_components))
            logging.debug("Restoration has {} components".format(rmesh_num_components))
            logging.debug("Replicator tool_type: {}".format(replicator["tool_type"]))
            logging.debug(
                "Replicator random_translation: {}".format(
                    replicator["random_translation"]
                )
            )
            logging.debug(
                "Replicator random_rotation: {}".format(replicator["random_rotation"])
            )
            logging.debug("Replicator set_offset: {}".format(replicator["set_offset"]))
            
            # Force the volume condition to be satisfied
            if break_method == "surface-area":
                amount_removed_vol = min_break

            # Force the surface area condition to be satisfied
            elif break_method == "volume":
                amount_removed_sa = min_break

            # Adjust the location of the tool
            if (
                (amount_removed_vol < min_break)
                or (amount_removed_sa < min_break)
                or (mesh_num_components > num_components)
                or (rmesh_num_components > num_components)
            ):
                logging.debug("Moving tool backwards")
                replicator["set_offset"][0] -= refinement_offset * (
                    refinement_decay ** itr
                )
            elif (
                # (amount_removed_vol > max_break)
                # or (amount_removed_sa > max_break)
                (mesh_num_components < num_components)
                or (rmesh_num_components < num_components)
            ):
                logging.debug("Moving tool forwards")
                replicator["set_offset"][0] += refinement_offset * (
                    refinement_decay ** itr
                )
            else:
                break

            # Retry the break
            mesh_out, rmesh_out, replicator, mesh_tool = break_mesh(
                mesh_in, replicator=replicator, offset=offset, return_tool=True, noise=noise,
            )

        else:
            logging.debug(
                "Failed {} times, re-randomizing tool".format(max_single_retries)
            )
            cur_retry += 1
            continue

        # Perform output validation
        if validate:
            # We removed all of the vertices, or no vertices
            if (len(mesh_out.vertices) == 0) or (
                len(mesh_out.vertices) == len(mesh_in.vertices)
            ):
                cur_retry += 1
                logging.debug("Mesh validation failed, all or no vertices removed")
                continue

            # This shouldn't happen
            elif (
                (not mesh_out.is_manifold())
                or (not mesh_out.is_closed())
                or (not rmesh_out.is_manifold())
                or (not rmesh_out.is_closed())
            ):
                cur_retry += 1
                logging.debug("Mesh validation failed, result is not waterproof")
                continue

        break

    # If we've completed the while loop then this mesh cant be broken
    else:
        raise errors.MeshBreakMaxRetriesError

    logging.debug(
        "Successfully removed {}%% volume".format(round(amount_removed_vol, 3))
    )
    logging.debug(
        "Successfully removed {}%% surface_area".format(round(amount_removed_sa, 3))
    )
    logging.debug("Broken has {} components".format(mesh_num_components))
    logging.debug("Restoration has {} components".format(rmesh_num_components))
    logging.debug("Broken has {} vertices".format(mesh_out.vertices.shape[0]))
    logging.debug("Restoration has {} vertices".format(rmesh_out.vertices.shape[0]))

    # Save metadata
    if save_meta:
        fracture_inds = np.logical_not(
            intersect_mesh(mesh_out.vertices, mesh_in.vertices)
        )
        f_meta = os.path.splitext(f_out)[0] + ".npz"
        if overwrite or not os.path.exists(f_meta):
            logging.debug("Saving metadata to: {}".format(f_meta))
            np.savez_compressed(
                f_meta,
                fracture_vertices=mesh_out.vertices[fracture_inds, :],
                **replicator,
            )
        else:
            return

    # Save the broken object
    tri_mesh_out_b = trimesh.Trimesh(vertices=mesh_out.vertices, faces=mesh_out.faces)
    if export_color:
        paint_mesh(
            tri_mesh_in,
            tri_mesh_out_b,
            intersect_mesh(tri_mesh_out_b.vertices, tri_mesh_in.vertices),
        )
    if export_normals:
        tri_mesh_out_b.vertex_normals
    if overwrite or not os.path.exists(f_out):
        logging.debug("Saving broken to: {}".format(f_out))
        tri_mesh_out_b.export(f_out)
    else:
        return

    # Save the restoration object
    if f_restoration:
        tri_mesh_out_r = trimesh.Trimesh(
            vertices=rmesh_out.vertices, faces=rmesh_out.faces
        )
        if export_color:
            paint_mesh(
                tri_mesh_in,
                tri_mesh_out_r,
                intersect_mesh(tri_mesh_out_r.vertices, tri_mesh_in.vertices),
            )
        if export_normals:
            tri_mesh_out_r.vertex_normals
        
        if overwrite or not os.path.exists(f_restoration):
            logging.debug("Saving restoration to: {}".format(f_restoration))
            tri_mesh_out_r.export(f_restoration)
        else:
            return

    # Save the restoration object
    if f_tool:
        tri_mesh_out_tool = trimesh.Trimesh(
            vertices=mesh_tool.vertices, faces=mesh_tool.faces
        )
        if export_normals:
            tri_mesh_out_tool.vertex_normals

        if overwrite or not os.path.exists(f_tool):
            logging.debug("Saving tool to: {}".format(f_tool))
            tri_mesh_out_tool.export(f_tool)
        else:
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Breaks an object")
    parser.add_argument(dest="input", type=str, help="Path to the input file.")
    parser.add_argument(dest="output", type=str, help="Path to the output file.")
    parser.add_argument(
        "--restoration",
        "-r",
        type=str,
        default=False,
        help="Optionally output restoration file. Will be appended with break "
        + "number.",
    )
    parser.add_argument(
        "--tool",
        type=str,
        default=False,
        help="Optionally output tool file. Will be appended with break " + "number.",
    )
    parser.add_argument(
        "--skip_validate",
        "-v",
        action="store_false",
        help="If passed will skip checking if object is watertight.",
    )
    parser.add_argument(
        "--meta",
        "-m",
        action="store_true",
        help="If passed will store the fracture vertices in a npz file.",
    )
    parser.add_argument(
        "--max_break",
        type=float,
        default=1.0,
        help="Max amount of the object to remove (by volume).",
    )
    parser.add_argument(
        "--min_break",
        type=float,
        default=0.0,
        help="Min amount of the object to remove (by volume).",
    )
    logger.add_logger_args(parser)
    args = parser.parse_args()
    logger.configure_logging(args)

    val = process(
        f_in=args.input,
        f_out=args.output,
        f_restoration=args.restoration,
        f_tool=args.tool,
        validate=args.skip_validate,
        save_meta=args.meta,
        max_break=args.max_break,
        min_break=args.min_break,
    )
