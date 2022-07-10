import argparse
import json
import os
import logging

import core
import numpy as np
import skimage.measure
import trimesh

import utils_3d
import trimesh
from PIL import Image
from reconstruction.model import LoadModel
from reconstruction.utils.data_prep import volume_to_point_cloud


def voxels2mesh(values, level=0.5, gradient_direction="descent", N=32, padding=0.1):

    # Try to extract an isosurface
    try:
        vertices, faces, _, _ = skimage.measure.marching_cubes(
            values,
            level=level,
            spacing=[(1 + (padding * 2)) / d for d in (N, N, N)],
            gradient_direction=gradient_direction,
        )
    except (ValueError, RuntimeError):
        logging.debug("Isosurface extraction failed")
        return None

    # The x and y channels are flipped in marching cubes
    vertices = np.hstack(
        [
            np.expand_dims(v, axis=1)
            for v in (vertices[:, 1], vertices[:, 0], vertices[:, 2])
        ]
    )

    # Center the shape
    vertices -= 0.5 + padding
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def save_pointcloud(f, vol):
    if vol.dtype != np.bool:
        vol = vol > 0

    pc = volume_to_point_cloud(vol)
    trimesh.PointCloud(vertices=pc).export(f)


def vol_to_point_cloud(vol):
    if vol.dtype != np.bool:
        vol = vol > 0

    return volume_to_point_cloud(vol)


def try_render(m, r):
    """
    Tries to render a mesh, if mesh is none, returns a blank image
    """
    if m is None:
        return (np.ones(r) * 255).astype(np.uint8)
    return utils_3d.render_mesh(m, resolution=r[:2], yrot=160)


def plot_loss(path_loss, path_out):
    import matplotlib.pyplot as plt

    with open(path_loss, "r") as file:
        acc = []
        for line in file:
            if line:
                acc.append(line)
        titles = acc.pop(0).split(",")
        values = np.array([[float(a_) for a_ in a.split(",")] for a in acc])
        plt.plot(values[:, 0], values[:, 1:])
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(path_out)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description=".")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        type=str,
        help="Path to ORGAN-3D experiment.",
    )
    arg_parser.add_argument(
        "--name",
        default="organ-3d",
        type=str,
        help="",
    )
    arg_parser.add_argument(
        "--render_threads",
        default=3,
        type=int,
        help="Number of threads to use for rendering.",
    )
    arg_parser.add_argument(
        "--stop",
        default=None,
        type=int,
        help="Stop inference after x samples.",
    )
    arg_parser.add_argument(
        "--overwrite_meshes",
        action="store_true",
        default=False,
        help="",
    )
    arg_parser.add_argument(
        "--overwrite_evals",
        action="store_true",
        default=False,
        help="",
    )
    arg_parser.add_argument(
        "--overwrite_renders",
        action="store_true",
        default=False,
        help="",
    )
    arg_parser.add_argument(
        "--skip_generate",
        action="store_true",
        default=False,
        help="",
    )
    arg_parser.add_argument(
        "--skip_render",
        action="store_true",
        default=False,
        help="",
    )
    arg_parser.add_argument(
        "--skip_eval",
        action="store_true",
        default=False,
        help="",
    )
    arg_parser.add_argument(
        "--skip_export_eval",
        action="store_true",
        default=False,
        help="",
    )
    core.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    core.configure_logging(args)

    assert os.environ["DATADIR"], "environment variable $DATADIR must be defined"

    # Load 3d-organ experiment specs
    specs_filename = core.find_specs(args.experiment_directory)
    args.experiment_directory = os.path.dirname(specs_filename)
    specs = json.load(open(specs_filename))

    plot_loss(
        os.path.join(args.experiment_directory, "training_log.csv"),
        os.path.join(args.experiment_directory, "training_log.png"),
    )

    # Get relevant data
    file_path_data = specs["DataSource"].replace("$DATADIR", os.environ["DATADIR"])
    test_split_file = specs["TestSplit"].replace("$DATADIR", os.environ["DATADIR"])
    test_split_root = specs["TestSplitRoot"].replace("$DATADIR", os.environ["DATADIR"])

    # Set all the parameters
    experiment_name = args.name
    overwrite_meshes = args.overwrite_meshes
    overwrite_evals = args.overwrite_evals
    overwrite_renders = args.overwrite_renders
    render_threads = args.render_threads
    skip_generate = args.skip_generate

    (
        COMPLETE_INDEX,
        BROKEN_INDEX,
        RESTORATION_INDEX,
        COMPLETE_VOXEL_INDEX,
        RESTORATION_VOXEL_INDEX,
    ) = [0, 1, 2, 3, 4]

    total_outputs = (0, 1, 2)
    composite = [(1, 2)]
    isosurface_level = 0.5
    render_resolution = (200, 200)

    logging.info("Loading the data handler ...")
    sdf_dataset = core.data.SamplesDataset(
        test_split_file,
        root=test_split_root,
    )

    # Create the reconstruction handler
    reconstruction_handler = core.handler.ReconstructionHandler(
        experiment_directory=args.experiment_directory,
        name=experiment_name,
        checkpoint="400",
        signiture=["nan"],
    )
    reconstruct_list = list(range(len(sdf_dataset)))
    if args.stop is not None:
        reconstruct_list = reconstruct_list[: args.stop]

    if not skip_generate:
        # Load model
        logging.info("Using {} network structure".format(
            specs.get("opt", "voxels-usegan")
        ))
        model = LoadModel(
            file_path_data,
            args.experiment_directory,
            opt=specs.get("opt", "voxels-usegan"),
            evaluate_mode=True,
            fracture_data=False,  # Do not fracture the data
            shuffle_data=False,
        )

        # Load data
        model._load_full_test_set()
        assert len(sdf_dataset) == len(model.test_loader.data["labels"]), \
            "size mismatch, {} vs {}".format(len(sdf_dataset), len(model.test_loader.data["labels"]))

        # Just extract the ones we want to reconstruct
        fractured_voxels, _, labels = model.full_test_data
        fractured_voxels = fractured_voxels[reconstruct_list, ...]

        # Predict for 2 iterations
        predicted_voxels1 = model.predict(fractured_voxels, labels)
        predicted_voxels2 = model.predict(predicted_voxels1, labels)

        # Run the primary loop
        logging.info("Processing {} samples".format(len(reconstruct_list)))

        for idx, ridx in enumerate(reconstruct_list):

            # Get the predicted and fractrured voxels
            pred_vox = predicted_voxels2[idx]
            frac_vox = fractured_voxels[idx]

            # Compute restoration as predicted (complete) minus the fractured
            rest_vox = pred_vox - frac_vox

            # Convert to floating point for isosurface extraction
            pred_vox = pred_vox.astype(float)
            rest_vox = rest_vox.astype(float)

            # Set the meshes
            if (
                not os.path.exists(
                    reconstruction_handler.path_mesh(ridx, COMPLETE_INDEX)
                )
                or not os.path.exists(
                    reconstruction_handler.path_mesh(ridx, COMPLETE_VOXEL_INDEX)
                )
                or overwrite_meshes
            ):
                mesh = voxels2mesh(pred_vox, level=isosurface_level)
                if mesh is not None:
                    reconstruction_handler.set_mesh(
                        mesh,
                        ridx,
                        COMPLETE_INDEX,
                        save=True,
                    )
                    reconstruction_handler.set_mesh(
                        trimesh.voxel.VoxelGrid(
                            pred_vox,
                        ).as_boxes(),
                        ridx,
                        COMPLETE_VOXEL_INDEX,
                        save=True,
                    )
            if (
                not os.path.exists(
                    reconstruction_handler.path_mesh(ridx, RESTORATION_INDEX)
                )
                or not os.path.exists(
                    reconstruction_handler.path_mesh(ridx, RESTORATION_VOXEL_INDEX)
                )
                or overwrite_meshes
            ):
                mesh = voxels2mesh(rest_vox, level=isosurface_level)
                if mesh is not None:
                    reconstruction_handler.set_mesh(
                        mesh,
                        ridx,
                        RESTORATION_INDEX,
                        save=True,
                    )
                    reconstruction_handler.set_mesh(
                        trimesh.voxel.VoxelGrid(
                            rest_vox,
                        ).as_boxes(),
                        ridx,
                        RESTORATION_VOXEL_INDEX,
                        save=True,
                    )

    # Spins up a multiprocessed renderer
    logging.info("Rendering results ...")
    core.handler.render_engine(
        data_handler=sdf_dataset,
        reconstruct_list=reconstruct_list,
        reconstruction_handler=reconstruction_handler,
        outputs=total_outputs,
        num_renders=3,
        resolution=render_resolution,
        composite=composite,
        overwrite=overwrite_renders,
        threads=render_threads,
    )

    logging.info("Building summary render")
    img = core.vis.image_results(
        data_handler=sdf_dataset,
        reconstruct_list=reconstruct_list,
        reconstruction_handler=reconstruction_handler,
        outputs=total_outputs,
        num_renders=3,
        resolution=render_resolution,
        composite=composite,
        knit_handlers=[],
    )
    path = os.path.join(
        reconstruction_handler.path_reconstruction(), "summary_img_{}.jpg"
    )
    logging.info("Saving summary render to: {}".format(path))
    core.vis.save_image_block(img, path)
