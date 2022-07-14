import argparse
import json
import logging
import os
import random
import math
import multiprocessing

import torch
import numpy as np
from collections import defaultdict

import core

STATUS_INDICATOR = None
STATUS_COUNTER = 0


def reconstruct(
    decoder,
    num_iterations,
    latent_size,
    break_latent_size,
    test_sdf,
    stat,
    lambda_ner,
    lambda_prox,
    clamp_dist,
    num_samples=30000,
    lr=5e-4,
    l2reg=False,
    code_reg_lambda=1e-4,
    iter_path=None,
):

    def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
    ):
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)

    if type(stat) == type(0.1):
        latent = torch.ones(1, latent_size).normal_(mean=0, std=stat).cuda()
        break_latent = torch.ones(1, break_latent_size).normal_(mean=0, std=stat).cuda()
    else:
        latent = torch.normal(stat[0].detach(), stat[1].detach()).cuda()
        break_latent = torch.normal(stat[0].detach(), stat[1].detach()).cuda()

    latent.requires_grad = True
    break_latent.requires_grad = True

    optimizer = torch.optim.Adam([latent, break_latent], lr=lr)

    loss_bce = torch.nn.BCELoss()
    loss_bce_logits = torch.nn.BCEWithLogitsLoss()
    zeros = torch.Tensor([0]).cuda()
    zeros.requires_grad = False
    ones = torch.Tensor([1]).cuda()
    ones.requires_grad = False
    loss_dict = defaultdict(lambda : [])

    prox_loss = torch.Tensor([0])
    ner_loss = torch.Tensor([0])
    for e in range(num_iterations):

        decoder.eval()

        pts, data_sdf = test_sdf
        pts, sdf_gt = core.data.select_samples(
            pts, data_sdf, num_samples, uniform_ratio=0.2
        )

        # Visualize
        # core.vis.plot_samples((pts, sdf_gt), n_plots=16).savefig("test.png")

        # Convert to tensors
        xyz = torch.from_numpy(pts).type(torch.float).cuda()
        sdf_gt = torch.from_numpy(sdf_gt).type(torch.float).cuda()


        adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)

        optimizer.zero_grad()

        latent_inputs = latent.expand(num_samples, -1)
        break_latent_inputs = break_latent.expand(num_samples, -1)
        inputs = torch.cat([latent_inputs, break_latent_inputs, xyz], dim=1).cuda()

        c_x, b_x, r_x, _ = decoder(inputs)

        # TODO: why is this needed?
        if e == 0:
            b_x = decoder(inputs, use_net=1)


        data_loss = loss_bce(b_x, sdf_gt) / num_samples
        loss = data_loss

        # Compute non-collapse loss
        if lambda_ner != 0.0:
            mean_diff = r_x.mean(axis=0)
            ner_loss = lambda_ner * loss_bce(
                mean_diff,
                ones,
            )

            loss = loss + ner_loss

        if lambda_prox != 0.0:
            mean_diff = (torch.sigmoid(c_x) - sdf_gt).pow(2).mean(axis=0)
            prox_loss = lambda_prox * loss_bce(
                mean_diff,
                zeros,
            )

            loss = loss + prox_loss

        # Regularization loss
        if l2reg:
            reg_loss = torch.mean(latent.pow(2)) 
            break_reg_loss = torch.mean(break_latent.pow(2)) 
            reg_loss = (reg_loss + break_reg_loss)* code_reg_lambda
            loss = loss + reg_loss

        if e % 10 == 0:
            loss_dict["epoch"].append(e)
            loss_dict["loss"].append(loss.item())
            loss_dict["data_loss"].append(data_loss.item())
            loss_dict["ner_loss"].append(ner_loss.item())
            loss_dict["prox_loss"].append(prox_loss.item())
            loss_dict["reg_loss"].append(reg_loss.item())
            loss_dict["mag"].append(torch.norm(latent).item())
            loss_dict["h_mag"].append(torch.norm(break_latent).item())
            logging.debug(
                "epoch: {:4d} | loss: {:1.5e} data_loss: {:1.5e} ner_loss: {:1.5e} prox_loss: {:1.5e} reg_loss: {:1.5e} rperc: {:1.2f}".format(
                    loss_dict["epoch"][-1],
                    loss_dict["loss"][-1],
                    loss_dict["data_loss"][-1],
                    loss_dict["ner_loss"][-1],
                    loss_dict["prox_loss"][-1],
                    loss_dict["reg_loss"][-1],
                    r_x.sum().item() / num_samples,
                )
            )

        loss.backward()
        optimizer.step()

    return dict(loss_dict), torch.cat([latent, break_latent], dim=1)


def callback():
    global STATUS_INDICATOR
    global STATUS_COUNTER
    try:
        STATUS_INDICATOR.increment()
    except AttributeError:
        print("Completed: {}".format(STATUS_COUNTER))
        STATUS_COUNTER += 1


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to reconstruct a shape given SDF "
        + "samples."
    )
    arg_parser.add_argument(
        "--input_mesh",
        required=True,
        help="gt fractured object, used only for rendering.",
    )
    arg_parser.add_argument(
        "--input_points",
        required=True,
        help="Sample points, specified as a .npz file.",
    )
    arg_parser.add_argument(
        "--input_sdf",
        required=True,
        help="Sample sdf, specified as a .npz file.",
    )
    arg_parser.add_argument(
        "--output_meshes",
        required=True,
        help="Path template to save the meshes to.",
    )
    arg_parser.add_argument(
        "--output_code",
        required=True,
        help="Path template to save the code to.",
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--threads",
        default=6,
        type=int,
        help="Number of threads to use for reconstruction.",
    )
    arg_parser.add_argument(
        "--render_threads",
        default=6,
        type=int,
        help="Number of threads to use for rendering.",
    )
    arg_parser.add_argument(
        "--num_iters",
        default=800,
        type=int,
        help="The number of iterations of latent code optimization to perform.",
    )
    arg_parser.add_argument(
        "--num_samples",
        default=8000,
        type=int,
        help="Number of samples to use.",
    )
    arg_parser.add_argument(
        "--stop",
        default=None,
        type=int,
        help="Stop inference after x samples.",
    )
    arg_parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="Randomized seed.",
    )
    arg_parser.add_argument(
        "--out_of_order",
        default=False,
        action="store_true",
        help="Randomize the order of inference.",
    )
    arg_parser.add_argument(
        "--name",
        default="ours_",
        type=str,
        help="",
    )
    arg_parser.add_argument(
        "--lambda_reg",
        default=1e-4,
        type=float,
        help="Regularization lambda value.",
    )
    arg_parser.add_argument(
        "--learning_rate",
        default=5e-3,
        type=float,
        help="Regularization lambda value.",
    )
    arg_parser.add_argument(
        "--lambda_ner",
        default=1e-3,
        type=float,
        help="Non-collapse alpha value.",
    )
    arg_parser.add_argument(
        "--lambda_prox",
        default=1e-3,
        type=float,
        help="Non-collapse alpha value.",
    )
    arg_parser.add_argument(
        "--uniform_ratio",
        default=None,
        type=float,
        help="Uniform Ratio.",
    )
    arg_parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite code.",
    )
    arg_parser.add_argument(
        "--gif",
        action="store_true",
        default=False,
        help="",
    )
    core.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    core.configure_logging(args)

    # assert os.environ["DATADIR"], "environment variable $DATADIR must be defined"

    specs_filename = core.find_specs(args.experiment_directory)
    specs = json.load(open(specs_filename))
    args.experiment_directory = os.path.dirname(specs_filename)

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    latent_size = specs["CodeLength"]

    lambda_ner = args.lambda_ner
    lambda_prox = args.lambda_prox
    logging.info("Using lambda_ner: {}".format(lambda_ner))
    logging.info("Using lambda_prox: {}".format(lambda_prox))

    num_samples = args.num_samples
    num_iterations = args.num_iters
    code_reg_lambda = args.lambda_reg
    lr = args.learning_rate
    name = args.name
    uniform_ratio = args.uniform_ratio
    if uniform_ratio is None:
        uniform_ratio = specs["UniformRatio"]

    test_split_file = specs["TestSplit"]
    break_latent_size = specs["BreakCodeLength"]

    network_outputs = (0, 1, 2, 3)
    total_outputs = (0, 1, 2, 3)
    eval_pairs = [(0, 0), (1, 1), (2, 2)]  # ORDER MATTERS
    metrics_version = "metrics.v4.npy"
    metrics = [
        "chamfer",
        # "union_score",
        # "intersection_score",
        # "non_protrusion_score",
        # "fractured_protrusion_score",
        # "component_error",
        # "connected_artifacts_score",
        "connected_artifacts_score2",
        # "complete_sameness",
        # "nested_binner",
        # "truth_tabler",
        # "iou",
        # "chamfer_complete",
        # "normal_consistency",
        # "chamfer_join",
        "join_dissimilarity",
    ]
    composite = [(1, 2), (1, 3)]
    render_resolution = (200, 200)
    do_code_regularization = True
    isosurface_level = 0.5
    use_sigmoid = True

    assert specs["NetworkArch"] in [
        "decoder_z_lb_occ_leaky",
    ], "wrong arch"

    network_kwargs = dict(
        decoder_kwargs=dict(
            latent_size=latent_size,
            tool_latent_size=break_latent_size,
            num_dims=3,
            do_code_regularization=do_code_regularization,
            **specs["NetworkSpecs"],
            **specs["SubnetSpecs"],
        ),
        decoder_constructor=arch.Decoder,
        experiment_directory=args.experiment_directory,
        checkpoint=args.checkpoint,
    )
    reconstruction_kwargs = dict(
        num_iterations=num_iterations,
        latent_size=latent_size,
        break_latent_size=break_latent_size,
        lambda_ner=lambda_ner,
        lambda_prox=lambda_prox,
        stat=0.01,  # [emp_mean,emp_var],
        clamp_dist=None,
        num_samples=num_samples,
        lr=lr,
        l2reg=do_code_regularization,
        code_reg_lambda=code_reg_lambda,
    )
    mesh_kwargs = dict(
        dims=[256, 256, 256],
        level=isosurface_level,
        gradient_direction="descent",
        batch_size=2 ** 14,
    )

    assert os.path.splitext(args.output_code)[-1] == ".pth"
    assert os.path.splitext(args.output_meshes)[-1] == ".obj"

    # Load the data
    xyz = np.load(args.input_points)["xyz"]
    sdf = np.load(args.input_sdf)["sdf"]
    sdf = core.sdf_to_occ(np.expand_dims(sdf, axis=1))
    assert len(xyz.shape) == 2 and len(sdf.shape) == 2

    # Load the network
    decoder = core.load_network(**network_kwargs)
    decoder.eval()

    # Reconstruct the code
    if not os.path.exists(args.output_code) or args.overwrite:
        losses, code = reconstruct(
            test_sdf=[xyz, sdf],
            decoder=decoder,
            **reconstruction_kwargs,
        )
        core.saver(args.output_code, code)
    else:
        code = core.loader(args.output_code)

    mesh_path_list = [
        os.path.splitext(args.output_meshes)[0]
        + str(shape_idx)
        + os.path.splitext(args.output_meshes)[-1]
        for shape_idx in range(3)
    ]

    # Reconstruct the meshes
    mesh_list = []
    for shape_idx, path in enumerate(mesh_path_list):
        if not os.path.exists(path) or args.overwrite:
            sigmoid = True
            if shape_idx in [1, 2]:
                sigmoid = False
            try:
                mesh = core.reconstruct.create_mesh(
                    vec=code,
                    decoder=decoder,
                    use_net=shape_idx,
                    sigmoid=sigmoid,
                    **mesh_kwargs,
                )
                mesh = core.colorize_mesh_from_index_auto(mesh, shape_idx)
                mesh.export(path)
            except core.errors.IsosurfaceExtractionError:
                logging.info(
                    "Isosurface extraction error for shape: {}".format(shape_idx)
                )
                mesh = None
        else:
            mesh = core.loader(path)
        mesh_list.append(mesh)

    # Create a render of the the restoration object with gt fractured mesh

    DURATION = 10  # in seconds
    FRAME_RATE = 2
    RESOLUTION = (600, 600)
    ZOOM = 2.0
    num_renders = DURATION * FRAME_RATE

    if mesh_list[2] is not None:
        gt_mesh = core.loader(args.input_mesh)
        gt_mesh.fix_normals()
        if args.gif:
            core.saver(
                f_out=os.path.splitext(args.output_meshes)[0] + "_f_r.gif",
                data=core.create_gif_rot(
                    [
                        core.colorize_mesh_from_index_auto(gt_mesh, 1),
                        core.colorize_mesh_from_index_auto(mesh_list[2], 2),
                    ],
                    num_renders=num_renders,
                    resolution=RESOLUTION,
                    zoom=ZOOM,
                    bg_color=0,
                ),
                loop=0,
                duration=(1 / num_renders) * DURATION * 1000,
            )
        else:
            core.saver(
                f_out=os.path.splitext(args.output_meshes)[0] + "_f_r.png",
                data=core.render_mesh(
                    [
                        core.colorize_mesh_from_index_auto(gt_mesh, 1),
                        core.colorize_mesh_from_index_auto(mesh_list[2], 2),
                    ],
                    resolution=RESOLUTION,
                    ztrans=ZOOM,
                    bg_color=0,
                ),
            )
