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
    loss_version=None,
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
    zeros_all = torch.ones((num_samples, 1)).cuda()
    zeros.requires_grad = False
    zeros_all.requires_grad = False
    ones = torch.Tensor([1]).cuda()
    ones_all = torch.ones((num_samples, 1)).cuda()
    ones.requires_grad = False
    ones_all.requires_grad = False
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
            if loss_version == "inside_nzr" or loss_version == "inside_both":
                ner_loss = lambda_ner * loss_bce(
                    r_x,
                    ones_all,
                )
            elif loss_version is None or loss_version == "inside_prox2":
                mean_diff = r_x.mean(axis=0)
                ner_loss = lambda_ner * loss_bce(
                    mean_diff,
                    ones,
                )
            else:
                raise RuntimeError()

            loss = loss + ner_loss

        if lambda_prox != 0.0:
            if loss_version == "inside_prox2" or loss_version == "inside_both":
                prox_loss = lambda_prox * loss_bce(
                    torch.sigmoid(c_x),
                    sdf_gt,
                )
            elif loss_version is None or loss_version == "inside_nzr":
                mean_diff = (torch.sigmoid(c_x) - sdf_gt).pow(2).mean(axis=0)
                prox_loss = lambda_prox * loss_bce(
                    mean_diff,
                    zeros,
                )
            else:
                raise RuntimeError()

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
        "--loss_version",
        default=None,
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
        "--overwrite_codes",
        action="store_true",
        default=False,
        help="",
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
        "--save_iter",
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
    arg_parser.add_argument(
        "--mesh_only",
        action="store_true",
        default=False,
        help="",
    )
    arg_parser.add_argument(
        "--split",
        default=None,
        type=int,
        help="",
    )
    core.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    core.configure_logging(args)

    assert os.environ["DATADIR"], "environment variable $DATADIR must be defined"

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
    render_threads = args.render_threads
    overwrite_codes = args.overwrite_codes
    overwrite_meshes = args.overwrite_meshes
    overwrite_evals = args.overwrite_evals
    overwrite_renders = args.overwrite_renders
    threads = args.threads
    save_iter = args.save_iter
    uniform_ratio = args.uniform_ratio
    mesh_only = args.mesh_only
    if uniform_ratio is None:
        uniform_ratio = specs["UniformRatio"]

    test_split_file = specs["TestSplit"]
    break_latent_size = specs["BreakCodeLength"]

    network_outputs = (0, 1, 2, 3)
    total_outputs = (0, 1, 2, 3)
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
        loss_version=args.loss_version,
    )
    mesh_kwargs = dict(
        dims=[256, 256, 256],
        level=isosurface_level,
        gradient_direction="descent",
        batch_size=2 ** 14,
    )

    # Get the data directory from environment variable
    test_split_file = test_split_file.replace("$DATADIR", os.environ["DATADIR"])
    data_source = specs["DataSource"].replace("$DATADIR", os.environ["DATADIR"])

    # Create and load the dataset
    reconstruction_handler = core.handler.ReconstructionHandler(
        experiment_directory=args.experiment_directory,
        dims=[256, 256, 256],
        name=name,
        checkpoint=args.checkpoint,
        overwrite=False,
        use_occ=specs["UseOccupancy"],
        signiture=[
            num_samples,
            num_iterations,
            lr,
            code_reg_lambda,
            lambda_ner,
            lambda_prox,
        ],
    )
    sdf_dataset = core.data.SamplesDataset(
        test_split_file,
        subsample=num_samples,
        uniform_ratio=uniform_ratio,
        use_occ=specs["UseOccupancy"],
        root=data_source,
    )

    reconstruct_list = list(range(len(sdf_dataset)))
    unseed = random.randint(0, 10000)
    if args.seed is not None:
        random.seed(args.seed)
        random.shuffle(reconstruct_list)
    if args.stop is not None:
        reconstruct_list = reconstruct_list[: args.stop]
    if args.out_of_order:
        random.seed(unseed)  # Unseed the random generator
        random.shuffle(reconstruct_list)
        random.seed(args.seed)  # Reseed the random generator
    if args.split is not None:
        reconstruct_list = reconstruct_list[: int(len(reconstruct_list) / args.split)]

    input_list, path_list = [], []
    if not mesh_only:
        for ii in reconstruct_list:

            # Generate the code if necessary
            path_code = reconstruction_handler.path_code(ii, create=True)
            if (not os.path.exists(path_code)) or overwrite_codes:
                if save_iter:
                    input_list.append(
                        dict(
                            test_sdf=sdf_dataset.get_broken_sample(ii),
                            iter_path=reconstruction_handler.path_values(
                                ii, 1, create=True
                            ),
                        )
                    )
                else:
                    input_list.append(
                        dict(
                            test_sdf=sdf_dataset.get_broken_sample(ii),
                        )
                    )
                path_list.append(path_code)

    # Spawn a threadpool to do reconstruction
    num_tasks = len(input_list)
    STATUS_INDICATOR = core.utils_multiprocessing.MultiprocessBar(num_tasks)

    with multiprocessing.Pool(threads) as pool:
        if not mesh_only:
            logging.info("Starting {} threads".format(threads))
            if num_tasks != 0:
                logging.info("Reconstructing {} codes".format(num_tasks))
                STATUS_INDICATOR.reset(num_tasks)
                futures_list = []

                # Cut the work into chunks
                step_size = math.ceil(num_tasks / threads)
                for idx in range(0, num_tasks, step_size):
                    start, end = idx, min(idx + step_size, num_tasks)
                    futures_list.append(
                        pool.apply_async(
                            core.utils_multiprocessing.reconstruct_chunk,
                            tuple(
                                (
                                    input_list[start:end],
                                    path_list[start:end],
                                    reconstruct,
                                    network_kwargs,
                                    reconstruction_kwargs,
                                    overwrite_codes,
                                    callback,
                                )
                            ),
                        )
                    )

                # Wait on threads and display a progress bar
                for f in futures_list:
                    f.get()

        input_list, path_list = [], []
        for ii in reconstruct_list:

            # Generate the mesh if necessary
            for shape_idx in network_outputs:
                path_mesh = reconstruction_handler.path_mesh(ii, shape_idx, create=True)
                path_values = reconstruction_handler.path_values(ii, shape_idx)
                if (
                    not os.path.exists(path_mesh)
                    or not os.path.exists(path_values)
                    or overwrite_meshes
                ):
                    sigmoid = True
                    if shape_idx in [1, 2]:
                        sigmoid = False
                    if os.path.exists(reconstruction_handler.path_code(ii)):
                        input_list.append(
                            dict(
                                vec=reconstruction_handler.get_code(ii),
                                use_net=shape_idx,
                                save_values=path_values,
                                sigmoid=sigmoid,
                            )
                        )
                        path_list.append(path_mesh)

        # Reconstruct meshes
        num_tasks = len(input_list)
        if num_tasks != 0:
            logging.info("Reconstructing {} meshes".format(num_tasks))
            STATUS_INDICATOR.reset(num_tasks)
            futures_list = []

            # Cut the work into chunks
            step_size = math.ceil(num_tasks / threads)
            for idx in range(0, num_tasks, step_size):
                start, end = idx, min(idx + step_size, num_tasks)

                futures_list.append(
                    pool.apply_async(
                        core.utils_multiprocessing.mesh_chunk,
                        tuple(
                            (
                                input_list[start:end],
                                path_list[start:end],
                                core.reconstruct.create_mesh,
                                network_kwargs,
                                mesh_kwargs,
                                overwrite_meshes,
                                callback,
                            )
                        ),
                    )
                )

            # Wait on threads and display a progress bar
            for f in futures_list:
                f.get()

    STATUS_INDICATOR.close()

    reconstruct_list.sort()

    # Spins up a multiprocessed renderer
    logging.info("Rendering results ...")
    try:
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
            # render_gt=True,
        )
    except ValueError:
        core.handler.render_engine(
            data_handler=sdf_dataset,
            reconstruct_list=reconstruct_list,
            reconstruction_handler=reconstruction_handler,
            outputs=total_outputs,
            num_renders=3,
            resolution=render_resolution,
            composite=False,
            overwrite=overwrite_renders,
            threads=render_threads,
            # render_gt=True,
        )

    logging.info("Building summary render")
    path = os.path.join(
        reconstruction_handler.path_reconstruction(), "summary_img_{}.jpg"
    )
    if not os.path.exists(path.format(0)):
        try:
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
        except ValueError:
            img = core.vis.image_results(
                data_handler=None,
                reconstruct_list=reconstruct_list,
                reconstruction_handler=reconstruction_handler,
                outputs=total_outputs,
                num_renders=3,
                resolution=render_resolution,
                composite=False,
                knit_handlers=[],
            )

        logging.info(
            "Saving summary render to: {}".format(
                path.replace(os.environ["DATADIR"], "$DATADIR")
            )
        )
        core.vis.save_image_block(img, path)

    metrics = [
        "chamfer",
        "connected_artifacts_score2",
        "normal_consistency",
    ]
    output_pairs = [(0, 0), (1, 1), (2, 2)]
    
    for metric in metrics:
        logging.info("Computing {} ...".format(metric))
        core.handler.eval_engine(
            reconstruct_list=reconstruct_list,
            output_pairs=output_pairs,
            threads=render_threads,
            overwrite=overwrite_evals,
            reconstruction_handler=reconstruction_handler,
            data_handler=sdf_dataset,
            metric=metric,
        )

    out_metrics = reconstruction_handler.path_metrics("metrics.npy")
    core.export_report(
        out_metrics=out_metrics,
        reconstruction_handler=reconstruction_handler,
        reconstruct_list=reconstruct_list,
        output_pairs=output_pairs,
        metrics=metrics,
    )