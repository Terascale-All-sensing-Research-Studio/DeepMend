import torch
import torch.utils.data as data_utils
import signal
import sys
import os
import logging
import math
import time

import socket
import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
import core
try:
    import neptune
except ImportError:
    pass

import matplotlib
matplotlib.use('Agg')

import core.workspace as ws


class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch):
        return self.value


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):

        return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length


def get_learning_rate_schedules(specs):

    schedule_specs = specs["LearningRateSchedule"]

    schedules = []

    for schedule_specs in schedule_specs:

        if schedule_specs["Type"] == "Step":
            schedules.append(
                StepLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Interval"],
                    schedule_specs["Factor"],
                )
            )
        elif schedule_specs["Type"] == "Warmup":
            schedules.append(
                WarmupLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Final"],
                    schedule_specs["Length"],
                )
            )
        elif schedule_specs["Type"] == "Constant":
            schedules.append(ConstantLearningRateSchedule(schedule_specs["Value"]))

        else:
            raise Exception(
                'no known learning rate schedule of type "{}"'.format(
                    schedule_specs["Type"]
                )
            )

    return schedules


def save_model(experiment_directory, filename, decoder, epoch):

    model_params_dir = ws.get_model_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "model_state_dict": decoder.state_dict()},
        os.path.join(model_params_dir, filename),
    )


def save_optimizer(experiment_directory, filename, optimizer, epoch):

    optimizer_params_dir = ws.get_optimizer_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )


def load_optimizer(experiment_directory, filename, optimizer):

    full_filename = os.path.join(
        ws.get_optimizer_params_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception(
            'optimizer state dict "{}" does not exist'.format(full_filename)
        )

    data = torch.load(full_filename)

    optimizer.load_state_dict(data["optimizer_state_dict"])

    return data["epoch"]


def save_latent_vectors(experiment_directory, filename, latent_vec, epoch):

    latent_codes_dir = ws.get_latent_codes_dir(experiment_directory, True)

    all_latents = latent_vec.state_dict()

    torch.save(
        {"epoch": epoch, "latent_codes": all_latents},
        os.path.join(latent_codes_dir, filename),
    )


# TODO: duplicated in workspace
def load_latent_vectors(experiment_directory, filename, lat_vecs):

    full_filename = os.path.join(
        ws.get_latent_codes_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception('latent state file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    if isinstance(data["latent_codes"], torch.Tensor):

        # for backwards compatibility
        if not lat_vecs.num_embeddings == data["latent_codes"].size()[0]:
            raise Exception(
                "num latent codes mismatched: {} vs {}".format(
                    lat_vecs.num_embeddings, data["latent_codes"].size()[0]
                )
            )

        if not lat_vecs.embedding_dim == data["latent_codes"].size()[2]:
            raise Exception("latent code dimensionality mismatch")

        for i, lat_vec in enumerate(data["latent_codes"]):
            lat_vecs.weight.data[i, :] = lat_vec

    else:
        lat_vecs.load_state_dict(data["latent_codes"])

    return data["epoch"]


def save_logs(
    experiment_directory,
    loss_log,
    lr_log,
    timing_log,
    lat_mag_log,
    param_mag_log,
    epoch,
):

    torch.save(
        {
            "epoch": epoch,
            "loss": loss_log,
            "learning_rate": lr_log,
            "timing": timing_log,
            "latent_magnitude": lat_mag_log,
            "param_magnitude": param_mag_log,
        },
        os.path.join(experiment_directory, ws.logs_filename),
    )


def load_logs(experiment_directory):

    full_filename = os.path.join(experiment_directory, ws.logs_filename)

    if not os.path.isfile(full_filename):
        raise Exception('log file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    return (
        data["loss"],
        data["learning_rate"],
        data["timing"],
        data["latent_magnitude"],
        data["param_magnitude"],
        data["epoch"],
    )


def clip_logs(loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, epoch):

    iters_per_epoch = len(loss_log) // len(lr_log)

    loss_log = loss_log[: (iters_per_epoch * epoch)]
    lr_log = lr_log[:epoch]
    timing_log = timing_log[:epoch]
    lat_mag_log = lat_mag_log[:epoch]
    for n in param_mag_log:
        param_mag_log[n] = param_mag_log[n][:epoch]

    return (loss_log, lr_log, timing_log, lat_mag_log, param_mag_log)


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def get_mean_latent_vector_magnitude(latent_vectors):
    return torch.mean(torch.norm(latent_vectors.weight.data.detach(), dim=1))


# >>> begin update: get mean magnitude from a list of lat vecs
def get_mean_latent_vector_magnitude_list(latent_vector_list):
    mag_list = [torch.mean(torch.norm(lv, dim=1)) for lv in latent_vector_list]
    return np.array(mag_list).mean()


# >>> end update


def append_parameter_magnitudes(param_mag_log, model):
    for name, param in model.named_parameters():
        if len(name) > 7 and name[:7] == "module.":
            name = name[7:]
        if name not in param_mag_log.keys():
            param_mag_log[name] = []
        param_mag_log[name].append(param.data.norm().item())


def main_function(experiment_directory, continue_from, batch_split):

    logging.debug("running " + experiment_directory)

    specs = ws.load_experiment_specifications(experiment_directory)

    logging.info("Experiment description: \n" + specs["Description"])

    data_source = specs["DataSource"]
    train_split_file = specs["TrainSplit"]

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    logging.debug(specs["NetworkSpecs"])

    # >>> begin update: added break lat vec
    latent_size = specs["CodeLength"]
    tool_latent_size = specs["BreakCodeLength"]
    use_break_loss = specs.get("BreakLoss", True)
    # >>> end update

    checkpoints = list(
        range(
            specs["SnapshotFrequency"],
            specs["NumEpochs"] + 1,
            specs["SnapshotFrequency"],
        )
    )

    for checkpoint in specs["AdditionalSnapshots"]:
        checkpoints.append(checkpoint)
    checkpoints.sort()

    lr_schedules = get_learning_rate_schedules(specs)

    grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)
    if grad_clip is not None:
        logging.debug("clipping gradients to max norm {}".format(grad_clip))

    def save_latest(epoch):

        save_model(experiment_directory, "latest.pth", decoder, epoch)
        save_optimizer(experiment_directory, "latest.pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, "latest.pth", lat_vecs, epoch)
        save_latent_vectors(
            experiment_directory, "latest_tool.pth", tool_lat_vecs, epoch
        )

    def save_checkpoints(epoch):

        save_model(experiment_directory, str(epoch) + ".pth", decoder, epoch)
        save_optimizer(experiment_directory, str(epoch) + ".pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, str(epoch) + ".pth", lat_vecs, epoch)
        save_latent_vectors(
            experiment_directory, str(epoch) + "_tool.pth", tool_lat_vecs, epoch
        )

    def signal_handler(sig, frame):
        logging.info("Stopping early...")
        sys.exit(0)

    def adjust_learning_rate(lr_schedules, optimizer, epoch):

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

    signal.signal(signal.SIGINT, signal_handler)

    num_samp_per_scene = specs["SamplesPerScene"]
    scene_per_batch = specs["ScenesPerBatch"]
    # >>> begin update: these are no longer needed
    # clamp_dist = specs["ClampingDistance"]
    # minT = -clamp_dist
    # maxT = clamp_dist
    # enforce_minmax = True
    # >>> end update

    do_code_regularization = get_spec_with_default(specs, "CodeRegularization", True)
    code_reg_lambda = get_spec_with_default(specs, "CodeRegularizationLambda", 1e-4)

    code_bound = get_spec_with_default(specs, "CodeBound", None)

    # >>> begin update: added a few passable arguments
    assert specs["NetworkArch"] in [
        "decoder_z_lb_occ_leaky",
    ], "wrong arch: {}".format(specs["NetworkArch"])
    one_code_per_complete = get_spec_with_default(specs, "OneCodePerComplete", False)
    reg_loss_warmup = get_spec_with_default(specs, "CodeRegularizationWarmup", 100)
    # >>> end update

    # >>> begin update: we need to pass a few more things to the network
    decoder = arch.Decoder(
        latent_size=latent_size,
        tool_latent_size=tool_latent_size,
        num_dims=3,
        do_code_regularization=do_code_regularization,
        **specs["NetworkSpecs"],
        **specs["SubnetSpecs"],
    ).cuda()
    # >>> end update

    logging.info("training with {} GPU(s)".format(torch.cuda.device_count()))

    # if torch.cuda.device_count() > 1:
    decoder = torch.nn.DataParallel(decoder)

    num_epochs = specs["NumEpochs"]
    log_frequency = get_spec_with_default(specs, "LogFrequency", 10)

    # >>> begin update: using our dataloader
    # Get the data directory from environment variable
    train_split_file = train_split_file.replace("$DATADIR", os.environ["DATADIR"])

    # Create and load the dataset
    sdf_dataset = core.data.SamplesDataset(
        train_split_file,
        learned_breaks=True,
        subsample=num_samp_per_scene,
        uniform_ratio=specs["UniformRatio"],
        use_occ=specs["UseOccupancy"],
        one_code_per_complete=one_code_per_complete,
        return_tool=use_break_loss, # If we're using break loss, we need the tool
    )
    # >>> end update

    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 1)
    logging.debug("loading data with {} threads".format(num_data_loader_threads))

    sdf_loader = data_utils.DataLoader(
        sdf_dataset,
        batch_size=scene_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=True,
    )

    logging.debug("torch num_threads: {}".format(torch.get_num_threads()))

    num_scenes = len(sdf_dataset)

    logging.info("There are {} scenes".format(num_scenes))

    logging.debug(decoder)

    # >>> begin update: added break lat vec
    num_complete_vecs = sdf_dataset.num_instances
    if one_code_per_complete:
        num_complete_vecs = sdf_dataset.num_shapes
    lat_vecs = torch.nn.Embedding(num_complete_vecs, latent_size, max_norm=code_bound)
    torch.nn.init.normal_(
        lat_vecs.weight.data,
        0.0,
        get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(latent_size),
    )
    logging.debug(
        "initialized with mean magnitude {}".format(
            get_mean_latent_vector_magnitude(lat_vecs)
        )
    )

    tool_lat_vecs = torch.nn.Embedding(
        sdf_dataset.num_instances, tool_latent_size, max_norm=code_bound
    )
    torch.nn.init.normal_(
        tool_lat_vecs.weight.data,
        0.0,
        get_spec_with_default(specs, "CodeInitStdDev", 1.0)
        / math.sqrt(tool_latent_size),
    )
    logging.debug(
        "initialized with mean break magnitude {}".format(
            get_mean_latent_vector_magnitude(tool_lat_vecs)
        )
    )
    # >>> end update

    # >>> begin update: using the bceloss
    loss_bce_with_logits = torch.nn.BCEWithLogitsLoss(reduction="sum")
    loss_bce = torch.nn.BCELoss(reduction="sum")
    # >>> end update

    # >>> begin update: added break lat vec
    optimizer_all = torch.optim.Adam(
        [
            {
                "params": decoder.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            },
            {
                "params": lat_vecs.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            },
            {
                "params": tool_lat_vecs.parameters(),
                "lr": lr_schedules[2].get_learning_rate(0),
            },
        ]
    )
    # >>> end update

    loss_log = []
    lr_log = []
    lat_mag_log = []
    timing_log = []
    param_mag_log = {}

    start_epoch = 1

    # >>> begin update: added network backup
    backup_location = specs.setdefault("NetBackupLocation", None)
    if backup_location is not None:
        try:
            backup_location = backup_location.replace(
                "$NETBACKUP", os.environ["NETBACKUP"]
            )
        except KeyError:
            pass
        core.train.network_backup(experiment_directory, backup_location)
    # >>> end update

    # >>> begin update: neptune logging
    neptune_name = specs.setdefault("NeptuneName", None)
    test_every = specs.setdefault("TestEvery", None)
    stop_netptune_after = get_spec_with_default(specs, "StopNeptuneAfter", 200)
    if "neptune" not in locals() and neptune_name is not None:
        neptune_name = None
        logging.warning("Could not import neptune, disabling")
    if neptune_name is not None:
        logging.info("Logging to neptune project: {}".format(neptune_name))
        neptune.init(
            project_qualified_name=neptune_name,
            api_token=os.getenv("NEPTUNE_API_TOKEN"),
        )
        params = specs
        params.update(
            {
                "hostname": str(socket.gethostname()),
                "experiment_dir": os.path.basename(experiment_directory),
                "device count": str(int(torch.cuda.device_count())),
                "loader threads": str(int(num_data_loader_threads)),
                "torch threads": str(int(torch.get_num_threads())),
            }
        )
        neptune.create_experiment(params=params)
    # >>> end update

    if continue_from is not None:

        logging.info('continuing from "{}"'.format(continue_from))

        lat_epoch = load_latent_vectors(
            experiment_directory, continue_from + ".pth", lat_vecs
        )

        # >>> begin update: loading break vecs
        lat_epoch = load_latent_vectors(
            experiment_directory, continue_from + "_tool.pth", tool_lat_vecs
        )
        # >>> end update

        model_epoch = ws.load_model_parameters(
            experiment_directory, continue_from, decoder
        )

        optimizer_epoch = load_optimizer(
            experiment_directory, continue_from + ".pth", optimizer_all
        )

        loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, log_epoch = load_logs(
            experiment_directory
        )

        if not log_epoch == model_epoch:
            loss_log, lr_log, timing_log, lat_mag_log, param_mag_log = clip_logs(
                loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, model_epoch
            )

        if not (model_epoch == optimizer_epoch and model_epoch == lat_epoch):
            raise RuntimeError(
                "epoch mismatch: {} vs {} vs {} vs {}".format(
                    model_epoch, optimizer_epoch, lat_epoch, log_epoch
                )
            )

        start_epoch = model_epoch + 1

        logging.debug("loaded")

    logging.info("starting from epoch {}".format(start_epoch))

    logging.info(
        "Number of decoder parameters: {}".format(
            sum(p.data.nelement() for p in decoder.parameters())
        )
    )
    logging.info(
        "Number of shape code parameters: {} (# codes {}, code dim {})".format(
            lat_vecs.num_embeddings * lat_vecs.embedding_dim,
            lat_vecs.num_embeddings,
            lat_vecs.embedding_dim,
        )
    )
    # >>> being update: added break vecs
    logging.info(
        "Number of tool shape code parameters: {} (# codes {}, code dim {})".format(
            tool_lat_vecs.num_embeddings * tool_lat_vecs.embedding_dim,
            tool_lat_vecs.num_embeddings,
            tool_lat_vecs.embedding_dim,
        )
    )
    # >>> end update

    # >>> being update: added tqdm indicator
    for epoch in tqdm.tqdm(
        range(start_epoch, num_epochs + 1), initial=start_epoch, total=num_epochs
    ):

        start = time.time()

        # logging.info("epoch {}...".format(epoch))

        if (stop_netptune_after is not False) and (epoch > stop_netptune_after):
            if neptune_name is not None:
                logging.info("Stopping netpune...")
                neptune.stop()
            neptune_name = None
        # >>> end update

        decoder.train()

        adjust_learning_rate(lr_schedules, optimizer_all, epoch)

        # >>> being update: data is in a slightly different format
        for data, inds in sdf_loader:
            # returns ((pts, cgt, bgt, rgt, tgt), (indices))

            for d in range(len(data)):
                data[d] = data[d].reshape(-1, data[d].shape[-1])
            num_sdf_samples = data[0].shape[0]

            # Disambiguate pts
            pts = data[0]
            pts.requires_grad = False

            # Disambiguate gt complete, broken, restoration, tool
            gts = data[1:]
            for d in range(len(gts)):
                gts[d].requires_grad = False

            # Disambiguate indices
            indices = inds[0]
            tool_indices = inds[1]

            # Chunk points
            xyz = pts.type(torch.float)
            xyz = torch.chunk(xyz, batch_split)

            # Chunk occ
            for d in range(len(gts)):
                gts[d] = gts[d].type(torch.float)
                gts[d] = torch.chunk(gts[d], batch_split)

            # Chunk indices
            indices = torch.chunk(
                indices.flatten(),
                batch_split,
            )
            tool_indices = torch.chunk(
                tool_indices.flatten(),
                batch_split,
            )
            # >>> end update

            batch_loss = 0.0

            optimizer_all.zero_grad()

            for i in range(batch_split):

                batch_vecs = lat_vecs(indices[i])
                # >>> begin update: added break vec
                tool_batch_vecs = tool_lat_vecs(tool_indices[i])

                input = torch.cat([batch_vecs, tool_batch_vecs, xyz[i]], dim=1)

                # NN optimization
                c_x, b_x, r_x, t_x = decoder(input.cuda())
                # >>> end update

                # >>> begin update: different loss
                try:
                    c_gt, b_gt, r_gt, t_gt = [g[i].cuda() for g in gts]
                except ValueError:
                    c_gt, b_gt, r_gt = [g[i].cuda() for g in gts]
                chunk_loss = (
                    loss_bce_with_logits(c_x, c_gt)
                    + loss_bce(b_x, b_gt)
                    + loss_bce(r_x, r_gt)
                ) / num_sdf_samples
                if use_break_loss:
                    break_loss = loss_bce_with_logits(t_x, t_gt) / num_sdf_samples
                    chunk_loss += break_loss
                # >>> end update

                # >>> begin update: neptune logging
                if neptune_name is not None:
                    neptune.log_metric("data loss", chunk_loss.item())
                    if use_break_loss:
                        neptune.log_metric("break loss", break_loss.item())
                    neptune.log_metric(
                        "c accuracy",
                        accuracy_score(
                            torch.sigmoid(c_x).cpu().detach().round().numpy(),
                            gts[0][i].numpy(),
                        ),
                    )
                    neptune.log_metric(
                        "b accuracy",
                        accuracy_score(
                            b_x.cpu().detach().round().numpy(),
                            gts[1][i].numpy(),
                        ),
                    )
                    neptune.log_metric(
                        "r accuracy",
                        accuracy_score(
                            r_x.cpu().detach().round().numpy(),
                            gts[2][i].numpy(),
                        ),
                    )
                    try:
                        neptune.log_metric(
                            "t accuracy",
                            accuracy_score(
                                torch.sigmoid(t_x).cpu().detach().round().numpy(),
                                gts[3][i].numpy(),
                            ),
                        )
                    except IndexError:
                        pass
                # >>> end update

                if do_code_regularization:
                    # >>> begin update: apply regularization to all vectors, renamed to l1_loss
                    l1_size_loss = torch.sum(torch.norm(batch_vecs, dim=1)) + torch.sum(
                        torch.norm(tool_batch_vecs, dim=1)
                    )
                    # >>> end update

                    # >>> begin update: added passable argument warmup, renamed to l1_loss
                    reg_loss = (
                        code_reg_lambda * min(1, epoch / reg_loss_warmup) * l1_size_loss
                    ) / num_sdf_samples
                    # >>> end update

                    # >>> begin update: neptune logging
                    if neptune_name is not None:
                        neptune.log_metric("reg loss", reg_loss.item())
                    # >>> end update

                    chunk_loss = chunk_loss + reg_loss.cuda()

                chunk_loss.backward()

                batch_loss += chunk_loss.item()

            logging.debug("loss = {}".format(batch_loss))

            loss_log.append(batch_loss)

            if grad_clip is not None:

                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)

            optimizer_all.step()

        # Log some images
        if (neptune_name is not None) and (test_every is not None):
            if epoch % test_every == 0:
                pts = xyz[-1][:num_samp_per_scene, :].cpu().detach().numpy()
                try:
                    neptune.log_image(
                        "t_x.jpg".format(epoch),
                        core.plt2numpy(
                            core.vis.plot_samples(
                                (
                                    pts, 
                                    t_x[:num_samp_per_scene, :].cpu().detach().numpy(),
                                ), 
                                n_plots=16,
                            )
                        ).astype(float) / 255
                    )
                except NameError:
                    pass
                neptune.log_image(
                    "c_x.jpg".format(epoch),
                    core.plt2numpy(
                        core.vis.plot_samples(
                            (
                                pts, 
                                c_x[:num_samp_per_scene, :].cpu().detach().numpy(),
                            ), 
                            n_plots=16,
                        )
                    ).astype(float) / 255
                )
                neptune.log_image(
                    "r_x.jpg".format(epoch),
                    core.plt2numpy(
                        core.vis.plot_samples(
                            (
                                pts, 
                                r_x[:num_samp_per_scene, :].cpu().detach().numpy(),
                            ), 
                            n_plots=16,
                        )
                    ).astype(float) / 255
                )
                neptune.log_image(
                    "b_x.jpg".format(epoch),
                    core.plt2numpy(
                        core.vis.plot_samples(
                            (
                                pts, 
                                b_x[:num_samp_per_scene, :].cpu().detach().numpy(),
                            ), 
                            n_plots=16,
                        )
                    ).astype(float) / 255
                )

        end = time.time()

        seconds_elapsed = end - start
        timing_log.append(seconds_elapsed)

        lr_log.append([schedule.get_learning_rate(epoch) for schedule in lr_schedules])

        lat_mag_log.append(get_mean_latent_vector_magnitude(lat_vecs))

        append_parameter_magnitudes(param_mag_log, decoder)

        # >>> begin update: neptune logging
        if neptune_name is not None:
            neptune.log_metric("z mag", get_mean_latent_vector_magnitude(lat_vecs))
            neptune.log_metric("t mag", get_mean_latent_vector_magnitude(tool_lat_vecs))
            neptune.log_metric("time", seconds_elapsed)
        # >>> end update

        if epoch in checkpoints:
            save_checkpoints(epoch)

            # >>> begin update: added network backup
            if backup_location is not None:
                core.train.network_backup(experiment_directory, backup_location)
            # >>> end update

        if epoch % log_frequency == 0:

            save_latest(epoch)
            save_logs(
                experiment_directory,
                loss_log,
                lr_log,
                timing_log,
                lat_mag_log,
                param_mag_log,
                epoch,
            )


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Train a DeepSDF autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--continue",
        "-c",
        dest="continue_from",
        help="A snapshot to continue from. This can be 'latest' to continue"
        + "from the latest running snapshot, or an integer corresponding to "
        + "an epochal snapshot.",
    )
    arg_parser.add_argument(
        "--batch_split",
        dest="batch_split",
        default=1,
        help="This splits the batch into separate subbatches which are "
        + "processed separately, with gradients accumulated across all "
        + "subbatches. This allows for training with large effective batch "
        + "sizes in memory constrained environments.",
    )
    core.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    core.configure_logging(args)

    main_function(args.experiment_directory, args.continue_from, int(args.batch_split))
