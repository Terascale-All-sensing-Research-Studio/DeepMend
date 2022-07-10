import os
import argparse
import logging
import random
import json
import time

import tqdm

import processor.utils as utils
import processor.shapenet as shapenet
import processor.logger as logger
from processor.utils import (
    GracefulProcessPoolExecutor,
    GracefulProcessPoolExecutorDebug,
)


def main(
    root_dir,
    ops,
    threads,
    overwrite,
    num_breaks,
    class_subsample,
    instance_subsample,
    max_break,
    min_break,
    num_renders,
    splits_file,
    train_ratio,
    debug,
    break_all,
    outoforder,
    break_method,
    use_tool,
    break_handle,
    reorient,
):

    logging.info("Performing the following operations: {}".format(ops))
    logging.info("Using {} thread(s)".format(threads))
    logging.info("Using splits file: {}".format(splits_file))

    if (splits_file is not None) and os.path.exists(splits_file):
        logging.info("Loading saved data from splits file {}".format(splits_file))
        object_id_dict = json.load(open(splits_file, "r"))

        root_dir = shapenet.shapenet_toplevel(root_dir)
        id_train_list = [
            shapenet.ShapeNetObject(root_dir, o[0], o[1])
            for o in object_id_dict["id_train_list"]
        ]
        id_test_list = [
            shapenet.ShapeNetObject(root_dir, o[0], o[1])
            for o in object_id_dict["id_test_list"]
        ]
        object_id_list = id_test_list + id_train_list
    else:
        # Obtain a list of all the objects in the shapenet dataset
        logging.info("Searching for objects ...")
        object_id_list, root_dir = shapenet.shapenet_search(
            root_dir, return_toplevel=True
        )
        logging.info("Found {} objects".format(len(object_id_list)))

        # Subsample this list, if required
        if (class_subsample is not None) or (instance_subsample is not None):
            class_list = list(set([o.class_id for o in object_id_list]))

            # Sample classes
            if class_subsample is not None:
                try:
                    class_list = random.sample(class_list, class_subsample)
                except ValueError:
                    raise ValueError(
                        "Requested too many samples, there are only {} objects".format(
                            len(class_list)
                        )
                    )

            # Group object by class
            id_by_class = []
            for c in class_list:
                id_by_class.append([o for o in object_id_list if o.class_id == c])

            # Sample instances
            if instance_subsample is not None:
                for idx in range(len(id_by_class)):
                    # It's often the case that
                    if len(id_by_class[idx]) < instance_subsample:
                        logging.warning(
                            "Only {} samples in class {}, adding all".format(
                                len(id_by_class[idx]), class_list[idx]
                            )
                        )
                    try:
                        id_by_class[idx] = random.sample(
                            id_by_class[idx],
                            min(instance_subsample, len(id_by_class[idx])),
                        )
                    except ValueError:
                        raise ValueError(
                            "Requested too many samples, there are only {} objects".format(
                                len(class_list)
                            )
                        )

            # Flatten list
            object_id_list = []
            for e in id_by_class:
                object_id_list.extend(e)

        # Split into a train test list
        id_train_list, id_test_list = utils.split_train_test(
            object_id_list, train_ratio
        )

        logging.info("Reduced to {} objects after sampling".format(len(object_id_list)))

        # Save the list
        logging.info("Saving data to splits file {}".format(splits_file))
        json.dump(
            {
                "id_train_list": [[o.class_id, o.instance_id] for o in id_train_list],
                "id_test_list": [[o.class_id, o.instance_id] for o in id_test_list],
            },
            open(splits_file, "w"),
        )
    logging.info("Building subdirectories ...")
    for o in object_id_list:
        o.build_dirs()

    if outoforder:
        random.shuffle(id_train_list)
        random.shuffle(id_test_list)
        object_id_list = id_test_list + id_train_list

    logging.info("Root dir at {}".format(root_dir))
    logging.info("Processing {} objects".format(len(object_id_list)))
    logging.info(
        "{} train objects, {} test objects".format(
            len(id_train_list), len(id_test_list)
        )
    )
    logging.info("{} classes".format(len(set([o.class_id for o in object_id_list]))))

    global GracefulProcessPoolExecutor
    # This will completely disable the pool
    if (threads == 1) and (debug):
        GracefulProcessPoolExecutor = GracefulProcessPoolExecutorDebug

    object_counter = {o: [0, 0] for o in ops}

    with GracefulProcessPoolExecutor(max_workers=threads) as executor:
        # Seed randomizer
        if not ((threads == 1) and (debug)):
            executor.map(
                utils.random_seeder, [int(time.time()) + i for i in range(threads)]
            )

        try:
            if "WATERPROOF" in ops:
                import processor.process_waterproof as waterproofer

                logging.info("Waterproofing ...")

                pbar = tqdm.tqdm(object_id_list, desc="Waterproofing")
                for obj in pbar:
                    pbar.write("[{}]".format(os.path.dirname(obj.path_normalized())))

                    # Run if the file doesnt already exist
                    f_in = obj.path_normalized()
                    f_out = obj.path_waterproofed()
                    if os.path.exists(f_in) and (
                        not os.path.exists(f_out) or overwrite
                    ):
                        executor.graceful_submit(
                            waterproofer.handsoff, f_in=f_in, f_out=f_out, normalize=False,
                        )
                executor.graceful_finish()

                # Count the number of successful objects
                object_counter["WATERPROOF"][1] = len(object_id_list)
                for obj in object_id_list:
                    if os.path.exists(obj.path_waterproofed()):
                        object_counter["WATERPROOF"][0] += 1

            if "CLEAN" in ops:
                import processor.process_normalize as normalizer

                logging.info("Cleaning ...")

                pbar = tqdm.tqdm(object_id_list, desc="Cleaning")
                for obj in pbar:
                    pbar.write("[{}]".format(os.path.dirname(obj.path_normalized())))

                    # Get the paths
                    f_in = obj.path_waterproofed()
                    f_out = obj.path_c()
                    if os.path.exists(f_in) and (
                        not os.path.exists(f_out) or overwrite
                    ):
                        executor.graceful_submit(
                            normalizer.normalize, f_in=f_in, f_out=f_out, reorient=reorient,
                        )
                executor.graceful_finish()

                # Count the number of successful objects
                object_counter["CLEAN"][1] = len(object_id_list)
                for obj in object_id_list:
                    if os.path.exists(obj.path_c()):
                        object_counter["CLEAN"][0] += 1

            if "BREAK" in ops:
                if break_handle:
                    import processor.process_break_handle as breaker
                else:
                    import processor.process_break as breaker

                logging.info("Breaking ...")

                if break_all:
                    process_list = object_id_list
                else:
                    process_list = id_test_list
                pbar = tqdm.tqdm(process_list, desc="Breaking")

                for obj in pbar:
                    pbar.write("[{}]".format(os.path.dirname(obj.path_normalized())))

                    # Get the paths
                    f_in = obj.path_c()

                    # Sumbit
                    for idx in range(num_breaks):
                        f_bro = obj.path_b(idx)
                        f_res = obj.path_r(idx)

                        # Save the tool if use_tool is passed
                        f_tool = False
                        if use_tool:
                            f_tool = obj.path_tool(idx)

                        if os.path.exists(f_in) and (
                            not os.path.exists(f_bro) or overwrite
                        ):
                            executor.graceful_submit(
                                breaker.process,
                                f_in=f_in,
                                f_out=f_bro,
                                f_restoration=f_res,
                                f_tool=f_tool,
                                validate=True,
                                min_break=min_break,
                                max_break=max_break,
                                break_method=break_method,
                            )
                executor.graceful_finish()

                # Count the number of successful objects
                object_counter["BREAK"][1] = len(process_list) * num_breaks
                for obj in process_list:
                    for idx in range(num_breaks):
                        if use_tool:
                            if (
                                os.path.exists(obj.path_b(idx))
                                and os.path.exists(obj.path_r(idx))
                                and os.path.exists(obj.path_tool(idx))
                            ):
                                object_counter["BREAK"][0] += 1
                        else:
                            if os.path.exists(obj.path_b(idx)) and os.path.exists(
                                obj.path_r(idx)
                            ):
                                object_counter["BREAK"][0] += 1

            if "RENDER" in ops:
                import processor.process_render as renderer

                logging.info("Rendering ...")

                pbar = tqdm.tqdm(object_id_list, desc="Rendering")
                for obj in pbar:
                    pbar.write("[{}]".format(os.path.dirname(obj.path_normalized())))

                    for angle in range(0, 360, int(360 / num_renders)):
                        # Process complete
                        f_in = obj.path_c()
                        f_out = obj.path_c_rendered(angle)
                        if os.path.exists(f_in) and (
                            not os.path.exists(f_out) or overwrite
                        ):
                            executor.graceful_submit(
                                renderer.process, f_in=f_in, f_out=f_out, yrot=angle
                            )

                        for idx in range(num_breaks):
                            # Process broken
                            f_in = obj.path_b(idx)
                            f_out = obj.path_b_rendered(idx, angle)
                            if os.path.exists(f_in) and (
                                not os.path.exists(f_out) or overwrite
                            ):
                                executor.graceful_submit(
                                    renderer.process, f_in=f_in, f_out=f_out, yrot=angle
                                )

                            # Process restoration
                            f_in = obj.path_r(idx)
                            f_out = obj.path_r_rendered(idx, angle)
                            if os.path.exists(f_in) and (
                                not os.path.exists(f_out) or overwrite
                            ):
                                executor.graceful_submit(
                                    renderer.process, f_in=f_in, f_out=f_out, yrot=angle
                                )

                executor.graceful_finish()

                # Count the number of successful objects
                object_counter["RENDER"][1] = (
                    len(object_id_list) * num_breaks * num_renders
                )
                for obj in object_id_list:
                    for angle in range(0, 360, int(360 / num_renders)):
                        for idx in range(num_breaks):
                            if (
                                os.path.exists(obj.path_c_rendered(angle))
                                and os.path.exists(obj.path_b_rendered(idx, angle))
                                and os.path.exists(obj.path_r_rendered(idx, angle))
                            ):
                                object_counter["RENDER"][0] += 1

            if "PARTIAL_SDF" in ops:
                import processor.process_partial_view as compute_sdf_partial

                logging.info("Computing partial sample points ...")

                pbar = tqdm.tqdm(object_id_list, desc="Computing Partial SDF")

                for obj in pbar:
                    pbar.write("[{}]".format(os.path.dirname(obj.path_normalized())))

                    # Process complete
                    for idx in range(num_breaks):
                        f_out = obj.path_b_partial_sdf(idx)
                        f_in = obj.path_b(idx)
                        f_rest = obj.path_r(idx)

                        if all([os.path.exists(f) for f in [f_in, f_rest]]) and (
                            not os.path.exists(f_out) or overwrite
                        ):
                            executor.graceful_submit(
                                compute_sdf_partial.process,
                                f_in=f_in,
                                f_rest=f_rest,
                                f_out=f_out,
                            )
                executor.graceful_finish()

                # Count the number of successful objects
                object_counter["PARTIAL_SDF"][1] = len(object_id_list) * num_breaks
                for obj in object_id_list:
                    for idx in range(num_breaks):
                        if os.path.exists(obj.path_b_partial_sdf(idx)):
                            object_counter["PARTIAL_SDF"][0] += 1

            if "SAMPLE" in ops:
                import processor.process_sample as sampler

                logging.info("Computing sample points ...")

                pbar = tqdm.tqdm(object_id_list, desc="Sampling")
                for obj in pbar:
                    pbar.write("[{}]".format(os.path.dirname(obj.path_normalized())))

                    # Process complete
                    for idx in range(num_breaks):
                        f_out = obj.path_sampled(idx)

                        # If we're using the tool then compute the sample points
                        # wrt to the complete, broken, restoration and the tool. 
                        # Else compute them wrt to the complete, broken, and 
                        # restoration objects.
                        if use_tool:
                            f_in = [obj.path_c(), obj.path_b(idx), obj.path_r(idx), obj.path_tool(idx)]
                        else:
                            f_in = [obj.path_c(), obj.path_b(idx), obj.path_r(idx)]

                        if all([os.path.exists(f) for f in f_in]) and (
                            not os.path.exists(f_out) or overwrite
                        ):
                            executor.graceful_submit(
                                sampler.process, f_in=f_in, f_out=f_out
                            )
                executor.graceful_finish()

                # Count the number of successful objects
                object_counter["SAMPLE"][1] = len(object_id_list) * num_breaks
                for obj in object_id_list:
                    for idx in range(num_breaks):
                        if os.path.exists(obj.path_sampled(idx)):
                            object_counter["SAMPLE"][0] += 1

            if "SPLINE" in ops:
                import processor.process_spline as spline_fit

                logging.info("Computing fracture spline fit ...")

                pbar = tqdm.tqdm(id_train_list, desc="Spline fit")
                for obj in pbar:
                    pbar.write("[{}]".format(os.path.dirname(obj.path_normalized())))

                    # Process complete
                    for idx in range(num_breaks):
                        f_out = obj.path_spline_sdf(idx)

                        f_in = [
                            obj.path_b(idx),
                            obj.path_b_sdf(idx),
                            obj.path_r(idx),
                            obj.path_r_sdf(idx),
                            obj.path_sampled(idx),
                        ]

                        if all([os.path.exists(f) for f in f_in]) and (
                            not os.path.exists(f_out) or overwrite
                        ):
                            executor.graceful_submit(
                                spline_fit.process, 
                                f_in=obj.path_b(idx),
                                f_sdf=obj.path_b_sdf(idx),
                                f_rest=obj.path_r(idx),
                                f_rest_sdf=obj.path_r_sdf(idx),
                                f_samp=obj.path_sampled(idx),
                                f_out=f_out,
                                f_plane=obj.path_spline_plane(idx),
                            )
                executor.graceful_finish()

                # Count the number of successful objects
                object_counter["SPLINE"][1] = len(id_train_list) * num_breaks
                for obj in id_train_list:
                    for idx in range(num_breaks):
                        if os.path.exists(obj.path_spline_sdf(idx)):
                            object_counter["SPLINE"][0] += 1

            if "SDF" in ops:
                import processor.process_sdf as compute_sdf

                logging.info("Computing SDF ...")

                pbar = tqdm.tqdm(object_id_list, desc="Computing SDF")
                for obj in pbar:
                    pbar.write("[{}]".format(os.path.dirname(obj.path_normalized())))

                    for idx in range(num_breaks):
                        # Process complete
                        f_in = obj.path_c()
                        f_out = obj.path_c_sdf(idx)
                        f_samp = obj.path_sampled(idx)
                        if (
                            os.path.exists(f_in)
                            and os.path.exists(f_samp)
                            and (not os.path.exists(f_out) or overwrite)
                        ):
                            executor.graceful_submit(
                                compute_sdf.process,
                                f_in=f_in,
                                f_out=f_out,
                                f_samp=f_samp,
                            )

                        if use_tool:

                            # Process tool
                            f_in = obj.path_tool(idx)
                            f_out = obj.path_tool_sdf(idx)
                            if (
                                os.path.exists(f_in)
                                and os.path.exists(f_samp)
                                and (not os.path.exists(f_out) or overwrite)
                            ):
                                executor.graceful_submit(
                                    compute_sdf.process,
                                    f_in=f_in,
                                    f_out=f_out,
                                    f_samp=f_samp,
                                )

                        # Process broken
                        f_in = obj.path_b(idx)
                        f_out = obj.path_b_sdf(idx)
                        if (
                            os.path.exists(f_in)
                            and os.path.exists(f_samp)
                            and (not os.path.exists(f_out) or overwrite)
                        ):
                            executor.graceful_submit(
                                compute_sdf.process,
                                f_in=f_in,
                                f_out=f_out,
                                f_samp=f_samp,
                            )

                        # Process restoration
                        f_in = obj.path_r(idx)
                        f_out = obj.path_r_sdf(idx)
                        if (
                            os.path.exists(f_in)
                            and os.path.exists(f_samp)
                            and (not os.path.exists(f_out) or overwrite)
                        ):
                            executor.graceful_submit(
                                compute_sdf.process,
                                f_in=f_in,
                                f_out=f_out,
                                f_samp=f_samp,
                            )

                executor.graceful_finish()

                # Count the number of successful objects
                object_counter["SDF"][1] = len(object_id_list) * num_breaks
                for obj in object_id_list:
                    for idx in range(num_breaks):
                        if use_tool:
                            if (
                                os.path.exists(obj.path_c_sdf(idx))
                                and os.path.exists(obj.path_b_sdf(idx))
                                and os.path.exists(obj.path_r_sdf(idx))
                                and os.path.exists(obj.path_tool_sdf(idx))
                            ):
                                object_counter["SDF"][0] += 1
                        else:
                            if (
                                os.path.exists(obj.path_c_sdf(idx))
                                and os.path.exists(obj.path_b_sdf(idx))
                                and os.path.exists(obj.path_r_sdf(idx))
                            ):
                                object_counter["SDF"][0] += 1

            if "OCC" in ops:
                import processor.process_occupancies as compute_occ

                logging.info("Computing OCC ...")

                pbar = tqdm.tqdm(object_id_list, desc="Computing OCC")
                for obj in pbar:
                    pbar.write("[{}]".format(os.path.dirname(obj.path_normalized())))

                    for idx in range(num_breaks):
                        # Process complete
                        f_in = obj.path_c()
                        f_out = obj.path_c_occ(idx)
                        f_samp = obj.path_sampled(idx)
                        if (
                            os.path.exists(f_in)
                            and os.path.exists(f_samp)
                            and (not os.path.exists(f_out) or overwrite)
                        ):
                            executor.graceful_submit(
                                compute_occ.process,
                                f_in=f_in,
                                f_out=f_out,
                                f_samp=f_samp,
                            )

                        # Process broken
                        f_in = obj.path_b(idx)
                        f_out = obj.path_b_occ(idx)
                        if (
                            os.path.exists(f_in)
                            and os.path.exists(f_samp)
                            and (not os.path.exists(f_out) or overwrite)
                        ):
                            executor.graceful_submit(
                                compute_occ.process,
                                f_in=f_in,
                                f_out=f_out,
                                f_samp=f_samp,
                            )

                        # Process restoration
                        f_in = obj.path_r(idx)
                        f_out = obj.path_r_occ(idx)
                        if (
                            os.path.exists(f_in)
                            and os.path.exists(f_samp)
                            and (not os.path.exists(f_out) or overwrite)
                        ):
                            executor.graceful_submit(
                                compute_occ.process,
                                f_in=f_in,
                                f_out=f_out,
                                f_samp=f_samp,
                            )

                executor.graceful_finish()

                # Count the number of successful objects
                object_counter["OCC"][1] = len(object_id_list) * num_breaks
                for obj in object_id_list:
                    for idx in range(num_breaks):
                        if (
                            os.path.exists(obj.path_c_occ(idx))
                            and os.path.exists(obj.path_b_occ(idx))
                            and os.path.exists(obj.path_r_occ(idx))
                        ):
                            object_counter["OCC"][0] += 1

            if "UNIFORM_OCC" in ops:
                import processor.process_occupancies_uniform as compute_occ_uni

                logging.info("Computing Uniform OCC ...")

                pbar = tqdm.tqdm(object_id_list, desc="Computing Uniform OCC")
                for obj in pbar:
                    pbar.write("[{}]".format(os.path.dirname(obj.path_normalized())))

                    # Process complete
                    f_in = obj.path_c()
                    f_out = obj.path_c_uniform_occ()
                    if os.path.exists(f_in) and (
                        not os.path.exists(f_out) or overwrite
                    ):
                        executor.graceful_submit(
                            compute_occ_uni.process,
                            f_in=f_in,
                            f_out=f_out,
                        )

                    for idx in range(num_breaks):
                        # Process broken
                        f_in = obj.path_b(idx)
                        f_out = obj.path_b_uniform_occ(idx)
                        if os.path.exists(f_in) and (
                            not os.path.exists(f_out) or overwrite
                        ):
                            executor.graceful_submit(
                                compute_occ_uni.process,
                                f_in=f_in,
                                f_out=f_out,
                            )

                        # Process restoration
                        f_in = obj.path_r(idx)
                        f_out = obj.path_r_uniform_occ(idx)
                        if os.path.exists(f_in) and (
                            not os.path.exists(f_out) or overwrite
                        ):
                            executor.graceful_submit(
                                compute_occ_uni.process,
                                f_in=f_in,
                                f_out=f_out,
                            )

                executor.graceful_finish()

                # Count the number of successful objects
                object_counter["UNIFORM_OCC"][1] = len(object_id_list) * num_breaks
                for obj in object_id_list:
                    for idx in range(num_breaks):
                        if (
                            os.path.exists(obj.path_c_uniform_occ())
                            and os.path.exists(obj.path_b_uniform_occ(idx))
                            and os.path.exists(obj.path_r_uniform_occ(idx))
                        ):
                            object_counter["UNIFORM_OCC"][0] += 1

            if "VOXEL_32" in ops:
                import processor.process_occupancies_uniform as compute_occ_uni

                logging.info("Computing Voxels ...")

                pbar = tqdm.tqdm(object_id_list, desc="Computing Voxels")
                for obj in pbar:
                    pbar.write("[{}]".format(os.path.dirname(obj.path_normalized())))

                    # Process complete
                    f_in = obj.path_c()
                    f_out = obj.path_c_voxel(size=32)
                    if os.path.exists(f_in) and (
                        not os.path.exists(f_out) or overwrite
                    ):
                        executor.graceful_submit(
                            compute_occ_uni.process,
                            f_in=f_in,
                            f_out=f_out,
                            dim=32,
                        )

                    for idx in range(num_breaks):
                        # Process broken
                        f_in = obj.path_b(idx)
                        f_out = obj.path_b_voxel(idx, size=32)
                        if os.path.exists(f_in) and (
                            not os.path.exists(f_out) or overwrite
                        ):
                            executor.graceful_submit(
                                compute_occ_uni.process,
                                f_in=f_in,
                                f_out=f_out,
                                dim=32,
                            )

                        # Process restoration
                        f_in = obj.path_r(idx)
                        f_out = obj.path_r_voxel(idx, size=32)
                        if os.path.exists(f_in) and (
                            not os.path.exists(f_out) or overwrite
                        ):
                            executor.graceful_submit(
                                compute_occ_uni.process,
                                f_in=f_in,
                                f_out=f_out,
                                dim=32,
                            )

                executor.graceful_finish()

                # Count the number of successful objects
                object_counter["VOXEL_32"][1] = len(object_id_list) * num_breaks
                for obj in object_id_list:
                    for idx in range(num_breaks):
                        if (
                            os.path.exists(obj.path_c_voxel(size=32))
                            and os.path.exists(obj.path_b_voxel(idx, size=32))
                            and os.path.exists(obj.path_r_voxel(idx, size=32))
                        ):
                            object_counter["VOXEL_32"][0] += 1

            if "UNIFORM_SDF" in ops:
                import processor.process_sdf_uniform as compute_sdf_uni

                logging.info("Computing Uniform SDF ...")

                pbar = tqdm.tqdm(object_id_list, desc="Computing Uniform SDF")
                for obj in pbar:
                    pbar.write("[{}]".format(os.path.dirname(obj.path_normalized())))

                    # Process complete
                    f_in = obj.path_c()
                    f_out = obj.path_c_uniform_sdf()
                    if os.path.exists(f_in) and (
                        not os.path.exists(f_out) or overwrite
                    ):
                        executor.graceful_submit(
                            compute_sdf_uni.process,
                            f_in=f_in,
                            f_out=f_out,
                        )

                    for idx in range(num_breaks):
                        # Process broken
                        f_in = obj.path_b(idx)
                        f_out = obj.path_b_uniform_sdf(idx)
                        if os.path.exists(f_in) and (
                            not os.path.exists(f_out) or overwrite
                        ):
                            executor.graceful_submit(
                                compute_sdf_uni.process,
                                f_in=f_in,
                                f_out=f_out,
                            )

                        # Process restoration
                        f_in = obj.path_r(idx)
                        f_out = obj.path_r_uniform_sdf(idx)
                        if os.path.exists(f_in) and (
                            not os.path.exists(f_out) or overwrite
                        ):
                            executor.graceful_submit(
                                compute_sdf_uni.process,
                                f_in=f_in,
                                f_out=f_out,
                            )

                executor.graceful_finish()

                # Count the number of successful objects
                object_counter["UNIFORM_SDF"][1] = len(object_id_list) * num_breaks
                for obj in object_id_list:
                    for idx in range(num_breaks):
                        if (
                            os.path.exists(obj.path_c_uniform_sdf())
                            and os.path.exists(obj.path_b_uniform_sdf(idx))
                            and os.path.exists(obj.path_r_uniform_sdf(idx))
                        ):
                            object_counter["UNIFORM_SDF"][0] += 1

        except KeyboardInterrupt:
            logging.info("Waiting for running processes ...")
            executor.graceful_finish()

    # Print out any errors encountered
    if len(executor.exceptions_log) > 0:
        logging.info("SUMMARY: The following errors were encountered ...")
        for k, v in executor.exceptions_log.items():
            logging.info("{}: {}".format(k, v))
    else:
        logging.info("SUMMARY: All operations completed successfully.")

    for o in ops:
        logging.info(
            "{} successfully processed {} out of {} breaks".format(
                o, object_counter[o][0], object_counter[o][1]
            )
        )


def validate_ops(ops, valid_ops):
    for op in ops:
        if op not in valid_ops:
            raise RuntimeError("Invalid operation {}".format(op))
    return ops


if __name__ == "__main__":
    valid_ops = [
        "WATERPROOF",
        "CLEAN",
        "BREAK",
        "SAMPLE",
        "SDF",
        "UNIFORM_SDF",
        "PARTIAL_SDF",
        "PRIMITIVE",
        "PRIMITIVE_ZONE",
        "SPLINE",
        "OCC",
        "UNIFORM_OCC",
        "RENDER",
        "VOXEL_32",
    ]

    parser = argparse.ArgumentParser(
        description="Applies a sequence of "
        + "transforms to all objects in a database in parallel. Upon "
        + "completion prints a summary of errors encountered during runtime."
    )
    parser.add_argument(
        dest="input",
        type=str,
        help="Location of the database. Pass the top level directory. For "
        + 'ShapeNet this would be "ShapeNet.v2". Models will be extracted '
        + "by name and by extension (ENSURE THERE ARE NO OTHER .obj FILES IN "
        + "THIS DIRECTORY).",
    )
    parser.add_argument(
        dest="splits",
        type=str,
        help=".json file path, this file will be created and will store the "
        + "ids of all objects in the training and testing split. Will be used "
        + "if preprocessing is restarted to accelerate initial steps.",
    )
    parser.add_argument(
        dest="ops",
        type=str,
        nargs="+",
        help="List of operations to apply. Supported operations are "
        + "{}.\n".format(valid_ops)
        + "",
    )
    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        default=1,
        help="Number of threads to use. This script uses multiprocessing so "
        + "it is not recommended to set this number higher than the number of "
        + "physical cores in your computer.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of training samples to testing samples that will be saved "
        + "to the split file.",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action="store_true",
        help="If passed will overwrite existing files. Else will skip existing "
        + "files.",
    )
    parser.add_argument(
        "--breaks",
        "-b",
        type=int,
        default=1,
        help="Number of breaks to generate for each object. This will only be "
        + "used if BREAK is passed.",
    )
    parser.add_argument(
        "--break_all",
        default=False,
        action="store_true",
        help="If passed will break the train and test set. Else will only break "
        + "the test set.",
    )
    parser.add_argument(
        "--break_method",
        type=str,
        default="surface-area",
        help="Which breaking method to use.",
    )
    parser.add_argument(
        "--use_tool",
        default=False,
        action="store_true",
        help="Save and compute the SDF values for the breaking tool.",
    )
    parser.add_argument(
        "--renders",
        "-r",
        type=int,
        default=4,
        help="Number of renders to generate for each object. This will only be "
        + "used if RENDER is passed.",
    )
    parser.add_argument(
        "--class_subsample",
        default=None,
        type=int,
        help="If passed, will randomly sample this many classes from the "
        + "dataset. Will override subsample flag.",
    )
    parser.add_argument(
        "--instance_subsample",
        default=None,
        type=int,
        help="If passed, will randomly sample this many instances from each "
        + "class from the dataset. Will override subsample flag.",
    )
    parser.add_argument(
        "--break_handle",
        action="store_true",
        default=False,
        help="If passed, will try and fracture only mug handles.",
    )
    parser.add_argument(
        "--reorient",
        action="store_true",
        default=False,
        help="If passed, will use PCA to reorient the model.",
    )
    parser.add_argument(
        "--max_break",
        default=0.5,
        type=float,
        help="Max amount (percentage based) of the source model to remove in a "
        + "given break. Breaks will be retried if they remove more than this "
        + "amount.",
    )
    parser.add_argument(
        "--min_break",
        default=0.3,
        type=float,
        help="Min amount (percentage based) of the source model to remove in a "
        + "given break. Breaks will be retried if they remove less than this "
        + "amount.",
    )
    parser.add_argument(
        "--outoforder",
        default=False,
        action="store_true",
        help="If passed, will shuffle the dataset before processing. Note this "
        + "will not alter the cotents of the split file. Use this option if "
        + "you plan to process data simultaneously with multiple scripts or PCS.",
    )
    logger.add_logger_args(parser)
    args = parser.parse_args()
    logger.configure_logging(args)

    validate_ops(args.ops, valid_ops)

    assert os.path.isdir(args.input), "Input directory does not exist: {}".format(
        args.input
    )
    if args.input[-1] == "/":
        args.input = args.input[:-1]

    assert not (args.use_tool and args.use_primitive)

    main(
        args.input,
        args.ops,
        args.threads,
        args.overwrite,
        args.breaks,
        args.class_subsample,
        args.instance_subsample,
        args.max_break,
        args.min_break,
        args.renders,
        args.splits,
        args.train_ratio,
        args.debug,
        args.break_all,
        args.outoforder,
        args.break_method,
        args.use_tool,
        args.break_handle,
        args.reorient,
    )
