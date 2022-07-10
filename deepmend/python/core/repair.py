import os
import argparse
import logging
import trimesh
import subprocess

import core


def repair_watertight_handsoff(
    f_in,
    f_out,
    timeout=None,
    verbose=False,
):

    cmd = ["python" + " " + __file__ + " " + f_in + " " + f_out]

    if verbose:
        cmd[0] += " --debug"

    # Badness, but prevents segfault
    logging.debug("Executing command in the shell: \n{}".format(cmd))
    try:
        subprocess.call(cmd, shell=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        logging.debug("Repair failed")
    
    if not os.path.exists(f_out):
        raise core.errors.IsosurfaceExtractionError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(dest="input", type=str, help="Path to the input file.")
    parser.add_argument(dest="output", type=str, help="Path to the output file.")
    core.add_common_args(parser)
    args = parser.parse_args()
    core.configure_logging(args)
    mesh = core.repair_watertight(
        core.loader(args.input),
    )
    core.saver(args.output, mesh)
