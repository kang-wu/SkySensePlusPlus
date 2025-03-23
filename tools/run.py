from antmmf.utils.env import setup_compatibility
from antmmf.utils.flags import flags
from antmmf.run import plain_run

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from lib import *  # noqa  make sure all modules have been registered.

usage = """
    Usage:
        python tools/run.py --config configs/foo/bar.yml [OPTIONS] [OPTS]

    Options:
        --config_override override.yml   configurations from this file will override the --config one. like
            python tools/run.py --config configs/foo/bar.yml --config_override my_foobar.yml

        --local_rank local rank of your machine, used in parallel mode

        OPTS: override specific value in config, like
            python tools/run.py --config configs/foo/bar.yml \\
                training_parameters.device cuda:0 \\
                training_parameters.max_epochs 5 \\
                task_attributes.hateful_memes.dataset_attributes.foo.images.train \\
                "[foo/defaults/images]"

    Priority:
        OPTS OVERRIDE --config_override OVERRIDE --config, see antmmf/common/build.py::build_config for details
"""


def run():
    parser = flags.get_parser()
    try:
        args = parser.parse_args()
        plain_run(args)
    except SystemExit:
        exit(2)


if __name__ == "__main__":
    setup_compatibility()
    run()