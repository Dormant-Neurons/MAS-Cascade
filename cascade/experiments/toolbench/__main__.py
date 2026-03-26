"""Allow running ToolBench experiments via: python -m cascade.experiments.toolbench [args]"""

import contextlib
import logging
import os

logging.getLogger("httpx").setLevel(logging.WARNING)

from cascade.experiments.toolbench.cli import parse_args
from cascade.experiments.toolbench.runnerTB import run_from_namespace

args = parse_args()
log_file = getattr(args, "log_file", None)

if log_file:
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as lf:
        with contextlib.redirect_stdout(lf):
            run_from_namespace(args)
else:
    run_from_namespace(args)
