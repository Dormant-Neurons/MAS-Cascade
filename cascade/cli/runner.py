"""
CLI entry point for running CSQA and ToolBench experiments.
"""

from __future__ import annotations

import argparse
from typing import Dict

from cascade.experiments.batch import run_configs
from cascade.experiments.config import load_and_normalize
from cascade.experiments.csqa import (
    Agent,
    AgentGraph,
    build_output_filename,
    build_parser,
    run_from_namespace,
)


def _collect_cli_overrides(parser: argparse.ArgumentParser, args: argparse.Namespace) -> Dict[str, object]:
    overrides: Dict[str, object] = {}
    for action in parser._actions:
        dest = action.dest
        if not dest or dest == argparse.SUPPRESS or dest == "config":
            continue
        default = action.default
        value = getattr(args, dest, default)
        if value != default:
            overrides[dest] = value
    return overrides


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = getattr(args, "config", "") or ""

    if config_path:
        overrides = _collect_cli_overrides(parser, args)
        overrides.setdefault("config", config_path)
        configs = load_and_normalize(config_path)
        if not configs:
            raise SystemExit("ERROR: configuration file is empty.")
        run_configs(configs, overrides=overrides, parser=parser)
    else:
        run_from_namespace(args)


if __name__ == "__main__":
    main()
