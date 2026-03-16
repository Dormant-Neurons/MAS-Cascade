"""
Batch execution helpers for Cascade experiments.
"""

from __future__ import annotations

import argparse
from typing import Dict, Iterable, Mapping, Optional

from .csqa import build_parser, run_from_namespace
from .toolbench import build_parser as tb_build_parser, run_from_namespace as tb_run_from_namespace


def _collect_defaults(parser: argparse.ArgumentParser) -> Dict[str, object]:
    defaults: Dict[str, object] = {}
    for action in parser._actions:
        if not action.dest or action.dest == argparse.SUPPRESS:
            continue
        defaults[action.dest] = action.default
    return defaults


def _materialize_namespace(
    base_values: Mapping[str, object],
    config: Mapping[str, object],
    overrides: Optional[Mapping[str, object]] = None,
) -> argparse.Namespace:
    merged: Dict[str, object] = dict(base_values)
    merged.update(config)
    if overrides:
        merged.update(overrides)
    return argparse.Namespace(**merged)


def run_configs(
    configs: Iterable[Mapping[str, object]],
    *,
    overrides: Optional[Mapping[str, object]] = None,
    parser: Optional[argparse.ArgumentParser] = None,
) -> None:
    """
    Execute a sequence of run configurations.

    Args:
        configs: Iterable of dictionaries matching the CLI argument names.
        overrides: Optional mapping applied on top of every configuration.
        parser: Optional custom parser; defaults to the CSQA parser.
    """
    parser = parser or build_parser()
    defaults = _collect_defaults(parser)

    # If scenario is specified in overrides, use it as a filter (not an override)
    scenario_filter = None
    if overrides and 'scenario' in overrides:
        scenario_filter = overrides.get('scenario')
        # Remove scenario from overrides so it doesn't override the config value
        overrides = dict(overrides)
        del overrides['scenario']

    for config in configs:
        # Skip configs that don't match the scenario filter
        if scenario_filter is not None:
            config_scenario = config.get('scenario')
            if config_scenario != scenario_filter:
                continue

        dataset = config.get("dataset", "")
        if isinstance(dataset, str) and "toolbench" in dataset.lower():
            tb_parser = tb_build_parser()
            tb_defaults = _collect_defaults(tb_parser)
            tb_namespace = _materialize_namespace(tb_defaults, config, overrides)
            tb_run_from_namespace(tb_namespace)
        else:
            namespace = _materialize_namespace(defaults, config, overrides)
            run_from_namespace(namespace)
