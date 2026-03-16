"""
Modular ToolBench experiment components.

Legacy runners (runner.py, runner_5tools.py, runner_implicit_5option.py) are
kept for backward compatibility.  The production-grade experiment pipeline
lives in agents.py / cli.py / runner_final.py and mirrors the CSQA module
architecture.
"""

from .agents import Agent, AgentGraph, build_agent_system_prompts, build_tasks_for_item
from .cli import build_parser, parse_args
from .runnerTB import (
    build_output_filename,
    load_toolbench_dataset,
    process_item_and_update_trust,
    run_dataset,
    run_from_namespace,
)

from cascade.experiments.logging import RunRecorder, compute_accuracy, log_mlflow_metrics_per_item
from cascade.experiments.trust import build_speaker_trust, load_trust_matrix, speaker_vector_to_matrix

__all__ = [
    "Agent",
    "AgentGraph",
    "RunRecorder",
    "build_parser",
    "parse_args",
    "build_output_filename",
    "load_toolbench_dataset",
    "process_item_and_update_trust",
    "run_dataset",
    "run_from_namespace",
    "build_agent_system_prompts",
    "build_tasks_for_item",
    "compute_accuracy",
    "log_mlflow_metrics_per_item",
    "build_speaker_trust",
    "load_trust_matrix",
    "speaker_vector_to_matrix",
]