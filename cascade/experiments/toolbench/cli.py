"""
Command-line interface definitions for ToolBench implicit 5-option experiments.

Mirrors the CSQA CLI structure with ToolBench-specific defaults.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Multi-Agent Cascade on ToolBench implicit 5-option tasks "
            "with trust + belief + persuasion + agreeableness (all CLI-controlled)."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Optional JSON/YAML config file describing one or more runs.",
    )

    # ---- dataset / data file ------------------------------------------------
    parser.add_argument(
        "--dataset",
        type=str,
        default="toolbench_implicit",
        help=(
            "Dataset name used in the output directory hierarchy. "
            "Default: toolbench_implicit"
        ),
    )
    parser.add_argument(
        "--dataset-file",
        type=str,
        default="data/toolbench/implicit_5option/all_implicit_5option.json",
        help=(
            "Path to the ToolBench implicit 5-option JSON data file. "
            "Default: data/toolbench/implicit_5option/all_implicit_5option.json"
        ),
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=0,
        help=(
            "Limit the number of tasks to process (0 = all tasks). "
            "Useful for quick testing. Default: 0 (process all)."
        ),
    )

    # ---- topology / agents --------------------------------------------------
    parser.add_argument(
        "--graph",
        choices=["star", "pure_star", "complete", "chain", "circle", "tree"],
        required=False,
        default="pure_star",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=6,
        help="Number of agents (nodes). Default: 6",
    )
    parser.add_argument(
        "--attackers",
        type=int,
        default=1,
        help="Number of attackers (ignored if --attackers-idx is set). Default: 1",
    )
    parser.add_argument(
        "--attackers-idx",
        type=str,
        default="",
        help="Comma-separated explicit attacker indices (0-based). If not set, placement is used.",
    )
    parser.add_argument(
        "--placement",
        type=str,
        default="center",
        help="For star: center|leaf|two_leaves|auto; others: auto. Default: center",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="Seed/sample id used in output path and warm-up sampling.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3-flash-preview",
        help="Model name. Default: gemini-3-flash-preview",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens per LLM response. Default: 2048",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature for LLM sampling. Default: None (use model's API default).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gemini",
        choices=["openai", "vllm", "gemini", "blablador"],
        help="LLM backend. Default: gemini",
    )
    parser.add_argument(
        "--model-dir-name",
        type=str,
        default="",
        help=(
            "Clean model name used for the output directory (e.g. 'gpt-oss-120b'). "
            "Defaults to --model if not specified. Useful when the API model ID "
            "contains spaces or special characters."
        ),
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL for vLLM server (default: http://localhost:8000/v1).",
    )
    parser.add_argument(
        "--auto-launch-vllm",
        action="store_true",
        help="Automatically launch a local vLLM server if backend is 'vllm'.",
    )
    parser.add_argument(
        "--vllm-gpu",
        type=int,
        default=None,
        help="Preferred GPU index for auto-launched vLLM server.",
    )
    parser.add_argument(
        "--vllm-port",
        type=int,
        default=None,
        help="Preferred port for auto-launched vLLM server.",
    )
    parser.add_argument(
        "--vllm-port-base",
        type=int,
        default=8000,
        help="Starting port when searching for a free vLLM port.",
    )
    parser.add_argument(
        "--vllm-log-dir",
        type=str,
        default="logs/vllm",
        help="Log directory for auto-launched vLLM servers.",
    )
    parser.add_argument(
        "--vllm-gpu-utilization",
        type=float,
        default=None,
        help="Pass through to vLLM --gpu-memory-utilization when auto launching.",
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=None,
        help="Pass through to vLLM --max-model-len when auto launching.",
    )
    parser.add_argument(
        "--vllm-extra-args",
        type=str,
        default="",
        help="Extra CLI args for vLLM server when auto launching.",
    )
    parser.add_argument(
        "--vllm-wait-seconds",
        type=int,
        default=120,
        help="Seconds to wait for auto-launched vLLM server to become ready.",
    )

    # ---- rounds / output ----------------------------------------------------
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of RE-GENERATE steps after initial round (total = 1 + rounds). Default: 10",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help=(
            "Path to a log file. When set, all informational output (print statements) "
            "is redirected to this file; only the tqdm progress bar remains on the terminal."
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="(Ignored) parallel threads; forced to 1.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Save pretty JSON records (else Python repr per line).",
    )
    parser.set_defaults(skip_existing=True)
    parser.add_argument(
        "--rerun-existing",
        action="store_false",
        dest="skip_existing",
        help="Process scenarios even if outputs already exist.",
    )

    # ---- trust init / adaptive ----------------------------------------------
    parser.add_argument(
        "--trust-matrix",
        type=str,
        default="",
        help="Path to NxN CSV; W[i,j] = listener i trust in speaker j.",
    )
    parser.add_argument(
        "--speaker-trust",
        type=str,
        default="",
        help="CSV of length N: w_j global trust in speaker j.",
    )
    parser.add_argument(
        "--trust-scheme",
        type=str,
        default="uniform",
        choices=["uniform", "hub_high", "hub_low", "random"],
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of warm-up items to learn initial trust.",
    )
    parser.add_argument(
        "--fixed-trust",
        action="store_true",
        help="Freeze trust after warm-up (no updates during main loop).",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel processing (only works with --fixed-trust).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum parallel workers when --parallel is enabled.",
    )
    parser.add_argument(
        "--trust-lr",
        type=float,
        default=0.1,
        help="Learning rate for trust updates.",
    )
    parser.add_argument(
        "--trust-gamma",
        type=float,
        default=1.0,
        help="Warm-up speaker prior sharpness.",
    )
    parser.add_argument(
        "--log-trust-csv",
        action="store_true",
        help="Save trust matrix after each question as CSV.",
    )
    parser.add_argument(
        "--self-as-speaker",
        dest="self_as_speaker",
        action="store_true",
        default=True,
        help="Include each agent's own vote as a speaker (default: enabled).",
    )
    parser.add_argument(
        "--no-self-as-speaker",
        dest="self_as_speaker",
        action="store_false",
        help="Disable using an agent's own vote as speaker.",
    )
    parser.add_argument(
        "--trust-threshold",
        type=float,
        default=0.0,
        help="Ignore speakers with W[i,j] below this value.",
    )
    parser.add_argument(
        "--trust-update-mode",
        type=str,
        default="always",
        choices=["always", "prefix_fraction", "prefix_fraction_random"],
        help="Control when adaptive trust updates run.",
    )
    parser.add_argument(
        "--trust-update-fraction",
        type=float,
        default=1.0,
        help="Fraction of questions that update trust in prefix-based modes.",
    )
    parser.add_argument(
        "--trust-update-random-fraction",
        type=float,
        default=0.0,
        help="Additional random fraction for trust updates.",
    )
    parser.add_argument(
        "--trust-update-seed",
        type=int,
        default=None,
        help="Seed for random trust-update selection.",
    )
    parser.add_argument(
        "--save-warmup-trust",
        action="store_true",
        help="Save warm-up initialized trust matrix CSV.",
    )
    parser.add_argument(
        "--no-trust",
        action="store_true",
        help="Disable trust weighting entirely.",
    )
    parser.add_argument(
        "--adaptive-attacker",
        action="store_true",
        help=(
            "Enable adaptive (sleeper) attacker: during warmup the attacker "
            "behaves as a benign agent to build high trust, then switches to "
            "attacking in the main run. Requires --warmup > 0 and trust enabled."
        ),
    )

    # ---- belief logging -----------------------------------------------------
    parser.add_argument(
        "--log-belief-csv",
        action="store_true",
        help="Save belief distributions per round as CSV.",
    )

    # ---- scenario -----------------------------------------------------------
    parser.add_argument(
        "--scenario",
        type=str,
        required=False,
        default=None,
        help="Scenario folder name for output directory hierarchy.",
    )

    # ---- trait levels -------------------------------------------------------
    parser.add_argument(
        "--persuasion-levels",
        type=str,
        default="",
        help="Comma-separated persuasion levels per agent (high,medium,low + _v2/_v3).",
    )
    parser.add_argument(
        "--agreeableness-levels",
        type=str,
        default="",
        help="Comma-separated agreeableness levels per agent (high,medium,low + _v2/_v3).",
    )
    parser.add_argument(
        "--vote-only",
        action="store_true",
        help="Vote-only mode: agents share only answer votes without reasons.",
    )

    # ---- MLflow tracking ----------------------------------------------------
    parser.add_argument(
        "--mlflow-tracking",
        action="store_true",
        default=False,
        help="Enable MLflow experiment tracking.",
    )
    parser.add_argument(
        "--no-mlflow-tracking",
        action="store_false",
        dest="mlflow_tracking",
        help="Disable MLflow experiment tracking (default).",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default=None,
        help="MLflow tracking URI.",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default="cascade-toolbench",
        help="MLflow experiment name. Default: cascade-toolbench",
    )

    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


__all__ = ["build_parser", "parse_args"]