"""
Command-line interface definitions for CSQA experiments.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Multi-Agent Cascade on CSQA with trust + belief + persuasion + "
            "agreeableness (all CLI-controlled)."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Optional JSON/YAML config file describing one or more runs.",
    )

    # topology / agents
    parser.add_argument(
        "--graph",
        choices=["star", "pure_star", "complete", "chain", "circle", "tree"],
        required=False,
        default=None,
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=6,
        help="Number of agents (nodes).",
    )
    parser.add_argument(
        "--attackers",
        type=int,
        default=0,
        help="Number of attackers (ignored if --attackers-idx is set).",
    )
    parser.add_argument(
        "--attackers-idx",
        type=str,
        default="",
        help="Comma-separated explicit attacker indices (0-based).",
    )
    parser.add_argument(
        "--placement",
        type=str,
        default="auto",
        help="For star: center|leaf|two_leaves|auto; others: auto.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="Seed/sample id used in output path and warm-up sampling.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="csqa",
        help="Dataset name (file will be loaded from ./data/{dataset}.jsonl). Default: csqa",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model name for methods.get_client().",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens per LLM response.",
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
        default="openai",
        choices=["openai", "vllm", "gemini", "blablador"],
        help="LLM backend: 'openai' for OpenAI API, 'vllm' for vLLM server, 'gemini' for Google Gemini API, 'blablador' for Helmholtz Blablador API.",
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
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of RE-GENERATE steps after initial round (total rounds = 1 + rounds). Default: 10",
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
        help="Process scenarios even if outputs already exist in the output directory.",
    )

    # trust init / adaptive
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
        help="CSV of length N: w_j global trust in speaker j (ignored if --trust-matrix).",
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
        help="Number of warm-up items to learn initial trust (accuracy-based).",
    )
    parser.add_argument(
        "--fixed-trust",
        action="store_true",
        help="Freeze trust after warm-up (no trust updates during main loop). Enables parallel processing.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel processing for main questions (only works with --fixed-trust).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum parallel workers when --parallel is enabled (default: 8).",
    )
    parser.add_argument(
        "--trust-lr",
        type=float,
        default=0.1,
        help="Learning rate for trust updates (how fast trust changes: 0=no change, 1=instant).",
    )
    parser.add_argument(
        "--trust-gamma",
        type=float,
        default=1.0,
        help="Warm-up speaker prior sharpness: w_j ∝ (eps+acc_j)^gamma.",
    )
    parser.add_argument(
        "--log-trust-csv",
        action="store_true",
        help="Save final trust matrix after each question as CSV.",
    )
    parser.add_argument(
        "--self-as-speaker",
        dest="self_as_speaker",
        action="store_true",
        default=True,
        help="Include each agent’s own vote as a speaker during aggregation (default: enabled).",
    )
    parser.add_argument(
        "--no-self-as-speaker",
        dest="self_as_speaker",
        action="store_false",
        help="Disable using an agent’s own vote as an input speaker.",
    )
    parser.add_argument(
        "--trust-threshold",
        type=float,
        default=0.0,
        help="During aggregation, ignore speakers with W[i,j] below this value.",
    )
    parser.add_argument(
        "--trust-update-mode",
        type=str,
        default="always",
        choices=["always", "prefix_fraction", "prefix_fraction_random"],
        help=(
            "Control when adaptive trust updates run: 'always' (default), "
            "'prefix_fraction' (only first N%% of questions), or "
            "'prefix_fraction_random' (first N%% plus random updates)."
        ),
    )
    parser.add_argument(
        "--trust-update-fraction",
        type=float,
        default=1.0,
        help=(
            "Fraction (0-1] of questions processed sequentially from the start that "
            "allow trust updates when using prefix-based modes. Default: 1.0."
        ),
    )
    parser.add_argument(
        "--trust-update-random-fraction",
        type=float,
        default=0.0,
        help=(
            "Additional fraction (0-1] of questions selected uniformly at random to "
            "update trust when using 'prefix_fraction_random'. Default: 0.0."
        ),
    )
    parser.add_argument(
        "--trust-update-seed",
        type=int,
        default=None,
        help=(
            "Optional seed controlling random selection of trust-update questions; "
            "defaults to --sample when omitted."
        ),
    )
    parser.add_argument(
        "--save-warmup-trust",
        action="store_true",
        help="Save warm-up initialized trust matrix CSV before main loop.",
    )
    parser.add_argument(
        "--no-trust",
        action="store_true",
        help="Disable trust weighting entirely (implies fixed trust, removes trust messaging).",
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

    # belief logging
    parser.add_argument(
        "--log-belief-csv",
        action="store_true",
        help="Save belief distributions per round as CSV (belief_logs).",
    )

    parser.add_argument(
        "--scenario",
        type=str,
        required=False,
        default=None,
        help="Scenario folder name (e.g., RingStar-6a-10r-6s, complete-6a-10r-6s-low-trust)",
    )

    parser.add_argument(
        "--persuasion-levels",
        type=str,
        default="",
        help=(
            "Comma-separated persuasion levels per agent: high,medium,low (length N). "
            "Adds prompt-based persuasion traits."
        ),
    )

    parser.add_argument(
        "--agreeableness-levels",
        type=str,
        default="",
        help=(
            "Comma-separated agreeableness levels per agent: high,medium,low (length N). "
            "Adds prompt-based agreeableness traits."
        ),
    )

    # FJ parametric control
    parser.add_argument(
        "--fj-gamma",
        type=str,
        default="",
        dest="fj_gamma",
        help=(
            "Comma-separated gamma (anchor pull) values per agent for FJ belief update mode. "
            "E.g. '1.0,0.0,0.0,0.0,0.0,0.0'. When set (with --fj-alpha), enables FJ mode "
            "which replaces agreeableness/persuasion trait blocks. "
            "gamma=1 anchors agent to initial belief; gamma=0 means no anchor."
        ),
    )
    parser.add_argument(
        "--fj-alpha",
        type=str,
        default="",
        dest="fj_alpha",
        help=(
            "Comma-separated alpha (self-weight) values per agent for FJ belief update mode. "
            "E.g. '1.0,0.0,0.0,0.0,0.0,0.0'. "
            "alpha=1 means agent ignores neighbors; alpha=0 means agent purely follows neighbors."
        ),
    )
    parser.add_argument(
        "--fj-w",
        type=str,
        default="",
        dest="fj_w",
        help=(
            "JSON-encoded list of per-agent neighbor weight dicts for FJ mode. "
            "Keys are neighbor agent indices (strings), values are float weights. "
            "E.g. '[{\"1\":0.2,\"2\":0.2},{\"0\":1.0},...]'. "
            "In YAML configs, use a list of dicts directly."
        ),
    )

    parser.add_argument(
        "--vote-only",
        action="store_true",
        help=(
            "Enable vote-only mode: agents share only their answer vote (A/B/C/D/E) "
            "without reasons or explanations."
        ),
    )

    # MLflow tracking
    parser.add_argument(
        "--mlflow-tracking",
        action="store_true",
        default=False,
        help="Enable MLflow experiment tracking (default: disabled).",
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
        help=(
            "MLflow tracking URI. If omitted, falls back to the MLFLOW_TRACKING_URI "
            "environment variable or sqlite:///mlruns.db."
        ),
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default="cascade-experiments",
        help="MLflow experiment name (default: cascade-experiments).",
    )

    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


__all__ = ["build_parser", "parse_args"]
