"""
High-level orchestration for CSQA cascade experiments.
"""

from __future__ import annotations

import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

import csv
import math
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import mlflow
import numpy as np
from tqdm import tqdm

try:
    from sqlalchemy.engine import url as sa_url
except ImportError:  # pragma: no cover - optional dependency
    sa_url = None

import json as _json

from cascade.core import methods, prompts
from cascade.core.utils import parse_int_list, parse_level_list, slugify

from .agents import Agent, AgentGraph, build_agent_system_prompts, build_tasks_for_item
from cascade.experiments.logging import RunRecorder, log_mlflow_metrics_per_item
from cascade.experiments.trust import load_trust_matrix, speaker_vector_to_matrix


def _parse_fj_params(args) -> list | None:
    """Parse --fj-gamma / --fj-alpha / --fj-w from an argparse namespace.

    Returns a list of per-agent dicts
    {"gamma": float, "alpha": float, "w": {str_idx: float}}
    or None when FJ mode is not enabled.
    """
    gamma_raw = getattr(args, "fj_gamma", None)
    alpha_raw = getattr(args, "fj_alpha", None)
    w_raw = getattr(args, "fj_w", None)

    # Helper: convert either a list (from YAML) or a comma-separated string (from CLI) to floats
    def _to_float_list(val):
        if not val and val != 0:
            return []
        if isinstance(val, list):
            return [float(x) for x in val]
        s = str(val).strip()
        if not s:
            return []
        return [float(x.strip()) for x in s.split(",") if x.strip()]

    gammas = _to_float_list(gamma_raw)
    alphas = _to_float_list(alpha_raw)

    if not gammas and not alphas:
        return None  # FJ mode not requested

    # w_raw may be: None/empty str (not set), a list of dicts (from YAML), or a JSON string (CLI)
    if not w_raw and w_raw != 0:
        ws: list = []
    elif isinstance(w_raw, list):
        ws = w_raw
    else:
        try:
            ws = _json.loads(str(w_raw))
        except Exception:
            ws = []

    n = max(len(gammas), len(alphas), len(ws))
    if n == 0:
        return None

    result = []
    for i in range(n):
        w_dict = {}
        if i < len(ws) and ws[i]:
            w_dict = {str(k): float(v) for k, v in ws[i].items()}
        result.append({
            "gamma": gammas[i] if i < len(gammas) else 0.0,
            "alpha": alphas[i] if i < len(alphas) else 0.5,
            "w": w_dict,
        })
    return result


def build_output_filename(args, attacker_ids=None):
    """
    <graph>_<agents>_<attackers>[_idx-<dash-joined-ids>].output
    No dataset or scenario in the name.
    """

    graph = getattr(args, "graph", None) or getattr(args, "graph_type", None) or "graph"
    graph = slugify(graph)
    agents = int(getattr(args, "agents", getattr(args, "num_agents", 0)))
    attackers = int(getattr(args, "attackers", getattr(args, "num_attackers", 0)))

    parts = [f"{graph}", f"{agents}", f"{attackers}"]

    if attacker_ids is not None:
        if isinstance(attacker_ids, (list, tuple)):
            idxs = "-".join(str(i) for i in attacker_ids)
        else:
            idxs = str(attacker_ids)
        parts.append(f"idx-{idxs}")

    return "_".join(parts) + ".output"


def process_item_and_update_trust(
    data: dict,
    ds_name: str,
    num_agents: int,
    reg_rounds: int,
    adj_matrix: np.ndarray,
    attacker_idx: Sequence[int],
    system_prompts: Sequence[str],
    model: str,
    max_tokens: int,
    temperature: float | None,
    save_json: bool,
    run_mode: str,
    sample_id: int,
    trust_matrix_ref: dict,
    fixed_trust: bool,
    trust_lr: float,
    log_trust_csv: bool,
    log_belief_csv: bool,
    self_as_speaker: bool,
    trust_threshold: float,
    trust_enabled: bool,
    persuasion_levels: Sequence[str] | None,
    agreeableness_levels: Sequence[str] | None,
    vote_only: bool = False,
    backend: str = "openai",
    vllm_url: str | None = None,
    mlflow_enabled: bool = False,
    item_index: int = 0,
    summary_recorder: RunRecorder | None = None,
    agent_labels: Sequence[str] | None = None,
    trust_update: bool = True,
    trust_stage: str | None = None,
    fj_params: list | None = None,
    *,
    scenario: str,
    model_name: str,
    seed_str: str,
    output_root: str,
):
    question = data["question"]
    choices = data["choices"]
    answer_key = data["answerKey"]
    task_id = data["id"]

    tasks = build_tasks_for_item(num_agents, attacker_idx, question, choices, answer_key)

    current_trust = trust_matrix_ref["W"].copy()
    current_error_state = trust_matrix_ref.get("error_state")
    agent_graph = AgentGraph(
        num_agents,
        adj_matrix,
        system_prompts,
        tasks,
        task_id,
        trust_matrix=current_trust,
        trust_enabled=trust_enabled,
        attacker_idx=attacker_idx,
        model_type=model,
        max_tokens=max_tokens,
        temperature=temperature,
        trust_lr=trust_lr,
        self_as_speaker=self_as_speaker,
        trust_threshold=trust_threshold,
        persuasion_levels=persuasion_levels,
        agreeableness_levels=agreeableness_levels,
        vote_only=vote_only,
        backend=backend,
        vllm_url=vllm_url,
        trust_error_state=current_error_state,
        fj_params=fj_params,
    )

    out_dir = Path(output_root) / model_name / ds_name / scenario / seed_str
    filename = f"{ds_name}_{run_mode}.output"

    agent_graph.first_generate()
    agent_graph.snapshot_beliefs("round0")
    round0_answers = {j: agent_graph.Agents[j].last_response.get("answer", "") for j in range(num_agents)}

    for r in range(reg_rounds):
        agent_graph.re_generate()
        agent_graph.snapshot_beliefs(f"round{r + 1}")

    should_update_trust = trust_enabled and not fixed_trust and trust_update
    if should_update_trust:
        agent_graph.update_trust_after_question(answer_key, round0_answers=round0_answers)
        trust_matrix_ref["W"] = agent_graph.trust_matrix.copy()
        trust_matrix_ref["error_state"] = agent_graph._trust_error_state.copy()

    if mlflow_enabled:
        log_mlflow_metrics_per_item(
            task_id,
            agent_graph,
            answer_key,
            step=item_index,
            agent_labels=agent_labels,
            trust_enabled=trust_enabled,
        )

    agent_graph.record["trust_update_applied"] = bool(should_update_trust)
    agent_graph.record["answer_key"] = answer_key
    if trust_stage:
        agent_graph.record["trust_stage"] = trust_stage

    out_path = out_dir / filename
    print(f"Output -> {out_path}")
    records_dir = out_dir / "records"
    agent_graph.save(
        str(out_path),
        save_json,
        item_index=item_index,
        split_dir=records_dir,
    )

    if summary_recorder is not None:
        summary_recorder.record_item(
            item_index=item_index,
            item_id=task_id,
            question=question,
            choices=choices,
            correct_answer=answer_key,
            agent_graph=agent_graph,
            belief_snapshots=agent_graph.record.get("belief_trajectory", []),
            trust_init=current_trust.copy() if trust_enabled else None,
            trust_final=agent_graph.trust_matrix.copy() if trust_enabled else None,
            trust_stage=trust_stage
            if trust_stage is not None
            else ("train_full" if should_update_trust else "eval_frozen"),
        )

    if trust_enabled and log_trust_csv:
        logs_dir = os.path.join(out_dir, "trust_logs")
        methods.create_directory(logs_dir)
        csv_path = os.path.join(logs_dir, f"trust_after_item_{task_id}.csv")
        np.savetxt(csv_path, agent_graph.trust_matrix, delimiter=",")

    if log_belief_csv:
        beliefs_dir = os.path.join(out_dir, "belief_logs")
        methods.create_directory(beliefs_dir)
        labels = ["A", "B", "C", "D", "E"]
        for k, snap in enumerate(agent_graph.record.get("belief_trajectory", [])):
            mat = np.array(snap["beliefs"], dtype=float)
            csv_path = os.path.join(
                beliefs_dir, f"belief_after_item_{task_id}_round{k}.csv"
            )
            np.savetxt(
                csv_path,
                mat,
                delimiter=",",
                header=",".join(labels),
                comments="",
            )


def run_dataset(
    ds_name: str,
    sample_id: int,
    attacker_idx: Sequence[int],
    graph_type: str,
    model: str,
    max_tokens: int,
    temperature: float | None,
    threads: int,
    num_agents: int,
    save_json: bool,
    reg_rounds: int,
    trust_matrix: np.ndarray,
    trust_enabled: bool,
    warmup: int,
    fixed_trust: bool,
    parallel: bool,
    max_workers: int,
    trust_lr: float,
    trust_gamma: float,
    log_trust_csv: bool,
    log_belief_csv: bool,
    self_as_speaker: bool,
    trust_threshold: float,
    trust_update_mode: str = "always",
    trust_update_fraction: float = 1.0,
    trust_update_random_fraction: float = 0.0,
    trust_update_seed: int | None = None,
    save_warmup_trust: bool = False,
    adaptive_attacker: bool = False,
    persuasion_levels: Sequence[str] | None = None,
    agreeableness_levels: Sequence[str] | None = None,
    vote_only: bool = False,
    backend: str = "openai",
    vllm_url: str | None = None,
    mlflow_tracking: bool = False,
    mlflow_uri: str | None = None,
    mlflow_experiment: str = "cascade-experiments",
    fj_params: list | None = None,
    *,
    scenario: str | None = None,
    output_root: str = "output",
    args_namespace=None,
    model_dir_name: str | None = None,
):
    args_ns = args_namespace
    scenario = scenario or (getattr(args_ns, "scenario", None) if args_ns else None)
    if not scenario:
        raise SystemExit("ERROR: scenario must be provided to run_dataset.")
    seed_str = str(sample_id)

    # Resolve the directory name for the model (clean name for output paths)
    model_dir_name = model_dir_name or model

    resolved_mlflow_uri: str | None = None

    if mlflow_tracking:
        env_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if mlflow_uri:
            resolved_mlflow_uri = mlflow_uri
            os.environ["MLFLOW_TRACKING_URI"] = resolved_mlflow_uri
        elif env_uri:
            resolved_mlflow_uri = env_uri
        else:
            resolved_mlflow_uri = "sqlite:///mlruns.db"
            os.environ.setdefault("MLFLOW_TRACKING_URI", resolved_mlflow_uri)

        if resolved_mlflow_uri and resolved_mlflow_uri.startswith("sqlite"):
            db_path = None
            if sa_url is not None:
                try:
                    db_path = sa_url.make_url(resolved_mlflow_uri).database
                except Exception:
                    db_path = None
            if not db_path:
                from urllib.parse import urlparse

                parsed = urlparse(resolved_mlflow_uri)
                path_part = parsed.path or ""
                if parsed.netloc:
                    path_part = f"{parsed.netloc}{path_part}"
                if path_part.startswith("//"):
                    db_path = os.path.normpath(path_part)
                else:
                    db_path = path_part.lstrip("/")

            if db_path and db_path != ":memory:":
                if not os.path.isabs(db_path):
                    db_path = os.path.abspath(db_path)
                db_dir = os.path.dirname(db_path)
                if db_dir and not os.path.exists(db_dir):
                    os.makedirs(db_dir, exist_ok=True)

        mlflow.set_tracking_uri(resolved_mlflow_uri)
        os.environ.pop("MLFLOW_EXPERIMENT_ID", None)
        mlflow.set_experiment(mlflow_experiment)
        model_slug = slugify(model) if model else "model"
        if seed_str:
            run_name = f"{model_slug}_{scenario}_seed{seed_str}"
        else:
            run_name = f"{model_slug}_{scenario}"
        mlflow.start_run(run_name=run_name)

        mlflow.log_param("dataset", ds_name)
        mlflow.log_param("graph_type", graph_type)
        mlflow.log_param("num_agents", num_agents)
        mlflow.log_param("num_attackers", len(attacker_idx))
        mlflow.log_param("attacker_indices", str(attacker_idx))
        mlflow.log_param("model", model)
        mlflow.log_param("backend", backend)
        mlflow.log_param("max_tokens", max_tokens)
        mlflow.log_param("temperature", temperature)
        mlflow.log_param("rounds", reg_rounds)
        mlflow.log_param("warmup", warmup)
        mlflow.log_param("sample_seed", sample_id)
        mlflow.log_param("fixed_trust", fixed_trust)
        mlflow.log_param("parallel", parallel)
        mlflow.log_param("max_workers", max_workers)
        mlflow.log_param("trust_lr", trust_lr)
        mlflow.log_param("trust_gamma", trust_gamma)
        mlflow.log_param("self_as_speaker", self_as_speaker)
        mlflow.log_param("trust_threshold", trust_threshold)
        mlflow.log_param("trust_enabled", trust_enabled)
        mlflow.log_param("vote_only", vote_only)
        if persuasion_levels:
            mlflow.log_param("persuasion_levels", str(persuasion_levels))
        if agreeableness_levels:
            mlflow.log_param("agreeableness_levels", str(agreeableness_levels))
        if backend == "vllm":
            mlflow.log_param("vllm_url", vllm_url)

        mlflow.set_tags(
            {
                "graph_type": graph_type,
                "num_agents": str(num_agents),
                "num_attackers": str(len(attacker_idx)),
                "backend": backend,
                "scenario": scenario,
            }
        )

        print(f"[MLflow] Experiment: {mlflow_experiment}")
        print(f"[MLflow] Run ID: {mlflow.active_run().info.run_id}")

    try:
        adj_matrix = methods.generate_adj(num_agents, graph_type)

        mode = f"{graph_type}_{num_agents}_{len(attacker_idx)}"
        if attacker_idx:
            idx_tag = "-".join(map(str, attacker_idx))
            mode += f"_idx-{idx_tag}"
        if trust_enabled:
            mode += "_trust_adapt" if not fixed_trust else "_trust_fixed"
        else:
            mode += "_notrust"

        scenario_dir = os.path.join(output_root, model_dir_name, ds_name, scenario)
        seed_dir = os.path.join(scenario_dir, seed_str)
        os.makedirs(seed_dir, exist_ok=True)

        out_dir = seed_dir
        agent_labels = [
            f"Attacker_{idx}" if idx in attacker_idx else f"Agent_{idx}"
            for idx in range(num_agents)
        ]
        summary_recorder = RunRecorder(
            dataset=ds_name,
            scenario=scenario,
            sample_id=sample_id,
            num_agents=num_agents,
            attacker_idx=attacker_idx,
            agent_labels=agent_labels,
        )

        params_csv = os.path.join(scenario_dir, "params.csv")

        param_row = {}
        if args_ns is not None:
            param_row.update(vars(args_ns))
        else:
            param_row.update(
                {
                    "dataset": ds_name,
                    "graph": graph_type,
                    "agents": num_agents,
                    "attackers": len(attacker_idx),
                    "attackers_idx": ",".join(map(str, attacker_idx)),
                    "model": model,
                    "max_tokens": max_tokens,
                    "rounds": reg_rounds,
                    "warmup": warmup,
                    "trust_lr": trust_lr,
                    "trust_gamma": trust_gamma,
                    "self_as_speaker": self_as_speaker,
                    "trust_threshold": trust_threshold,
                    "vote_only": vote_only,
                }
            )
        param_row["sample_id"] = sample_id
        param_row["scenario"] = scenario
        param_row["backend"] = backend
        param_row["vllm_url"] = vllm_url
        param_row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        param_row["seed"] = seed_str
        param_row["trust_enabled"] = trust_enabled

        fieldnames = ["timestamp", "seed"] + sorted(
            key for key in param_row.keys() if key not in {"timestamp", "seed"}
        )
        write_header = not os.path.exists(params_csv)
        with open(params_csv, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(param_row)

        dataset_path = os.path.join("data", f"{ds_name}.jsonl")
        dataset_iter = methods.get_dataset(dataset_path)
        dataset = list(dataset_iter)
        # Normalize toolbench format to CSQA format
        if dataset and "query" in dataset[0]:
            _labels = ["A", "B", "C", "D", "E"]
            normalized = []
            for entry in dataset:
                options = entry.get("options", [])
                while len(options) < 5:
                    options.append("Unknown Tool")
                options = options[:5]
                normalized.append({
                    "id": str(entry["task_id"]),
                    "question": entry["query"],
                    "choices": {"label": _labels[:], "text": options},
                    "answerKey": entry["correct_answer"],
                })
            dataset = normalized
        print(f"[INFO] Loaded {len(dataset)} questions from {ds_name}")

        trust_guidance = ""
        if trust_enabled and not fixed_trust:
            trust_guidance = getattr(prompts, "trust_guidance_block", "") or ""

        system_prompts = build_agent_system_prompts(
            num_agents,
            attacker_idx,
            persuasion_levels,
            agreeableness_levels,
            trust_guidance=trust_guidance,
            fj_params=fj_params,
        )

        print(
            f"Graph: {graph_type}_{num_agents} | Attackers: {attacker_idx} "
            f"| Total rounds: {1 + reg_rounds} | Sample: {sample_id}"
        )
        if trust_enabled:
            print("Using adaptive asymmetric trust W[i,j] (listener i trusts speaker j).")
        else:
            print("Uniform weighting: agents share views without adaptive weights.")
        if persuasion_levels:
            print(f"Persuasion levels: {persuasion_levels}")
        if agreeableness_levels:
            print(f"Agreeableness levels: {agreeableness_levels}")
        print(f"Output -> {out_dir}/{ds_name}_{mode}.output")

        if trust_enabled and warmup > 0 and len(dataset) > 0:
            total = len(dataset)
            k = min(warmup, total)
            rng = random.Random(sample_id)
            warm_idx = sorted(rng.sample(range(total), k))
            warm_subset = [dataset[i] for i in warm_idx]
            with open(os.path.join(out_dir, "warmup_indices.txt"), "w") as fh:
                fh.write(",".join(map(str, warm_idx)) + "\n")

            # Adaptive attacker: during warmup the attacker defends the CORRECT
            # answer to guarantee high accuracy and build maximum trust.
            if adaptive_attacker:
                warmup_attacker_idx = attacker_idx
                warmup_system_prompts = system_prompts  # uses attacker system prompt
                warmup_defends_correct = True
                print(
                    f"[Adaptive Attacker] Warmup: attacker agents {list(attacker_idx)} "
                    f"defend the CORRECT answer to build high trust."
                )
            else:
                warmup_attacker_idx = attacker_idx
                warmup_system_prompts = system_prompts
                warmup_defends_correct = False

            acc = np.zeros(num_agents, dtype=float)
            print(
                f"[Warm-up] Running {k} random items (with {reg_rounds} re-gen rounds) "
                "to initialize speaker reliability..."
            )
            for data in tqdm(warm_subset, desc="Warm-up (random K)"):
                question = data["question"]
                choices = data["choices"]
                answer_key = data["answerKey"]
                task_id = data["id"]

                tasks = build_tasks_for_item(
                    num_agents, warmup_attacker_idx, question, choices, answer_key,
                    attacker_defends_correct=warmup_defends_correct,
                )
                graph = AgentGraph(
                    num_agents,
                    adj_matrix,
                    warmup_system_prompts,
                    tasks,
                    task_id,
                    trust_matrix=trust_matrix.copy(),
                    trust_enabled=True,
                    attacker_idx=warmup_attacker_idx,
                    model_type=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    trust_lr=trust_lr,
                    self_as_speaker=self_as_speaker,
                    trust_threshold=trust_threshold,
                    persuasion_levels=persuasion_levels,
                    agreeableness_levels=agreeableness_levels,
                    backend=backend,
                    vllm_url=vllm_url,
                    fj_params=fj_params,
                )
                graph.first_generate()

                # Measure accuracy from round 0 (independent ability),
                # before deliberation can override individual answers.
                key = str(answer_key).strip().upper()
                for j in range(num_agents):
                    answer = str(graph.Agents[j].last_response.get("answer", "")).strip().upper()
                    acc[j] += 1.0 if answer == key else 0.0

                for _ in range(reg_rounds):
                    graph.re_generate()

            acc /= float(k)
            # Absolute trust: start from 0.5 base, scale by accuracy^gamma.
            # No normalization — values stay in [0, 1] independently.
            weights = np.power(np.clip(acc, 0.0, 1.0), trust_gamma)
            weights = np.clip(weights, 0.0, 1.0)

            trust_matrix = speaker_vector_to_matrix(weights)
            print("[Warm-up] Initialized trust matrix from warm-up accuracies.")

            if save_warmup_trust:
                warmup_csv = os.path.join(
                    out_dir, f"{ds_name}_{mode}.warmup_initial_trust.csv"
                )
                np.savetxt(warmup_csv, trust_matrix, delimiter=",")
                print(f"[Warm-up] Saved initial trust matrix -> {warmup_csv}")

            warm_set = set(warm_idx)
            dataset = [dataset[i] for i in range(total) if i not in warm_set]

        if not trust_enabled:
            trust_matrix = np.ones((num_agents, num_agents), dtype=float)

        trust_matrix_ref = {
            "W": trust_matrix.copy(),
            "error_state": np.zeros((num_agents, num_agents), dtype=float),
        }

        def build_trust_update_plan(total_items: int):
            plan = []
            if total_items <= 0:
                return plan
            mode = (trust_update_mode or "always").strip().lower()
            frac = max(0.0, min(1.0, trust_update_fraction))
            rand_frac = max(0.0, min(1.0, trust_update_random_fraction))
            seed_value = (
                trust_update_seed if trust_update_seed is not None else sample_id
            )
            rng = random.Random(seed_value)

            if not trust_enabled:
                return [
                    {"update": False, "stage": "trust_disabled"}
                    for _ in range(total_items)
                ]
            if fixed_trust:
                return [
                    {"update": False, "stage": "trust_frozen"}
                    for _ in range(total_items)
                ]

            if mode == "always":
                return [
                    {"update": True, "stage": "train_full"}
                    for _ in range(total_items)
                ]

            if mode not in {"prefix_fraction", "prefix_fraction_random"}:
                raise SystemExit(
                    f"ERROR: Unsupported trust_update_mode='{trust_update_mode}'."
                )

            prefix_count = int(math.ceil(frac * total_items))
            prefix_count = max(0, min(prefix_count, total_items))
            plan = [
                {"update": False, "stage": "eval_frozen"} for _ in range(total_items)
            ]
            for idx in range(prefix_count):
                plan[idx] = {"update": True, "stage": "train_prefix"}

            if mode == "prefix_fraction_random" and total_items > prefix_count:
                remaining_indices = list(range(prefix_count, total_items))
                random_count = int(math.ceil(rand_frac * total_items))
                random_count = max(0, min(random_count, len(remaining_indices)))
                if random_count > 0:
                    rng.shuffle(remaining_indices)
                    for idx in remaining_indices[:random_count]:
                        plan[idx] = {"update": True, "stage": "train_random"}

            return plan

        trust_update_plan = build_trust_update_plan(len(dataset))
        if trust_update_plan and trust_enabled:
            planned_updates = sum(1 for entry in trust_update_plan if entry["update"])
            print(
                f"[Trust] Planned updates for {planned_updates}/{len(trust_update_plan)} "
                f"questions (mode={trust_update_mode}, prefix_frac={trust_update_fraction}, "
                f"random_frac={trust_update_random_fraction})."
            )

        if fixed_trust and parallel:
            print(
                f"[Info] Fixed trust + parallel mode: processing {len(dataset)} "
                f"questions with {max_workers} workers."
            )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_data = {}
                for idx, data in enumerate(dataset):
                    update_info = (
                        trust_update_plan[idx]
                        if idx < len(trust_update_plan)
                        else {"update": True, "stage": "train_full"}
                    )
                    future = executor.submit(
                        process_item_and_update_trust,
                        data,
                        ds_name,
                        num_agents,
                        reg_rounds,
                        adj_matrix,
                        attacker_idx,
                        system_prompts,
                        model,
                        max_tokens,
                        temperature,
                        save_json,
                        mode,
                        sample_id,
                        trust_matrix_ref,
                        fixed_trust,
                        trust_lr,
                        log_trust_csv,
                        log_belief_csv,
                        self_as_speaker,
                        trust_threshold,
                        trust_enabled,
                        persuasion_levels,
                        agreeableness_levels,
                        vote_only,
                        backend,
                        vllm_url,
                        mlflow_tracking,
                        idx,
                        summary_recorder,
                        agent_labels,
                        trust_update=bool(update_info.get("update", True)),
                        trust_stage=str(update_info.get("stage", "")) or None,
                        fj_params=fj_params,
                        scenario=scenario,
                        model_name=model_dir_name,
                        seed_str=seed_str,
                        output_root=output_root,
                    )
                    future_to_data[future] = data

                for future in tqdm(
                    as_completed(future_to_data),
                    total=len(dataset),
                    desc="CSQA (parallel)",
                ):
                    try:
                        future.result()
                    except Exception as exc:
                        data = future_to_data[future]
                        print(
                            f"\n[ERROR] Failed to process item {data.get('id', 'unknown')}: {exc}"
                        )

        elif fixed_trust:
            if trust_enabled:
                print("[Info] Fixed trust mode: trust frozen after warm-up. Processing sequentially.")
            else:
                print("[Info] No-trust mode: uniform weights, processing sequentially.")
            for idx, data in enumerate(tqdm(dataset, desc="CSQA")):
                update_info = (
                    trust_update_plan[idx]
                    if idx < len(trust_update_plan)
                    else {"update": False, "stage": "trust_frozen"}
                )
                process_item_and_update_trust(
                    data,
                    ds_name,
                    num_agents,
                    reg_rounds,
                    adj_matrix,
                    attacker_idx,
                    system_prompts,
                    model,
                    max_tokens,
                    temperature,
                    save_json,
                    mode,
                    sample_id,
                    trust_matrix_ref,
                    fixed_trust,
                    trust_lr,
                    log_trust_csv,
                    log_belief_csv,
                    self_as_speaker,
                    trust_threshold,
                    trust_enabled,
                    persuasion_levels,
                    agreeableness_levels,
                    vote_only,
                    backend,
                    vllm_url,
                    mlflow_tracking,
                    idx,
                    summary_recorder,
                    agent_labels,
                    trust_update=bool(update_info.get("update", False)),
                    trust_stage=str(update_info.get("stage", "")) or None,
                    fj_params=fj_params,
                    scenario=scenario,
                    model_name=model_dir_name,
                    seed_str=seed_str,
                    output_root=output_root,
                )
        else:
            if threads != 1:
                print(
                    "[Info] For adaptive trust correctness, processing items sequentially (threads=1)."
                )
            threads = 1

            for idx, data in enumerate(tqdm(dataset, desc="CSQA")):
                update_info = (
                    trust_update_plan[idx]
                    if idx < len(trust_update_plan)
                    else {"update": True, "stage": "train_full"}
                )
                process_item_and_update_trust(
                    data,
                    ds_name,
                    num_agents,
                    reg_rounds,
                    adj_matrix,
                    attacker_idx,
                    system_prompts,
                    model,
                    max_tokens,
                    temperature,
                    save_json,
                    mode,
                    sample_id,
                    trust_matrix_ref,
                    fixed_trust,
                    trust_lr,
                    log_trust_csv,
                    log_belief_csv,
                    self_as_speaker,
                    trust_threshold,
                    trust_enabled,
                    persuasion_levels,
                    agreeableness_levels,
                    vote_only,
                    backend,
                    vllm_url,
                    mlflow_tracking,
                    idx,
                    summary_recorder,
                    agent_labels,
                    trust_update=bool(update_info.get("update", True)),
                    trust_stage=str(update_info.get("stage", "")) or None,
                    fj_params=fj_params,
                    scenario=scenario,
                    model_name=model_dir_name,
                    seed_str=seed_str,
                    output_root=output_root,
                )

        recorder_outputs = summary_recorder.export(out_dir)
        if recorder_outputs.get("agent_csv"):
            print(f"[Summary] Per-agent log saved -> {recorder_outputs['agent_csv']}")
        if recorder_outputs.get("item_csv"):
            print(f"[Summary] Per-question summary -> {recorder_outputs['item_csv']}")
        if recorder_outputs.get("metrics_csv"):
            print(f"[Summary] Agent metrics -> {recorder_outputs['metrics_csv']}")
        if recorder_outputs.get("round_csv"):
            print(f"[Summary] Round metrics -> {recorder_outputs['round_csv']}")
        if recorder_outputs.get("trust_csv"):
            print(f"[Summary] Trust progression -> {recorder_outputs['trust_csv']}")

        if trust_enabled:
            final_W_path = f"{out_dir}/{ds_name}_{mode}.final_trust.csv"
            np.savetxt(final_W_path, trust_matrix_ref["W"], delimiter=",")
            print(f"[Done] Final trust matrix saved -> {final_W_path}")

        if mlflow_tracking:
            if trust_enabled:
                final_trust_mean = float(np.mean(trust_matrix_ref["W"]))
                final_trust_std = float(np.std(trust_matrix_ref["W"]))
                final_trust_min = float(np.min(trust_matrix_ref["W"]))
                final_trust_max = float(np.max(trust_matrix_ref["W"]))

                mlflow.log_metric("final_trust_mean", final_trust_mean)
                mlflow.log_metric("final_trust_std", final_trust_std)
                mlflow.log_metric("final_trust_min", final_trust_min)
                mlflow.log_metric("final_trust_max", final_trust_max)

                mlflow.log_artifact(final_W_path, artifact_path="trust")

                if log_trust_csv:
                    trust_logs_dir = os.path.join(out_dir, "trust_logs")
                    if os.path.exists(trust_logs_dir):
                        mlflow.log_artifacts(trust_logs_dir, artifact_path="trust_logs")

            if log_belief_csv:
                belief_logs_dir = os.path.join(out_dir, "belief_logs")
                if os.path.exists(belief_logs_dir):
                    mlflow.log_artifacts(belief_logs_dir, artifact_path="belief_logs")

            if recorder_outputs.get("agent_csv"):
                mlflow.log_artifact(
                    recorder_outputs["agent_csv"], artifact_path="summaries"
                )
            if recorder_outputs.get("item_csv"):
                mlflow.log_artifact(
                    recorder_outputs["item_csv"], artifact_path="summaries"
                )
            if recorder_outputs.get("metrics_csv"):
                mlflow.log_artifact(
                    recorder_outputs["metrics_csv"], artifact_path="summaries"
                )
            if recorder_outputs.get("round_csv"):
                mlflow.log_artifact(
                    recorder_outputs["round_csv"], artifact_path="summaries"
                )
            if recorder_outputs.get("trust_csv"):
                mlflow.log_artifact(
                    recorder_outputs["trust_csv"], artifact_path="summaries"
                )
            if recorder_outputs.get("round_prob_plot"):
                mlflow.log_artifact(
                    recorder_outputs["round_prob_plot"], artifact_path="figures"
                )
            if recorder_outputs.get("round_accuracy_plot"):
                mlflow.log_artifact(
                    recorder_outputs["round_accuracy_plot"], artifact_path="figures"
                )
            if recorder_outputs.get("trust_plot"):
                mlflow.log_artifact(
                    recorder_outputs["trust_plot"], artifact_path="figures"
                )
            for metric_name, metric_value in recorder_outputs.get("metrics", {}).items():
                if metric_value is None or not np.isfinite(metric_value):
                    continue
                mlflow.log_metric(metric_name, float(metric_value))

            params_csv_path = os.path.join(scenario_dir, "params.csv")
            if os.path.exists(params_csv_path):
                mlflow.log_artifact(params_csv_path, artifact_path="config")

            print(f"[MLflow] Run completed: {mlflow.active_run().info.run_id}")
            view_uri = resolved_mlflow_uri or mlflow.get_tracking_uri()
            if view_uri:
                print(f"[MLflow] View at: {view_uri}")

    finally:
        if mlflow_tracking and mlflow.active_run():
            mlflow.end_run()


def run_from_namespace(args):
    """Execute the CSQA experiment given an argparse namespace."""
    random.seed(args.sample)
    np.random.seed(args.sample)
    os.environ["PYTHONHASHSEED"] = str(args.sample)

    persuasion_levels = parse_level_list(args.persuasion_levels)
    agreeableness_levels = parse_level_list(args.agreeableness_levels)
    fj_params = _parse_fj_params(args)

    if args.parallel and not args.fixed_trust and not getattr(args, "no_trust", False):
        raise SystemExit(
            "ERROR: --parallel requires --fixed-trust. "
            "Parallel processing is only safe when trust is not being updated."
        )

    dataset = args.dataset
    graph_type = args.graph
    num_agents = args.agents
    model = args.model
    model_dir_name = getattr(args, "model_dir_name", "") or model
    sample_id = args.sample
    k_attackers = args.attackers
    save_json = bool(getattr(args, "save_json", False) or getattr(args, "json", False))
    reg_rounds = max(0, args.rounds)
    threads = max(1, args.threads)

    seed_dir = Path("output") / model_dir_name / dataset / args.scenario / str(sample_id)
    if seed_dir.exists():
        has_outputs = any(seed_dir.glob("*.output"))
        summaries_dir = seed_dir / "summaries"
        has_summaries = summaries_dir.is_dir() and any(summaries_dir.iterdir())
        is_complete = has_outputs and has_summaries
        if is_complete and getattr(args, "skip_existing", True):
            print(
                f"[SKIP] Completed run found for scenario '{args.scenario}' "
                f"(sample {sample_id}) at {seed_dir}. Use --rerun-existing to rerun."
            )
            return
        if has_outputs:
            print(
                f"[RE-RUN] Existing output detected for scenario '{args.scenario}' "
                f"(sample {sample_id}) at {seed_dir}. "
                f"Removing seed directory and rerunning."
            )
            try:
                shutil.rmtree(seed_dir)
                print(f"[CLEANUP] Removed seed directory {seed_dir}.")
            except OSError as exc:
                print(
                    f"[WARN] Failed to remove seed directory {seed_dir}: {exc}. "
                    "Proceeding with rerun."
                )

    if not graph_type:
        raise SystemExit("ERROR: graph topology must be set via --graph or config.")
    if not args.scenario:
        raise SystemExit("ERROR: scenario name must be set via --scenario or config.")

    explicit_idx = parse_int_list(args.attackers_idx)

    if k_attackers <= 0:
        attacker_idx = []
    elif explicit_idx is not None:
        attacker_idx = explicit_idx
        k_attackers = len(attacker_idx)
    else:
        placement = str(args.placement).strip()
        if graph_type in ("star", "pure_star"):
            if placement in ("center", "0"):
                attacker_idx = [0][:k_attackers]
            elif placement in ("leaf", "1"):
                attacker_idx = [1][:k_attackers]
            elif placement in ("two_leaves", "2"):
                attacker_idx = [1, 2][:k_attackers]
            else:
                attacker_idx = (
                    [0]
                    if k_attackers == 1
                    else list(range(1, min(1 + k_attackers, num_agents)))
                )
        else:
            attacker_idx = list(range(min(k_attackers, num_agents)))

    if any(i < 0 or i >= num_agents for i in attacker_idx):
        raise SystemExit(
            f"ERROR: attacker indices {attacker_idx} out of range for num_agents={num_agents}"
        )
    if len(set(attacker_idx)) != len(attacker_idx):
        raise SystemExit(f"ERROR: duplicate entries in attacker indices: {attacker_idx}")

    trust_enabled = not getattr(args, "no_trust", False)

    if trust_enabled:
        if args.trust_matrix.strip():
            trust_matrix = load_trust_matrix(args.trust_matrix, num_agents)
        else:
            trust_matrix = np.full((num_agents, num_agents), 0.5, dtype=float)
    else:
        trust_matrix = np.ones((num_agents, num_agents), dtype=float)
        args.fixed_trust = True
        args.trust_lr = 0.0
        args.log_trust_csv = False

    if args.backend == "vllm" and getattr(args, "auto_launch_vllm", False):
        from cascade.experiments.vllm import ensure_vllm_server

        server = ensure_vllm_server(
            model=model,
            requested_gpu=args.vllm_gpu,
            explicit_port=args.vllm_port,
            port_start=args.vllm_port_base,
            log_dir=args.vllm_log_dir,
            gpu_memory_utilization=args.vllm_gpu_utilization,
            max_model_len=args.vllm_max_model_len,
            extra_args=args.vllm_extra_args,
            wait_seconds=args.vllm_wait_seconds,
        )
        if server.base_url:
            args.vllm_url = server.base_url
        print(
            f"[auto-vllm] Using server on GPU {server.gpu_index} "
            f"pid={server.pid} port={server.port}"
        )

    print("Attacker Idx:", attacker_idx)
    if trust_enabled:
        print("Trust initialized to 0.5 (neutral) for all agents")
    else:
        print("Uniform weighting active (no adaptive updates).")

    run_dataset(
        dataset,
        sample_id,
        attacker_idx,
        graph_type,
        model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        threads=threads,
        num_agents=num_agents,
        save_json=save_json,
        reg_rounds=reg_rounds,
        trust_matrix=trust_matrix,
        trust_enabled=trust_enabled,
        warmup=args.warmup,
        fixed_trust=args.fixed_trust,
        parallel=args.parallel,
        max_workers=args.max_workers,
        trust_lr=args.trust_lr,
        trust_gamma=args.trust_gamma,
        log_trust_csv=bool(args.log_trust_csv),
        log_belief_csv=bool(args.log_belief_csv),
        self_as_speaker=bool(args.self_as_speaker),
        trust_threshold=float(args.trust_threshold),
        trust_update_mode=str(getattr(args, "trust_update_mode", "always") or "always"),
        trust_update_fraction=float(
            getattr(args, "trust_update_fraction", 1.0) or 0.0
        ),
        trust_update_random_fraction=float(
            getattr(args, "trust_update_random_fraction", 0.0) or 0.0
        ),
        trust_update_seed=(
            args.trust_update_seed
            if getattr(args, "trust_update_seed", None) is not None
            else None
        ),
        save_warmup_trust=bool(args.save_warmup_trust),
        adaptive_attacker=bool(getattr(args, "adaptive_attacker", False)),
        persuasion_levels=persuasion_levels,
        agreeableness_levels=agreeableness_levels,
        vote_only=bool(args.vote_only),
        backend=args.backend,
        vllm_url=args.vllm_url,
        mlflow_tracking=bool(args.mlflow_tracking),
        mlflow_uri=args.mlflow_uri,
        mlflow_experiment=args.mlflow_experiment,
        fj_params=fj_params,
        scenario=args.scenario,
        output_root="output",
        args_namespace=args,
        model_dir_name=model_dir_name,
    )


__all__ = [
    "build_output_filename",
    "process_item_and_update_trust",
    "run_dataset",
    "run_from_namespace",
]


