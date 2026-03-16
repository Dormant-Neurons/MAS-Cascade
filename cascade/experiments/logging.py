"""
Logging utilities and structured recorders for CSQA experiments.
"""

from __future__ import annotations

import csv
import os
import threading
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import mlflow
import numpy as np
import matplotlib.pyplot as plt

# Thread-safe lock for MLflow logging when experiments run in parallel
mlflow_lock = threading.Lock()


def compute_accuracy(dataset: Sequence[dict], agent_graph) -> np.ndarray:
    """Compute accuracy for each agent across the given dataset."""
    n = agent_graph.num_agents
    correct = np.zeros(n, dtype=int)
    total = len(dataset)

    for data in dataset:
        answer_key = data["answerKey"].strip().upper()
        for j in range(n):
            ans = str(agent_graph.Agents[j].last_response.get("answer", "")).strip().upper()
            if ans == answer_key:
                correct[j] += 1

    return correct / total if total > 0 else np.zeros(n)


def log_mlflow_metrics_per_item(
    item_id: str,
    agent_graph,
    answer_key: str,
    step: int,
    agent_labels: Sequence[str] | None = None,
    *,
    trust_enabled: bool = True,
):
    """Log per-item metrics to MLflow in a thread-safe fashion."""
    with mlflow_lock:
        n = agent_graph.num_agents
        labels = ["A", "B", "C", "D", "E"]

        for i in range(n):
            agent = agent_graph.Agents[i]
            ans = agent.last_response.get("answer", "None")
            belief = agent.last_response.get("belief", {k: 0.2 for k in labels})
            label_name = (
                agent_labels[i] if agent_labels and i < len(agent_labels) else f"Agent_{i}"
            )

            is_correct = 1.0 if ans.strip().upper() == answer_key.strip().upper() else 0.0
            mlflow.log_metric(f"{label_name}_correct", is_correct, step=step)

            belief_vec = np.array([belief.get(k, 0.0) for k in labels], dtype=float)
            total = belief_vec.sum()
            if total > 0:
                belief_vec = belief_vec / total
            entropy_val = -np.sum(belief_vec * np.log(belief_vec + 1e-12))
            max_entropy = np.log(len(labels))
            normalized_entropy = entropy_val / max_entropy
            mlflow.log_metric(f"{label_name}_belief_entropy", normalized_entropy, step=step)

            mlflow.log_metric(f"{label_name}_belief_max", float(np.max(belief_vec)), step=step)

        correct_count = sum(
            1
            for i in range(n)
            if agent_graph.Agents[i].last_response.get("answer", "").strip().upper()
            == answer_key.strip().upper()
        )
        mlflow.log_metric("accuracy", correct_count / n if n else 0.0, step=step)

        if trust_enabled:
            trust_mean = float(np.mean(agent_graph.trust_matrix))
            trust_std = float(np.std(agent_graph.trust_matrix))
            trust_min = float(np.min(agent_graph.trust_matrix))
            trust_max = float(np.max(agent_graph.trust_matrix))

            mlflow.log_metric("trust_mean", trust_mean, step=step)
            mlflow.log_metric("trust_std", trust_std, step=step)
            mlflow.log_metric("trust_min", trust_min, step=step)
            mlflow.log_metric("trust_max", trust_max, step=step)


class RunRecorder:
    """Collect structured per-item and per-agent logs for easier inspection."""

    def __init__(
        self,
        dataset: str,
        scenario: str,
        sample_id: int,
        num_agents: int,
        attacker_idx: Iterable[int],
        agent_labels: Sequence[str] | None,
    ):
        self.dataset = dataset
        self.scenario = scenario
        self.sample_id = sample_id
        self.num_agents = num_agents
        self.attacker_idx = set(attacker_idx or [])
        self.labels = ["A", "B", "C", "D", "E"]
        self.agent_labels = agent_labels or [f"Agent_{i}" for i in range(num_agents)]
        self.agent_label_map = {i: self.agent_labels[i] for i in range(num_agents)}

        self.lock = threading.Lock()
        self.agent_rows: List[Dict] = []
        self.item_rows: List[Dict] = []
        self.round_stats: Dict[int, Dict[str, float]] = {}
        self.round_labels: List[str] = []
        self.trust_series: List[Dict[str, float]] = []

        self.agent_metrics: Dict[int, Dict[str, float]] = {
            agent_id: {
                "role": "attacker" if agent_id in self.attacker_idx else "defender",
                "label": self.agent_label_map.get(agent_id, f"Agent_{agent_id}"),
                "total": 0,
                "correct": 0,
                "prob_correct_sum": 0.0,
                "prob_selected_sum": 0.0,
                "prob_selected_count": 0,
            }
            for agent_id in range(num_agents)
        }

        self.agent_fieldnames = [
            "dataset",
            "scenario",
            "sample_id",
            "item_index",
            "item_order",
            "item_id",
            "trust_stage",
            "question",
            "choices",
            "correct_answer",
            "agent_id",
            "agent_label",
            "role",
            "answer",
            "is_correct",
            "prob_correct",
            "prob_selected",
            "belief_A",
            "belief_B",
            "belief_C",
            "belief_D",
            "belief_E",
            "belief_max",
            "belief_argmax",
            "trust_out_mean",
            "trust_in_mean",
            "reason",
            "memory",
            "trust_stage",
        ]

        self.item_fieldnames = [
            "dataset",
            "scenario",
            "sample_id",
            "item_index",
            "item_order",
            "item_id",
            "trust_stage",
            "question",
            "choices",
            "correct_answer",
            "majority_answer",
            "defender_majority",
            "attacker_answers",
            "num_agents",
            "num_attackers",
            "accuracy_overall",
            "accuracy_defenders",
            "accuracy_attackers",
            "avg_prob_correct",
            "avg_prob_correct_defenders",
            "avg_prob_correct_attackers",
        ]

        self.metrics_fieldnames = [
            "dataset",
            "scenario",
            "sample_id",
            "agent_id",
            "agent_label",
            "role",
            "num_items",
            "num_correct",
            "accuracy",
            "avg_prob_correct",
            "avg_prob_selected",
        ]

    @staticmethod
    def _compact(text):
        if text is None:
            return ""
        return " ".join(str(text).split())

    @staticmethod
    def _majority(values):
        filtered = [v for v in values if v not in ("", "None", None)]
        if not filtered:
            return ""
        counts = Counter(filtered)
        return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]

    @staticmethod
    def _empty_round_stats():
        return {
            "prob_sum": 0.0,
            "prob_sum_def": 0.0,
            "prob_sum_att": 0.0,
            "acc_sum": 0.0,
            "acc_sum_def": 0.0,
            "acc_sum_att": 0.0,
            "count": 0,
            "count_def": 0,
            "count_att": 0,
        }

    @staticmethod
    def _mean_from_pairs(matrix, senders, receivers):
        values = [
            matrix[i, j]
            for i in senders
            for j in receivers
            if i != j
        ]
        return float(np.mean(values)) if values else float("nan")

    def record_item(
        self,
        *,
        item_index: int,
        item_id: str,
        question: str,
        choices: dict,
        correct_answer: str,
        agent_graph,
        belief_snapshots=None,
        trust_init=None,
        trust_final=None,
        trust_stage: str | None = None,
    ):
        trust_stage = (trust_stage or "").strip()
        choice_labels = choices.get("label", []) if isinstance(choices, dict) else []
        choice_text = choices.get("text", []) if isinstance(choices, dict) else []
        choices_pairs = " | ".join(
            f"{lbl}: {txt}" for lbl, txt in zip(choice_labels, choice_text)
        )

        question_compact = self._compact(question)
        choices_compact = self._compact(choices_pairs)
        correct_answer = (correct_answer or "").strip().upper()
        correct_idx = (
            self.labels.index(correct_answer) if correct_answer in self.labels else None
        )

        item_rows = []
        defenders = []
        attackers = []

        for agent_id, agent in enumerate(agent_graph.Agents):
            last = agent.last_response or {}
            answer_raw = str(last.get("answer", "None")).strip()
            answer_upper = answer_raw.upper()
            belief_raw = last.get("belief") or {}
            belief = {label: float(belief_raw.get(label, 0.0)) for label in self.labels}
            prob_correct = belief.get(correct_answer, 0.0)
            prob_selected = (
                belief.get(answer_upper, 0.0) if answer_upper in self.labels else None
            )
            belief_max = max(belief.values()) if belief else 0.0
            belief_argmax = max(belief, key=belief.get) if belief else ""
            reason_text = self._compact(last.get("reason", ""))
            memory_text = self._compact(agent.short_mem[-1]) if agent.short_mem else ""
            role = "attacker" if agent_id in self.attacker_idx else "defender"
            is_correct = (
                1
                if answer_upper in self.labels and answer_upper == correct_answer
                else 0
            )
            label_name = self.agent_label_map.get(agent_id, f"Agent_{agent_id}")

            trust_out_mean = (
                float(np.mean(agent_graph.trust_matrix[agent_id]))
                if agent_graph.trust_matrix.size
                else 0.0
            )
            trust_in_mean = (
                float(np.mean(agent_graph.trust_matrix[:, agent_id]))
                if agent_graph.trust_matrix.size
                else 0.0
            )

            row = {
                "dataset": self.dataset,
                "scenario": self.scenario,
                "sample_id": self.sample_id,
                "item_index": int(item_index),
                "item_order": int(item_index) + 1,
                "item_id": item_id,
                "question": question_compact,
                "choices": choices_compact,
                "correct_answer": correct_answer,
                "agent_id": agent_id,
                "agent_label": label_name,
                "role": role,
                "answer": answer_upper or "None",
                "is_correct": is_correct,
                "prob_correct": prob_correct,
                "prob_selected": prob_selected if prob_selected is not None else 0.0,
                "belief_A": belief.get("A", 0.0),
                "belief_B": belief.get("B", 0.0),
                "belief_C": belief.get("C", 0.0),
                "belief_D": belief.get("D", 0.0),
                "belief_E": belief.get("E", 0.0),
                "belief_max": belief_max,
                "belief_argmax": belief_argmax,
                "trust_out_mean": trust_out_mean,
                "trust_in_mean": trust_in_mean,
                "reason": reason_text,
                "memory": memory_text,
                "trust_stage": trust_stage,
            }

            item_rows.append(row)
            (defenders if role == "defender" else attackers).append(row)

            metrics = self.agent_metrics.get(agent_id)
            if metrics is not None:
                metrics["total"] += 1
                metrics["correct"] += is_correct
                metrics["prob_correct_sum"] += prob_correct
                if prob_selected is not None:
                    metrics["prob_selected_sum"] += prob_selected
                    metrics["prob_selected_count"] += 1

        accuracy_overall = (
            sum(r["is_correct"] for r in item_rows) / len(item_rows) if item_rows else 0.0
        )
        accuracy_defenders = (
            sum(r["is_correct"] for r in defenders) / len(defenders) if defenders else 0.0
        )
        accuracy_attackers = (
            sum(r["is_correct"] for r in attackers) / len(attackers) if attackers else 0.0
        )
        avg_prob_correct = (
            float(np.mean([r["prob_correct"] for r in item_rows])) if item_rows else 0.0
        )
        avg_prob_correct_defenders = (
            float(np.mean([r["prob_correct"] for r in defenders])) if defenders else 0.0
        )
        avg_prob_correct_attackers = (
            float(np.mean([r["prob_correct"] for r in attackers])) if attackers else 0.0
        )

        item_summary = {
            "dataset": self.dataset,
            "scenario": self.scenario,
            "sample_id": self.sample_id,
            "item_index": int(item_index),
            "item_order": int(item_index) + 1,
            "item_id": item_id,
            "trust_stage": trust_stage,
            "question": question_compact,
            "choices": choices_compact,
            "correct_answer": correct_answer,
            "majority_answer": self._majority([r["answer"] for r in item_rows]),
            "defender_majority": self._majority([r["answer"] for r in defenders]),
            "attacker_answers": "|".join(
                r["answer"] for r in attackers if r["answer"] not in ("", "None")
            )
            or "",
            "num_agents": len(item_rows),
            "num_attackers": len(attackers),
            "accuracy_overall": accuracy_overall,
            "accuracy_defenders": accuracy_defenders,
            "accuracy_attackers": accuracy_attackers,
            "avg_prob_correct": avg_prob_correct,
            "avg_prob_correct_defenders": avg_prob_correct_defenders,
            "avg_prob_correct_attackers": avg_prob_correct_attackers,
        }

        defender_indices = [i for i in range(self.num_agents) if i not in self.attacker_idx]
        attacker_indices = list(self.attacker_idx)

        round_updates = []
        round_labels_candidate = None
        if belief_snapshots and correct_idx is not None:
            round_labels_candidate = [
                snap.get("where", f"round{idx}") for idx, snap in enumerate(belief_snapshots)
            ]
            for ridx, snap in enumerate(belief_snapshots):
                beliefs = snap.get("beliefs", [])
                if not beliefs:
                    continue
                delta = self._empty_round_stats()
                for agent_id, belief_vec in enumerate(beliefs):
                    if not belief_vec or len(belief_vec) <= correct_idx:
                        continue
                    belief_vec = list(map(float, belief_vec))
                    prob_correct_round = belief_vec[correct_idx]
                    predicted_idx = int(np.argmax(belief_vec))
                    is_correct_round = 1 if self.labels[predicted_idx] == correct_answer else 0
                    delta["prob_sum"] += prob_correct_round
                    delta["acc_sum"] += is_correct_round
                    delta["count"] += 1
                    if agent_id in self.attacker_idx:
                        delta["prob_sum_att"] += prob_correct_round
                        delta["acc_sum_att"] += is_correct_round
                        delta["count_att"] += 1
                    else:
                        delta["prob_sum_def"] += prob_correct_round
                        delta["acc_sum_def"] += is_correct_round
                        delta["count_def"] += 1
                round_updates.append((ridx, delta))

        trust_entry = None
        if trust_init is not None and trust_final is not None:
            trust_init = np.array(trust_init, dtype=float)
            trust_final = np.array(trust_final, dtype=float)
            defenders_set = defender_indices
            attackers_set = attacker_indices

            def_to_att_init = self._mean_from_pairs(trust_init, defenders_set, attackers_set)
            def_to_att_final = self._mean_from_pairs(trust_final, defenders_set, attackers_set)
            att_to_def_init = self._mean_from_pairs(trust_init, attackers_set, defenders_set)
            att_to_def_final = self._mean_from_pairs(trust_final, attackers_set, defenders_set)
            def_to_def_init = self._mean_from_pairs(trust_init, defenders_set, defenders_set)
            def_to_def_final = self._mean_from_pairs(trust_final, defenders_set, defenders_set)

            trust_entry = {
                "item_index": int(item_index),
                "item_order": int(item_index) + 1,
                "item_id": item_id,
                "trust_stage": trust_stage,
                "mean_init": float(np.mean(trust_init)),
                "mean_final": float(np.mean(trust_final)),
                "def_to_att_init": def_to_att_init,
                "def_to_att_final": def_to_att_final,
                "att_to_def_init": att_to_def_init,
                "att_to_def_final": att_to_def_final,
                "def_to_def_init": def_to_def_init,
                "def_to_def_final": def_to_def_final,
            }

        with self.lock:
            self.agent_rows.extend(item_rows)
            self.item_rows.append(item_summary)
            if round_labels_candidate and not self.round_labels:
                self.round_labels = round_labels_candidate
            for ridx, delta in round_updates:
                stats = self.round_stats.setdefault(ridx, self._empty_round_stats())
                for key, value in delta.items():
                    stats[key] += value
            if trust_entry:
                self.trust_series.append(trust_entry)

    def export(self, out_dir: str) -> Dict[str, str]:
        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        summaries_dir = out_dir_path / "summaries"
        summaries_dir.mkdir(parents=True, exist_ok=True)
        outputs: Dict[str, str] = {}

        if self.agent_rows:
            agent_path = summaries_dir / "run_agent_log.csv"
            with agent_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.agent_fieldnames)
                writer.writeheader()
                writer.writerows(self.agent_rows)
            outputs["agent_csv"] = str(agent_path)

        if self.item_rows:
            item_path = summaries_dir / "run_item_summary.csv"
            with item_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.item_fieldnames)
                writer.writeheader()
                writer.writerows(self.item_rows)
            outputs["item_csv"] = str(item_path)

        metrics_rows = []
        metrics_summary = {}
        for agent_id, metrics in self.agent_metrics.items():
            total = metrics["total"]
            num_correct = metrics["correct"]
            accuracy = num_correct / total if total else 0.0
            avg_prob_correct = metrics["prob_correct_sum"] / total if total else 0.0
            avg_prob_selected = (
                metrics["prob_selected_sum"] / metrics["prob_selected_count"]
                if metrics["prob_selected_count"] > 0
                else 0.0
            )
            label_name = metrics.get("label", f"Agent_{agent_id}")
            metrics_row = {
                "dataset": self.dataset,
                "scenario": self.scenario,
                "sample_id": self.sample_id,
                "agent_id": agent_id,
                "agent_label": label_name,
                "role": metrics["role"],
                "num_items": total,
                "num_correct": num_correct,
                "accuracy": accuracy,
                "avg_prob_correct": avg_prob_correct,
                "avg_prob_selected": avg_prob_selected,
            }
            metrics_rows.append(metrics_row)
            metrics_summary[f"{label_name}_accuracy_final"] = accuracy
            metrics_summary[f"{label_name}_avg_prob_correct_final"] = avg_prob_correct
            metrics_summary[f"{label_name}_avg_prob_selected_final"] = avg_prob_selected

        if metrics_rows:
            metrics_path = summaries_dir / "run_agent_metrics.csv"
            with metrics_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.metrics_fieldnames)
                writer.writeheader()
                writer.writerows(metrics_rows)
            outputs["metrics_csv"] = str(metrics_path)

        if self.round_stats:
            round_path = summaries_dir / "round_metrics.csv"
            fieldnames = [
                "round_index",
                "round_label",
                "prob_correct_overall",
                "prob_correct_defenders",
                "prob_correct_attackers",
                "accuracy_overall",
                "accuracy_defenders",
                "accuracy_attackers",
            ]
            with round_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                for ridx in sorted(self.round_stats.keys()):
                    stats = self.round_stats[ridx]
                    row = {
                        "round_index": ridx,
                        "round_label": self.round_labels[ridx]
                        if ridx < len(self.round_labels)
                        else f"round{ridx}",
                        "prob_correct_overall": stats["prob_sum"] / stats["count"]
                        if stats["count"]
                        else float("nan"),
                        "prob_correct_defenders": stats["prob_sum_def"] / stats["count_def"]
                        if stats["count_def"]
                        else float("nan"),
                        "prob_correct_attackers": stats["prob_sum_att"] / stats["count_att"]
                        if stats["count_att"]
                        else float("nan"),
                        "accuracy_overall": stats["acc_sum"] / stats["count"]
                        if stats["count"]
                        else float("nan"),
                        "accuracy_defenders": stats["acc_sum_def"] / stats["count_def"]
                        if stats["count_def"]
                        else float("nan"),
                        "accuracy_attackers": stats["acc_sum_att"] / stats["count_att"]
                        if stats["count_att"]
                        else float("nan"),
                    }
                    writer.writerow(row)
            outputs["round_csv"] = str(round_path)

            rounds = [
                self.round_labels[r] if r < len(self.round_labels) else f"round{r}"
                for r in sorted(self.round_stats.keys())
            ]
            indices = np.arange(len(rounds))
            overall_probs = [
                self.round_stats[r]["prob_sum"] / self.round_stats[r]["count"]
                if self.round_stats[r]["count"]
                else float("nan")
                for r in sorted(self.round_stats.keys())
            ]
            def_probs = [
                self.round_stats[r]["prob_sum_def"] / self.round_stats[r]["count_def"]
                if self.round_stats[r]["count_def"]
                else float("nan")
                for r in sorted(self.round_stats.keys())
            ]
            att_probs = [
                self.round_stats[r]["prob_sum_att"] / self.round_stats[r]["count_att"]
                if self.round_stats[r]["count_att"]
                else float("nan")
                for r in sorted(self.round_stats.keys())
            ]

            plt.figure(figsize=(8, 5))
            plt.plot(indices, overall_probs, marker="o", label="Overall")
            if any(np.isfinite(def_probs)):
                plt.plot(indices, def_probs, marker="o", label="Defenders")
            if any(np.isfinite(att_probs)):
                plt.plot(indices, att_probs, marker="o", label="Attackers")
            plt.xticks(indices, rounds)
            plt.ylim(0, 1)
            plt.xlabel("Round")
            plt.ylabel("Average probability on correct option")
            plt.title("Belief convergence across rounds")
            plt.grid(True, alpha=0.3)
            plt.legend()
            round_prob_plot = summaries_dir / "round_probabilities.png"
            plt.tight_layout()
            plt.savefig(round_prob_plot, dpi=300)
            plt.close()
            outputs["round_prob_plot"] = str(round_prob_plot)

            overall_acc = [
                self.round_stats[r]["acc_sum"] / self.round_stats[r]["count"]
                if self.round_stats[r]["count"]
                else float("nan")
                for r in sorted(self.round_stats.keys())
            ]
            def_acc = [
                self.round_stats[r]["acc_sum_def"] / self.round_stats[r]["count_def"]
                if self.round_stats[r]["count_def"]
                else float("nan")
                for r in sorted(self.round_stats.keys())
            ]
            att_acc = [
                self.round_stats[r]["acc_sum_att"] / self.round_stats[r]["count_att"]
                if self.round_stats[r]["count_att"]
                else float("nan")
                for r in sorted(self.round_stats.keys())
            ]

            plt.figure(figsize=(8, 5))
            plt.plot(indices, overall_acc, marker="o", label="Overall")
            if any(np.isfinite(def_acc)):
                plt.plot(indices, def_acc, marker="o", label="Defenders")
            if any(np.isfinite(att_acc)):
                plt.plot(indices, att_acc, marker="o", label="Attackers")
            plt.xticks(indices, rounds)
            plt.ylim(0, 1)
            plt.xlabel("Round")
            plt.ylabel("Accuracy")
            plt.title("Accuracy across rounds")
            plt.grid(True, alpha=0.3)
            plt.legend()
            round_accuracy_plot = summaries_dir / "round_accuracy.png"
            plt.tight_layout()
            plt.savefig(round_accuracy_plot, dpi=300)
            plt.close()
            outputs["round_accuracy_plot"] = str(round_accuracy_plot)

        if self.trust_series:
            trust_csv_path = summaries_dir / "trust_overview.csv"
            trust_fieldnames = [
                "item_index",
                "item_order",
                "item_id",
                "trust_stage",
                "mean_init",
                "mean_final",
                "def_to_att_init",
                "def_to_att_final",
                "att_to_def_init",
                "att_to_def_final",
                "def_to_def_init",
                "def_to_def_final",
            ]
            with trust_csv_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=trust_fieldnames)
                writer.writeheader()
                writer.writerows(self.trust_series)
            outputs["trust_csv"] = str(trust_csv_path)

            orders = [entry["item_order"] for entry in self.trust_series]
            def_att_init = [entry["def_to_att_init"] for entry in self.trust_series]
            def_att_final = [entry["def_to_att_final"] for entry in self.trust_series]
            att_def_init = [entry["att_to_def_init"] for entry in self.trust_series]
            att_def_final = [entry["att_to_def_final"] for entry in self.trust_series]
            def_def_init = [entry["def_to_def_init"] for entry in self.trust_series]
            def_def_final = [entry["def_to_def_final"] for entry in self.trust_series]

            plt.figure(figsize=(8, 5))
            if any(np.isfinite(def_att_init)):
                plt.plot(orders, def_att_init, linestyle="--", label="Defenders → Attacker (init)")
            if any(np.isfinite(att_def_init)):
                plt.plot(orders, att_def_init, linestyle="--", label="Attacker → Defenders (init)")
            if any(np.isfinite(def_def_init)):
                plt.plot(orders, def_def_init, linestyle="--", label="Defender ↔ Defender (init)")
            if any(np.isfinite(def_att_final)):
                plt.plot(orders, def_att_final, marker="o", label="Defenders → Attacker (final)")
            if any(np.isfinite(att_def_final)):
                plt.plot(orders, att_def_final, marker="o", label="Attacker → Defenders (final)")
            if any(np.isfinite(def_def_final)):
                plt.plot(orders, def_def_final, marker="o", label="Defender ↔ Defender (final)")
            plt.ylim(0, 1)
            plt.xlabel("Task order")
            plt.ylabel("Trust weight")
            plt.title("Trust evolution across tasks")
            plt.grid(True, alpha=0.3)
            plt.legend()
            trust_plot_path = summaries_dir / "trust_over_tasks.png"
            plt.tight_layout()
            plt.savefig(trust_plot_path, dpi=300)
            plt.close()
            outputs["trust_plot"] = str(trust_plot_path)

            final_trust = self.trust_series[-1]
            metrics_summary["final_mean_trust"] = final_trust["mean_final"]
            metrics_summary["final_def_to_att_trust"] = final_trust["def_to_att_final"]
            metrics_summary["final_att_to_def_trust"] = final_trust["att_to_def_final"]
            metrics_summary["final_def_to_def_trust"] = final_trust["def_to_def_final"]

        outputs["metrics"] = metrics_summary
        return outputs


__all__ = [
    "compute_accuracy",
    "log_mlflow_metrics_per_item",
    "RunRecorder",
    "mlflow_lock",
]
