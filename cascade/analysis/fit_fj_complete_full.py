#!/usr/bin/env python3
"""
FJ COMPLETE: Fit on ALL rounds (no train/test split).

Physics perspective: Find parameters that best describe belief dynamics
under a fully-connected topology.

Complete topology:
- Every agent is connected to every other agent (no self-loop)
- Each agent i has neighbor weights W[i,j] over all j != i
- Each row W[i,:] is parameterized by softmax over (N-1) logits

Model (vectorized over options k):
b_i(t+1) = gamma_i * b_i(0)
         + (1-gamma_i) * [ alpha_i * b_i(t)
                         + (1-alpha_i) * sum_{j!=i} W_ij * b_j(t) ]

This script:
1) Fits parameters on ALL rounds using all transitions
2) Outputs per-question parameters and fit quality metrics
3) Computes (1-gamma_i)*(1-alpha_i) for each agent
"""

from pathlib import Path
from io import StringIO
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ===================== CONFIG =====================
BELIEF_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/gemini-3-flash-preview/csqa_test30")

K = 5  # Number of answer options
SST_MIN = 1e-6  # R² stability threshold

OUT_FIT_SUMMARY = "fj_fit_summary.csv"
OUT_PARAMS = "fj_fit_params.csv"
# ==================================================


def find_belief_logs(base_dir: Path):
    """
    Recursively find all belief_logs directories and their CSV files.
    Returns list of (experiment_name, belief_csv_path) tuples.
    """
    results = []
    for belief_log_dir in base_dir.rglob("belief_logs"):
        experiment_name = belief_log_dir.parent.parent.name
        for csv_file in belief_log_dir.glob("*_beliefs.csv"):
            results.append((experiment_name, csv_file))
    return results


def parse_belief_csv(path: Path):
    """Parse consolidated belief CSV into structured format."""
    lines = path.read_text().splitlines()
    start_idx = next(i for i, l in enumerate(lines) if l.startswith("round,"))
    header_options = lines[start_idx + 1].split(",")

    remaining = len(header_options) - 1
    assert remaining % K == 0, f"{path.name}: header mismatch"
    n_agents = remaining // K

    options = header_options[1:1 + K]
    colnames = ["round"] + [f"Agent_{ai}_{o}" for ai in range(n_agents) for o in options]
    data_lines = lines[start_idx + 2:]
    df = pd.read_csv(StringIO(",".join(colnames) + "\n" + "\n".join(data_lines)))

    df["round"] = df["round"].str.replace("round", "", regex=False).astype(int)
    df = df.sort_values("round").reset_index(drop=True)

    return df, n_agents, options


def stable_softmax(v: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    v = v - np.max(v)
    ex = np.exp(v)
    s = ex.sum()
    return ex / s if s > 0 else np.full_like(ex, 1.0 / len(ex))


def unpack_theta_complete(theta: np.ndarray, n_agents: int):
    """
    theta = [gamma (N), alpha (N), W_logits (N*(N-1))]

    Each agent i has (N-1) logits -> softmax -> neighbor weights over j != i.
    W[i,i] is fixed to 0 (no self-loop).
    """
    gamma = theta[:n_agents]
    alpha = theta[n_agents:2 * n_agents]

    W = np.zeros((n_agents, n_agents), dtype=float)
    offset = 2 * n_agents
    for i in range(n_agents):
        others = [j for j in range(n_agents) if j != i]
        logits = theta[offset: offset + (n_agents - 1)]
        weights = stable_softmax(logits)
        for idx, j in enumerate(others):
            W[i, j] = weights[idx]
        offset += (n_agents - 1)

    return gamma, alpha, W


def fj_loss_complete(theta, S_all, b0):
    """
    One-step MSE over ALL transitions.

    S_all: (T, N, K) containing rounds 0..T-1.
    Compare predicted S[t+1] vs empirical S[t+1] for t=0..T-2.
    """
    T, n_agents, _ = S_all.shape
    gamma, alpha, W = unpack_theta_complete(theta, n_agents)

    err = []
    for t in range(T - 1):
        Bt = S_all[t]
        mix = W @ Bt
        Bt_next_pred = (
            gamma[:, None] * b0
            + (1 - gamma)[:, None] * (
                alpha[:, None] * Bt
                + (1 - alpha)[:, None] * mix
            )
        )
        err.append((S_all[t + 1] - Bt_next_pred).reshape(-1))

    resid = np.concatenate(err, axis=0)
    return float(np.mean(resid ** 2))


def compute_global_r2(y_true: np.ndarray, y_pred: np.ndarray):
    """Compute global R² across all dimensions."""
    resid = y_true - y_pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")


def compute_filtered_r2_per_dim(S_true: np.ndarray, S_pred: np.ndarray, sst_min=SST_MIN):
    """
    S_true, S_pred: (T, N, K) over evaluation window.
    Returns mean R² across dims with SST > threshold, and used count.
    """
    T, n_agents, num_opts = S_true.shape
    r2_list = []
    for i in range(n_agents):
        for k in range(num_opts):
            y = S_true[:, i, k]
            yhat = S_pred[:, i, k]
            ss_res = float(np.sum((y - yhat) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            if ss_tot > sst_min:
                r2_list.append(1.0 - ss_res / ss_tot)
    return (float(np.mean(r2_list)) if r2_list else float("nan")), len(r2_list), n_agents * num_opts


def main():
    belief_files = find_belief_logs(BELIEF_DIR)
    if not belief_files:
        print(f"No belief_logs found in {BELIEF_DIR}")
        return

    print(f"Found {len(belief_files)} belief CSV files across experiments")
    print("Fitting FJ COMPLETE model on ALL rounds (no train/test split)\n")

    fit_rows = []
    params_rows = []

    for experiment_name, path in belief_files:
        try:
            df, n_agents, options = parse_belief_csv(path)
            if n_agents < 2:
                print(f"⚠ {experiment_name}/{path.name}: N={n_agents} too small; skipping")
                continue

            state_cols = [f"Agent_{ai}_{o}" for ai in range(n_agents) for o in options]
            S_full_flat = df[state_cols].to_numpy()
            S_full = S_full_flat.reshape(len(df), n_agents, K)

            T = len(df)
            if T < 2:
                print(f"⚠ {experiment_name}/{path.name}: not enough rounds (has {T}); skipping")
                continue

            b0 = S_full[0]

            gamma0 = np.full(n_agents, 0.2, dtype=float)
            alpha0 = np.full(n_agents, 0.5, dtype=float)
            w_logits0 = np.zeros(n_agents * (n_agents - 1), dtype=float)
            theta0 = np.concatenate([gamma0, alpha0, w_logits0])

            bounds = [(0.0, 1.0)] * (2 * n_agents) + [(None, None)] * (n_agents * (n_agents - 1))

            res = minimize(
                fj_loss_complete,
                theta0,
                args=(S_full, b0),
                method="L-BFGS-B",
                bounds=bounds,
                options=dict(maxiter=3000, ftol=1e-12),
            )

            gamma_hat, alpha_hat, W_hat = unpack_theta_complete(res.x, n_agents)

            fitted_pred_steps = []
            for t in range(T - 1):
                Bt = S_full[t]
                mix = W_hat @ Bt
                Bt_next = (
                    gamma_hat[:, None] * b0
                    + (1 - gamma_hat)[:, None] * (
                        alpha_hat[:, None] * Bt
                        + (1 - alpha_hat)[:, None] * mix
                    )
                )
                fitted_pred_steps.append(Bt_next)

            fitted_pred = np.stack(fitted_pred_steps, axis=0)
            fitted_true = S_full[1:]

            mse = float(np.mean((fitted_true - fitted_pred) ** 2))
            rmse = float(np.sqrt(mse))
            mae = float(np.mean(np.abs(fitted_true - fitted_pred)))
            r2_global = compute_global_r2(fitted_true.reshape(-1), fitted_pred.reshape(-1))
            r2_filt, r2_used, r2_total = compute_filtered_r2_per_dim(fitted_true, fitted_pred)

            for i in range(n_agents):
                neighbor_influence = float((1 - gamma_hat[i]) * (1 - alpha_hat[i]))
                rec = {
                    "experiment": experiment_name,
                    "question_file": path.name,
                    "agent": i,
                    "is_hub": int(i == 0),
                    "gamma": float(gamma_hat[i]),
                    "alpha": float(alpha_hat[i]),
                    "neighbor_influence": neighbor_influence,
                }
                for j in range(n_agents):
                    rec[f"W_{j}"] = float(W_hat[i, j])
                params_rows.append(rec)

            fit_rows.append(
                {
                    "experiment": experiment_name,
                    "question_file": path.name,
                    "num_agents": n_agents,
                    "num_rounds": int(T),
                    "hub_idx": 0,
                    "topology": "complete",
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae,
                    "r2_global": r2_global,
                    "r2_filtered_mean": r2_filt,
                    "r2_dims_used": int(r2_used),
                    "r2_dims_total": int(r2_total),
                    "optimizer_success": bool(res.success),
                    "optimizer_iterations": int(res.nit),
                    "optimizer_fun": float(res.fun),
                }
            )

            print(
                f"✓ {experiment_name}/{path.name}: "
                f"MSE={mse:.3e}, R²g={r2_global:.3f}, R²f={r2_filt:.3f} "
                f"(nit={int(res.nit)})"
            )

        except Exception as e:
            print(f"✗ {experiment_name}/{path.name}: {e}")

    if fit_rows:
        out_summary = BELIEF_DIR / OUT_FIT_SUMMARY
        pd.DataFrame(fit_rows).to_csv(out_summary, index=False)
        print(f"\n✓ Saved fit summary to {out_summary}")
    if params_rows:
        out_params = BELIEF_DIR / OUT_PARAMS
        pd.DataFrame(params_rows).to_csv(out_params, index=False)
        print(f"✓ Saved parameters to {out_params}")

    print("\nDone.")


if __name__ == "__main__":
    main()
