#!/usr/bin/env python3
"""
FJ Star Topology Prediction: Fixed-parameter and Incremental.

Star topology:
  - Hub (agent 0) connected to all leaves
  - Leaves connected only to hub
  - Hub neighbor weights: W[hub, leaf_j] = w_j  (learned, sum to 1)
  - Leaf neighbor weights: W[leaf, hub]   = 1    (fixed)

Model (per agent, vectorized over K answer options):
  Hub:  b_c(t+1) = γ_c * s_c + (1-γ_c) * [α_c * b_c(t) + (1-α_c) * Σ_{j∈leaves} w_j * b_j(t)]
  Leaf: b_i(t+1) = γ_i * s_i + (1-γ_i) * [α_i * b_i(t) + (1-α_i) * b_c(t)]

where s_i = b_i(0) is the stubborn opinion (round-0 belief).

Two modes:
  1. Fixed:       fit on rounds 0..TRAIN_ROUNDS-1, multi-step rollout for rounds PREDICT_START+
  2. Incremental: fit on rounds 0..TRAIN_ROUNDS-1, then online gradient update per new round

Usage:
  python star_predict.py <belief_logs_dir>             # single directory
  python star_predict.py <root_dir> --recursive        # find all belief files under root_dir
"""

from pathlib import Path
from io import StringIO
from collections import defaultdict
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ── CONFIG ────────────────────────────────────────────────────────────────────
BELIEF_DIR    = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
RECURSIVE     = "--recursive" in sys.argv or "-r" in sys.argv
K             = 5      # number of answer options
TRAIN_ROUNDS  = 8      # fit on rounds 0..TRAIN_ROUNDS-1
PREDICT_START = 8      # evaluate predictions from this round onward
HUB_IDX       = 0      # index of the hub agent
SST_MIN       = 1e-6   # min total variance for filtered R²

INCREMENTAL_LR    = 0.01
INCREMENTAL_STEPS = 50

# Results are saved to .../scenario/run_id/predict/ (sibling of belief_logs)
# ─────────────────────────────────────────────────────────────────────────────


# ── DATA LOADING ──────────────────────────────────────────────────────────────

def parse_belief_csv(path: Path):
    """Load a consolidated belief CSV → (DataFrame, N agents, option labels)."""
    lines = path.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("round,"))
    header = lines[start + 1].split(",")

    remaining = len(header) - 1
    assert remaining % K == 0, f"{path.name}: header length mismatch"
    N = remaining // K
    options = header[1:1 + K]

    colnames   = ["round"] + [f"Agent_{a}_{o}" for a in range(N) for o in options]
    data_block = "\n".join(lines[start + 2:])
    df = pd.read_csv(StringIO(",".join(colnames) + "\n" + data_block))
    df["round"] = df["round"].str.replace("round", "", regex=False).astype(int)
    df = df.sort_values("round").reset_index(drop=True)
    return df, N, options


# ── FJ MODEL HELPERS ──────────────────────────────────────────────────────────

def stable_softmax(v: np.ndarray) -> np.ndarray:
    v = v - np.max(v)
    ex = np.exp(v)
    s  = ex.sum()
    return ex / s if s > 0 else np.full_like(ex, 1.0 / len(ex))


def unpack_theta(theta: np.ndarray, N: int):
    """
    Unpack flat parameter vector into (gamma, alpha, W).

    theta layout:  [gamma × N | alpha × N | hub_logits × (N-1)]
      - gamma, alpha in [0,1] (enforced via optimizer bounds)
      - hub_logits → softmax → hub neighbor weights over leaves (sum to 1)
      - Leaf rows: W[leaf, hub] = 1 (fixed, not in theta)
    """
    gamma = theta[:N]
    alpha = theta[N:2 * N]

    leaves = [i for i in range(N) if i != HUB_IDX]
    hub_w  = stable_softmax(theta[2 * N: 2 * N + (N - 1)])

    W = np.zeros((N, N))
    for idx, leaf in enumerate(leaves):
        W[HUB_IDX, leaf] = hub_w[idx]   # hub → each leaf
    for leaf in leaves:
        W[leaf, HUB_IDX] = 1.0          # leaf → hub (fixed)

    return gamma, alpha, W


def fj_one_step(gamma: np.ndarray, alpha: np.ndarray, W: np.ndarray,
                b0: np.ndarray, b_prev: np.ndarray) -> np.ndarray:
    """One-step FJ update: b(t+1) from b(t).  All arrays (N, K)."""
    mix = W @ b_prev   # (N, K): weighted neighbor beliefs
    return (
        gamma[:, None] * b0
        + (1 - gamma)[:, None] * (alpha[:, None] * b_prev + (1 - alpha)[:, None] * mix)
    )


def train_loss(theta: np.ndarray, S_train: np.ndarray, b0: np.ndarray) -> float:
    """One-step teacher-forced MSE across all transitions in the training window."""
    T, N, _ = S_train.shape
    gamma, alpha, W = unpack_theta(theta, N)
    err = []
    for t in range(T - 1):
        pred = fj_one_step(gamma, alpha, W, b0, S_train[t])
        err.append((S_train[t + 1] - pred).reshape(-1))
    return float(np.mean(np.concatenate(err) ** 2))


def single_step_loss(theta: np.ndarray, b_prev: np.ndarray,
                     b_true: np.ndarray, b0: np.ndarray, N: int) -> float:
    """MSE for a single transition (used in incremental gradient updates)."""
    gamma, alpha, W = unpack_theta(theta, N)
    pred = fj_one_step(gamma, alpha, W, b0, b_prev)
    return float(np.mean((b_true - pred) ** 2))


def numerical_gradient(theta: np.ndarray, b_prev: np.ndarray,
                        b_true: np.ndarray, b0: np.ndarray,
                        N: int, eps: float = 1e-5) -> np.ndarray:
    """Finite-difference gradient of single_step_loss."""
    grad = np.zeros_like(theta)
    f0   = single_step_loss(theta, b_prev, b_true, b0, N)
    for i in range(len(theta)):
        tp    = theta.copy()
        tp[i] += eps
        grad[i] = (single_step_loss(tp, b_prev, b_true, b0, N) - f0) / eps
    return grad


def clip_params(theta: np.ndarray, N: int) -> np.ndarray:
    """Clip gamma and alpha to [0, 1] after a gradient step."""
    theta[:2 * N] = np.clip(theta[:2 * N], 0.0, 1.0)
    return theta


# ── METRICS ──────────────────────────────────────────────────────────────────

def global_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")


def filtered_r2(S_true: np.ndarray, S_pred: np.ndarray):
    """
    Per-(agent, option) R², averaged only over dimensions where SST > SST_MIN.
    Returns (mean_r2, n_used_dims, n_total_dims).
    """
    T, N, K_ = S_true.shape
    r2s = []
    for i in range(N):
        for k in range(K_):
            y, yh  = S_true[:, i, k], S_pred[:, i, k]
            ss_res = float(np.sum((y - yh) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            if ss_tot > SST_MIN:
                r2s.append(1.0 - ss_res / ss_tot)
    mean = float(np.mean(r2s)) if r2s else float("nan")
    return mean, len(r2s), N * K_


# ── INITIAL FIT (shared by both modes) ───────────────────────────────────────

def initial_fit(S_train: np.ndarray, b0: np.ndarray, N: int):
    """Fit FJ parameters on the training window via L-BFGS-B."""
    theta0 = np.concatenate([np.full(N, 0.2), np.full(N, 0.5), np.zeros(N - 1)])
    bounds = [(0.0, 1.0)] * (2 * N) + [(None, None)] * (N - 1)
    return minimize(
        train_loss, theta0, args=(S_train, b0),
        method="L-BFGS-B", bounds=bounds,
        options=dict(maxiter=3000, ftol=1e-12),
    )


# ── MODE 1: FIXED-PARAMETER ───────────────────────────────────────────────────

def run_fixed(S_full: np.ndarray, rounds: np.ndarray, N: int,
              b0: np.ndarray, question_file: str):
    """
    Fit on rounds 0..TRAIN_ROUNDS-1, then multi-step rollout from round
    TRAIN_ROUNDS-1 and evaluate against empirical rounds PREDICT_START+.
    """
    T_total = len(S_full)
    S_train = S_full[:TRAIN_ROUNDS]

    res           = initial_fit(S_train, b0, N)
    gamma, alpha, W = unpack_theta(res.x, N)

    # ---- Training metrics (one-step teacher-forced) ----
    train_preds = [fj_one_step(gamma, alpha, W, b0, S_train[t])
                   for t in range(TRAIN_ROUNDS - 1)]
    train_pred  = np.stack(train_preds)          # (TRAIN_ROUNDS-1, N, K)
    train_true  = S_train[1:]
    train_mse   = float(np.mean((train_true - train_pred) ** 2))
    train_r2g   = global_r2(train_true.reshape(-1), train_pred.reshape(-1))
    train_r2f, train_used, train_dims = filtered_r2(train_true, train_pred)

    # ---- Multi-step rollout from last training round ----
    b_curr  = S_full[TRAIN_ROUNDS - 1].copy()
    rollout = [b_curr]
    for _ in range(T_total - TRAIN_ROUNDS):
        b_curr = fj_one_step(gamma, alpha, W, b0, b_curr)
        rollout.append(b_curr)

    offset     = max(1, PREDICT_START - (TRAIN_ROUNDS - 1))
    pred_eval  = np.stack(rollout[offset:])
    true_eval  = S_full[(TRAIN_ROUNDS - 1) + offset:]
    eval_rounds = rounds[(TRAIN_ROUNDS - 1) + offset:]

    test_mse = float(np.mean((true_eval - pred_eval) ** 2))
    test_r2g = global_r2(true_eval.reshape(-1), pred_eval.reshape(-1))
    test_r2f, test_used, test_dims = filtered_r2(true_eval, pred_eval)

    # ---- Build output rows ----
    summary = {
        "question_file": question_file, "num_agents": N,
        "num_rounds_total": T_total, "topology": "star",
        "train_mse": train_mse, "train_rmse": float(np.sqrt(train_mse)),
        "train_r2_global": train_r2g, "train_r2_filtered": train_r2f,
        "train_r2_dims_used": train_used, "train_r2_dims_total": train_dims,
        "test_mse": test_mse, "test_rmse": float(np.sqrt(test_mse)),
        "test_mae": float(np.mean(np.abs(true_eval - pred_eval))),
        "test_r2_global": test_r2g, "test_r2_filtered": test_r2f,
        "test_r2_dims_used": test_used, "test_r2_dims_total": test_dims,
        "optimizer_success": bool(res.success),
        "optimizer_iterations": int(res.nit),
        "optimizer_fun": float(res.fun),
    }

    params = []
    for i in range(N):
        row = {"question_file": question_file, "agent": i,
               "gamma_hat": float(gamma[i]), "alpha_hat": float(alpha[i])}
        row.update({f"W_{j}": float(W[i, j]) for j in range(N)})
        params.append(row)

    predictions = []
    for idx, r in enumerate(eval_rounds):
        row = {"question_file": question_file, "round": int(r)}
        row.update({f"emp_{d}":  float(v) for d, v in enumerate(true_eval[idx].reshape(-1))})
        row.update({f"pred_{d}": float(v) for d, v in enumerate(pred_eval[idx].reshape(-1))})
        predictions.append(row)

    perround = [
        {"question_file": question_file, "round": int(r),
         "mse": float(np.mean((true_eval[i] - pred_eval[i]) ** 2))}
        for i, r in enumerate(eval_rounds)
    ]

    return summary, params, predictions, perround


# ── MODE 2: INCREMENTAL ───────────────────────────────────────────────────────

def run_incremental(S_full: np.ndarray, rounds: np.ndarray, N: int,
                    b0: np.ndarray, question_file: str):
    """
    Fit on rounds 0..TRAIN_ROUNDS-1, then for each subsequent round:
      1. Predict using current parameters
      2. Observe true round
      3. Update parameters via gradient descent on that single transition
    """
    T_total = len(S_full)
    S_train = S_full[:TRAIN_ROUNDS]

    res   = initial_fit(S_train, b0, N)
    theta = res.x.copy()

    perround  = []
    all_true  = []
    all_pred  = []

    for t in range(TRAIN_ROUNDS, T_total):
        gamma, alpha, W = unpack_theta(theta, N)

        b_prev = S_full[t - 1]
        b_pred = fj_one_step(gamma, alpha, W, b0, b_prev)
        b_true = S_full[t]

        mse = float(np.mean((b_true - b_pred) ** 2))
        mae = float(np.mean(np.abs(b_true - b_pred)))
        all_true.append(b_true.reshape(-1))
        all_pred.append(b_pred.reshape(-1))

        perround.append({
            "question_file": question_file,
            "train_rounds": t,
            "predict_round": int(rounds[t]),
            "test_mse": mse,
            "test_mae": mae,
        })

        # Update parameters on the new transition
        for _ in range(INCREMENTAL_STEPS):
            grad  = numerical_gradient(theta, b_prev, b_true, b0, N)
            theta = clip_params(theta - INCREMENTAL_LR * grad, N)

    test_r2 = global_r2(np.concatenate(all_true), np.concatenate(all_pred))
    test_mses = [r["test_mse"] for r in perround]
    test_maes = [r["test_mae"] for r in perround]

    summary = {
        "question_file": question_file, "num_agents": N,
        "num_rounds_total": T_total, "min_train_rounds": TRAIN_ROUNDS,
        "num_test_rounds": T_total - TRAIN_ROUNDS,
        "initial_train_mse": float(res.fun),
        "mean_test_mse": float(np.mean(test_mses)),
        "mean_test_mae": float(np.mean(test_maes)),
        "test_r2_global": test_r2,
    }

    gamma_f, alpha_f, W_f = unpack_theta(theta, N)
    params = []
    for i in range(N):
        row = {"question_file": question_file, "agent": i,
               "gamma_hat": float(gamma_f[i]), "alpha_hat": float(alpha_f[i])}
        row.update({f"W_{j}": float(W_f[i, j]) for j in range(N)})
        params.append(row)

    return summary, perround, params


# ── SCENARIO HELPERS ─────────────────────────────────────────────────────────

def scenario_name(path: Path) -> str:
    """Scenario dir name: 3 levels above the belief file (.../scenario/run/belief_logs/file)."""
    if path.parent.name == "belief_logs":
        return path.parent.parent.parent.name
    return path.parent.name


def predict_dir(path: Path) -> Path:
    """Output dir: sibling of belief_logs, i.e. .../scenario/run/predict/"""
    if path.parent.name == "belief_logs":
        return path.parent.parent / "predict"
    return path.parent / "predict"


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    if RECURSIVE:
        files = sorted(BELIEF_DIR.rglob("*_beliefs.csv"))
        print(f"Recursive search under: {BELIEF_DIR}")
    else:
        files = sorted(BELIEF_DIR.glob("*_beliefs.csv"))

    if not files:
        print(f"No *_beliefs.csv files found {'under' if RECURSIVE else 'in'} {BELIEF_DIR}")
        return

    # Group files by their belief_logs directory (one predict/ folder per run)
    groups: dict = defaultdict(list)
    for path in files:
        groups[path.parent].append(path)

    print(f"Processing {len(files)} files across {len(groups)} run(s)  [star topology]")
    print(f"Train: rounds 0..{TRAIN_ROUNDS - 1}  |  Predict from: round {PREDICT_START}\n")

    for belief_logs_dir, group_files in sorted(groups.items()):
        scenario = scenario_name(group_files[0])
        out_dir  = predict_dir(group_files[0])
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"── {scenario}  ({len(group_files)} files)  →  {out_dir}")

        fixed_summaries, fixed_params, fixed_preds, fixed_perround = [], [], [], []
        inc_summaries,   inc_perround,  inc_params                 = [], [], []

        for path in sorted(group_files):
            question_id = path.name
            try:
                df, N, options = parse_belief_csv(path)

                if N < 2:
                    print(f"    skip {question_id}: N={N} (need ≥ 2)"); continue
                if len(df) < TRAIN_ROUNDS:
                    print(f"    skip {question_id}: only {len(df)} rounds (need ≥ {TRAIN_ROUNDS})"); continue
                if len(df) <= PREDICT_START:
                    print(f"    skip {question_id}: no rounds after predict start {PREDICT_START}"); continue

                state_cols = [f"Agent_{a}_{o}" for a in range(N) for o in options]
                S_full     = df[state_cols].to_numpy().reshape(len(df), N, K)
                rounds     = df["round"].to_numpy()
                b0         = S_full[0]

                fs, fp, fpr, fperr = run_fixed(S_full, rounds, N, b0, question_id)
                fixed_summaries.append(fs)
                fixed_params.extend(fp)
                fixed_preds.extend(fpr)
                fixed_perround.extend(fperr)

                isumm, iperr, ipar = run_incremental(S_full, rounds, N, b0, question_id)
                inc_summaries.append(isumm)
                inc_perround.extend(iperr)
                inc_params.extend(ipar)

                print(
                    f"    {question_id}:  "
                    f"fixed R²={fs['test_r2_global']:.3f}  |  "
                    f"inc MSE={isumm['mean_test_mse']:.3e}"
                )

            except Exception as e:
                print(f"    ERROR {question_id}: {e}")

        def save(rows, name):
            if rows:
                pd.DataFrame(rows).to_csv(out_dir / name, index=False)
                print(f"    saved {name}")

        print()
        save(fixed_summaries, "star_fixed_summary.csv")
        save(fixed_params,    "star_fixed_params.csv")
        save(fixed_preds,     "star_fixed_predictions.csv")
        save(fixed_perround,  "star_fixed_perround.csv")
        save(inc_summaries,   "star_incremental_summary.csv")
        save(inc_perround,    "star_incremental_perround.csv")
        save(inc_params,      "star_incremental_params.csv")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
