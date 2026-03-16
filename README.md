<div align="center">

# Don't Trust Stubborn Neighbors: 
# A Security Framework for Adversarial Multi-Agent Networks

*Code for the paper*

![Teaser](Fig/teaser_fig.png)

</div>

---

Multi-agent LLM systems are vulnerable to adversarial agents that attempt to manipulate group decisions through repeated discussion. This repository provides a framework for:

- **Running adversarial multi-agent experiments** on CommonsenseQA and ToolBench benchmarks across different network topologies
- **Evaluating trust-adaptive defenses** that dynamically reweight peer influence to isolate malicious agents
- **Fitting the Friedkin–Johnsen (FJ) belief dynamics model** to LLM agent behavior to analytically predict adversarial outcomes

---

## Repository Structure

```
cascade/
├── core/
│   ├── methods.py          # LLM client/backend adapters (OpenAI, vLLM, Gemini)
│   ├── prompts.py          # Agent prompt templates (persuasion & agreeableness traits)
│   └── utils.py            # Belief normalization, adjacency matrix generation
├── experiments/
│   ├── config.py           # YAML/JSON config loader
│   ├── batch.py            # Multi-scenario batch runner
│   ├── trust.py            # Trust matrix construction and updates
│   ├── csqa/
│   │   ├── agents.py       # Agent and AgentGraph classes
│   │   ├── runnerCQ.py     # CommonsenseQA orchestrator
│   │   └── cli.py          # CLI argument parser
│   └── toolbench/
│       ├── agents.py       # ToolBench task builder
│       ├── runnerTB.py     # ToolBench orchestrator
│       └── cli.py
└── analysis/
    ├── compute_asr.py              # Attack Success Rate (ASR) computation
    ├── consolidate_belief_logs.py  # Belief trajectory aggregation
    ├── fit_fj_complete_full.py     # FJ model fitting for complete graphs
    ├── fit_fj_star_full.py         # FJ model fitting for star graphs
    ├── star_predict.py             # FJ predictions for star topology
    ├── complete_predict.py         # FJ predictions for complete topology
    └── summarize_beliefs.py        # Belief summary generation
configs/                    # YAML experiment configs
data/                       # JSONL datasets (csqa, toolbench)
monitor.py                  # Scenario completion monitor
```

---

## Setup

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Create a `.env` file in the project root:**

```env
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=https://api.openai.com/v1

# Only required for Gemini backend
GOOGLE_CLOUD_PROJECT=your_project_id
```

---

## Running Experiments

Experiments are driven by YAML config files. Each config has a `defaults` block and a `configs` list of scenarios.

```bash
# CommonsenseQA
python -m cascade.experiments.csqa --config configs/qwen3-235b_all_experiments.yaml

# ToolBench
python -m cascade.experiments.toolbench --config configs/qwen3-235b_all_experiments.yaml
```

To run a single experiment without a config file, pass all options directly on the CLI:

```bash
# Benign baseline — 6 agents, complete graph, no attacker
python -m cascade.experiments.csqa \
  --model gpt-4o-mini \
  --backend openai \
  --graph complete \
  --agents 6 \
  --rounds 10 \
  --dataset csqa_100

# Adversarial — 1 attacker (high persuasion, low agreeableness) and 5 benign agents
python -m cascade.experiments.csqa \
  --model gpt-4o-mini \
  --backend openai \
  --graph complete \
  --agents 6 \
  --attackers 1 \
  --placement 0 \
  --rounds 10 \
  --dataset csqa_100 \
  --persuasion-levels "high_v3,low_v3,low_v3,low_v3,low_v3,low_v3" \
  --agreeableness-levels "low_v3,high_v3,high_v3,high_v3,high_v3,high_v3" \
  --no-trust \
  --scenario complete-6a-Ihigh-Alow-benign-Ilow-Ahigh-v3-csqa
```

The `--scenario` flag sets the output folder name under `output/`. 

### Backends

| Backend | Description |
|---------|-------------|
| `openai` | Any OpenAI-compatible API (default) |
| `vllm` | Self-hosted open-weight models via vLLM |
| `gemini` | Google Vertex AI |
| `blablador` | Helmholtz Blablador API |

### Network Topologies

| `--graph` | Description |
|-----------|-------------|
| `complete` | Fully connected — all agents hear all others |
| `pure_star` | Hub + isolated leaves — only hub aggregates |


### Trust Experiment Configurations

| Config | Name | Description |
|--------|------|-------------|
| `exp1` | T-W  | Warmup + fixed trust, static attacker |
| `exp2` | T-WA | Warmup + fixed trust, adaptive attacker |
| `exp3` | T-WS | Warmup + sparse trust updates, adaptive attacker |
| `exp4` | T-S  | Random sparse trust, no warmup, static attacker |

**Attacker types:**
- **Static** — defends the incorrect answer every round
- **Adaptive** — mimics a benign agent during warmup, then pivots to attack

---

## Analysis

```bash
# Compute Attack Success Rate across all completed runs
python -m cascade.analysis.compute_asr --output-dir output

# Aggregate belief trajectory logs
python -m cascade.analysis.consolidate_belief_logs --output-dir output

# Fit Friedkin–Johnsen model (complete graphs)
python cascade/analysis/fit_fj_complete_full.py output/

# Fit Friedkin–Johnsen model (star graphs)
python cascade/analysis/fit_fj_star_full.py output/

```

**ASR definition:**
```
ASR = (# benign agents initially correct who flip to wrong answer at final round)
    / (# benign agents initially correct)
```

Results are written to `output/{model}/{dataset}/{scenario}/summaries/`.

---

## Output Structure

```
output/
└── {model}/
    └── {dataset}/
        └── {scenario}/
            └── {sample_id}/
                ├── records/        # Per-question JSON logs (full dialogue)
                ├── summaries/      # ASR and accuracy summaries
                ├── belief_logs/    # Belief trajectory CSVs (per agent, per round)
                └── trust_logs/     # Trust matrix evolution
```

---

## Data

| File | Description |
|------|-------------|
| `data/csqa_100.jsonl` | 100-question CommonsenseQA subset |
| `data/toolbench_100.jsonl` | 100-question ToolBench subset |

Questions follow the standard CSQA format:

```json
{
  "id": "...",
  "question": "What do people aim to do at work?",
  "choices": {"label": ["A","B","C","D","E"], "text": ["complete", "learn", ...]},
  "answerKey": "A"
}
```

---

## Citation

If you use this code, please cite:

```bibtex
 
TO BE UPDATED

```