# Don't Trust Stubborn Neighbors: A Security Framework for Agentic Networks

Code for the paper "Don't Trust Stubborn Neighbors: A Security Framework for Agentic Networks".

LLM-based multi-agent systems (MAS) are increasingly deployed for agentic tasks, but their interactive nature introduces security risks: malicious agents can exploit communication channels to propagate misinformation and manipulate collective outcomes. We study how such manipulation arises and spreads using the Friedkin–Johnsen opinion formation model from social sciences as a theoretical framework for LLM-MAS. We find that a single highly stubborn and persuasive agent can trigger a persuasion cascade that takes over group dynamics. To counter this, we propose a trust-adaptive defense that dynamically adjusts inter-agent trust to limit adversarial influence while preserving cooperative performance.

---

## Structure

```
cascade/
├── core/          # LLM backends, prompt templates
├── experiments/
│   ├── csqa/      # CommonsenseQA runner, agent graphs, trust module
│   └── toolbench/ # ToolBench runner
└── analysis/      # ASR computation, belief fitting, figures
configs/           # YAML experiment configs
data/              # JSONL datasets (csqa, toolbench)
```

---

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=https://api.openai.com/v1

# Only if using Gemini backend
GOOGLE_CLOUD_PROJECT=your_project_id
```

---

## Running Experiments

Experiments are driven by YAML config files. Each file has a `defaults` block and a `configs` list of scenarios.

```bash
# CommonsenseQA and ToolBench
python -m cascade.experiments.csqa --config configs/qwen3-235b_all_experiments.yaml

```

### Backends

Set `backend` in your config to one of:

| Backend | Notes |
|---------|-------|
| `openai` | Any OpenAI-compatible API (default) |
| `vllm` | Self-hosted via vLLM server |
| `gemini` | Google Vertex AI |

### Trust experiments

| Suffix | Name | Description |
|--------|------|-------------|
| `exp1` | T-W | Warmup + fixed trust, static attacker |
| `exp2` | T-WA | Warmup + fixed trust, adaptive attacker |
| `exp3` | T-WS | Warmup + sparse trust updates, adaptive attacker |
| `exp4` | T-S | Random sparse trust, no warmup, static attacker |

---

## Analysis

```bash
# Attack Success Rate across all completed runs
python -m cascade.analysis.compute_asr --output-dir output


Results are written to `output/{model}/{dataset}/{scenario}/summaries/`.

---


## Data

| File | Description |
|------|-------------|
| `data/csqa_100.jsonl` | 100-question CommonsenseQA subset |
| `data/toolbench_100.jsonl` | 100-question ToolBench subset |