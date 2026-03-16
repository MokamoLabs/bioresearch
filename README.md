# BioResearch: Autonomous Biology ML Research

Three autonomous ML research products chained into one pipeline: **molecules → cells → patients**.

An LLM agent (Claude Opus 4.6) autonomously modifies training code, runs multi-seed experiments on cloud GPUs, evaluates against frozen metrics with statistical rigor, and keeps or reverts changes. ~100 experiments per overnight campaign.

## Architecture

```
bioresearch/
├── engine/          # Core autoresearch loop, orchestrator, population search, metrics
├── infra/           # Modal GPU dispatch, Colab setup
├── knowledge/       # Biological knowledge retrieval (gene embeddings, PPI, pathways)
├── domains/
│   ├── perturbation/  # AutoPerturb: predict post-perturbation gene expression
│   ├── molecules/     # AutoMol: 22-endpoint ADMET profiling + molecular generation
│   └── trials/        # AutoTrial: clinical trial success prediction (chains all three)
├── web/             # Gradio web UI
├── cli.py           # CLI: search | predict | serve
└── tests/           # 38 tests covering engine, metrics, and all domains
```

Each domain has:
- **`prepare.py`** (FROZEN): Data loading, preprocessing, evaluation metrics. Never modified.
- **`train.py`** (MUTABLE): The model code the agent autonomously improves.

## Quick Start

```bash
# Install
uv sync

# Run any domain's baseline
uv run python domains/perturbation/train.py
uv run python domains/molecules/train.py
uv run python domains/trials/train.py

# Run tests
uv run pytest tests/ -v

# CLI
uv run python cli.py predict --domain molecules --input "c1ccccc1O"
uv run python cli.py predict --domain trials
uv run python cli.py serve  # Launch Gradio UI on port 7860
```

## The Three Products

### AutoPerturb
Predict post-perturbation gene expression for unseen perturbations.

| Metric | Role | Direction |
|---|---|---|
| `pearson_deg` | PRIMARY | Higher (Pearson r on top-20 DEGs) |
| `mse_top20_deg` | GUARD | Lower (must not degrade >10%) |
| `direction_acc` | GUARD | Higher (must stay >0.7) |
| `cross_context` | BONUS | Lower (generalization gap) |

### AutoMol
22-endpoint ADMET profiling with clinical importance weighting (hERG and DILI get 2x). Includes a molecular generation layer.

| Metric | Role | Direction |
|---|---|---|
| `composite_admet` | PRIMARY | Higher (weighted average across 22 endpoints) |
| `gen_profile_match` | PRIMARY | Higher (generated molecules matching target profile) |
| `gen_diversity` | GUARD | Higher (>0.7 Tanimoto diversity) |

### AutoTrial
Clinical trial success prediction. Chains AutoMol (ADMET features) and AutoPerturb (perturbation signatures) as input features.

| Metric | Role | Direction |
|---|---|---|
| `auroc` | PRIMARY | Higher (discrimination) |
| `calibration_ece` | GUARD | Lower (<0.15) |
| `net_value` | GUARD | Higher (>0: $1B success, -$200M failure) |
| `lift_at_10` | BONUS | Higher (concentration in top 10%) |

## Running Overnight Campaigns

The intended deployment: **Colab orchestrates, Modal executes**.

```
┌──────────────────────────┐
│   Google Colab (H100)    │  Orchestration:
│  ┌────────────────────┐  │  - Claude API calls (propose modifications)
│  │  overnight_campaign│──┼──> keep/revert state machine
│  │  .ipynb            │  │  - experiment logging & plots
│  └────────┬───────────┘  │
└───────────┼──────────────┘
            │ dispatches via Modal SDK
            ▼
┌──────────────────────────┐
│   Modal (H200 GPUs)      │  Execution:
│  ┌──────┐ ┌──────┐       │  - 5 seeds × 1 modification
│  │seed=0│ │seed=1│ ...   │  - 10-min training budget each
│  │train │ │train │       │  - returns metric vectors
│  └──────┘ └──────┘       │  - ~12 experiments/hour
└──────────────────────────┘
```

### Prerequisites

1. **Anthropic API key** for Claude agent
2. **Modal account** — sign up at [modal.com](https://modal.com), run `modal setup` locally to get your token ID and secret
3. **Google Colab** (Pro/Pro+ recommended for overnight runtime)

### Quick Start: Open the Notebook

The campaign notebook is at **`notebooks/overnight_campaign.ipynb`**. Upload it to Colab, or:

1. Push this repo to GitHub
2. Open `notebooks/overnight_campaign.ipynb` in Colab via GitHub URL

The notebook walks through setup, authentication, domain selection, and launching the loop.

### Step-by-Step

**1. Set credentials in Colab Secrets** (key icon in left sidebar):
- `ANTHROPIC_API_KEY` = your Anthropic key
- `MODAL_TOKEN_ID` = your Modal token ID
- `MODAL_TOKEN_SECRET` = your Modal token secret

**2. Configure the campaign** (cell 4 of the notebook):
```python
DOMAIN = 'perturbation'   # which domain to optimize
ITERATIONS = 100           # ~100 experiments overnight
NUM_SEEDS = 5              # statistical rigor
TIME_BUDGET = 600          # seconds per seed on Modal
USE_MODAL = True           # dispatch to Modal GPUs
POPULATION = 0             # 0 = single agent, 4 = population search
```

**3. Run and walk away.** The notebook prints live progress. Results auto-save to `results/<domain>/`.

### Keeping Colab Alive Overnight

Colab disconnects after ~90 minutes of inactivity. To prevent this:
- **Keep the browser tab open** and visible
- Use [Colab Keep Alive](https://chrome.google.com/webstore/detail/colab-alive/eihnaflcafllhmdojdknpfcjnbighfkb) browser extension
- **Colab Pro+** has longer runtime limits and background execution

### What Happens During a Campaign

Each iteration (~5-10 min with Modal):
1. Claude reads current `train.py` + experiment history + bio knowledge
2. Proposes a focused modification with hypothesis
3. Modified code dispatches to 5 Modal H200 GPUs in parallel
4. Results evaluated: Welch's t-test (p < 0.05), Cohen's d (> 0.3), guard rails
5. KEEP if statistically + meaningfully better, REVERT otherwise
6. Repeat

### Population Search

For broader architecture exploration, run K agents in parallel with tournament selection:
```python
POPULATION = 4  # 4 agents, bottom 25% adopts top 25% code every 10 iterations
```

### Monitoring Results

Results save to `results/<domain>/`:
- `experiments.tsv` — full experiment log
- `campaign_summary.json` — best metrics, total iterations
- `metrics_plot.png` — metric progression over iterations

### Recommended Campaign Order

1. **perturbation** first — fastest iteration, smallest model
2. **molecules** second — 22-endpoint ADMET, more complex search space
3. **trials** last — benefits from improved molecules/perturbation models via cross-project features

## Optional Dependencies

```bash
uv sync --extra bio    # scanpy, anndata, pertpy (perturbation data)
uv sync --extra chem   # rdkit, PyTDC (molecular fingerprints, ADMET benchmarks)
uv sync --extra gpu    # torch (GPU models)
uv sync --extra web    # gradio (web UI)
uv sync --extra infra  # modal (cloud GPU dispatch)
```

## Pre-computing Knowledge Sources

Before running campaigns, optionally pre-compute biological knowledge embeddings:

```bash
uv run python -m knowledge.precompute --all
```

This computes gene text embeddings, Gene Ontology vectors, PPI network features, pathway memberships, ESM structure embeddings, and drug-target matrices. The agent can then selectively incorporate these priors during search.
