# Kaggle notebooks

The experiments were run on Kaggle (GPU T4 x2, 12-hour sessions).

| Notebook | Runs | Extra outputs |
| --- | --- | --- |
| `kaggle_experiments.ipynb` | `PLAN = "mix"` (proposed TSR/PTM, 3 seeds) by default; also `"sweep"`, `"mini"`, `"confirm"` | `proposed_methods.csv`, `mixing_weights_summary.csv`, `segment_weights_summary.csv` |
| `kaggle_continual.ipynb` | `PLAN = "novel+mini"`: fixed task-decoupled heads + federated continual onboarding | `forgetting_overview.csv`, `forgetting_curves.png` |

**Setup**
1. Create a Kaggle dataset containing `judgments_labelled.csv` and this repo's `fl_judicial.py`,
   `run_queue.py` and `summarize_results.py`. The plans `mix`, `mini`, `confirm` and `novel` pick the
   base setting from an earlier sweep, so also include a `runs/` folder with the sweep's
   `all_results.csv` (or run `PLAN = "sweep"` first).
2. In the notebook: **Add Input** → that dataset (and, to merge earlier results, the output of your
   previous notebook run). Settings: **GPU T4 x2**, **Internet On**.
3. **Save Version → Save & Run All (Commit)**.

**Sessions and resuming.** Each queue stops launching new runs after `BUDGET_HOURS`. Finished runs are
skipped and interrupted federated runs resume from their last round (`ckpt.pt`). To continue,
attach this notebook's output as input and run it again; the first cells merge all attached
`runs/` folders into one `all_results.csv`.
