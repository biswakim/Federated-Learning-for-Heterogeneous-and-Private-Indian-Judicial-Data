# Federated Learning on Indian High Court Judgments

Predicting **case duration** and **disposal nature** from judgment text across five Indian High
Courts with federated learning, so that no court has to share its judgments.

Course project, IISER Bhopal (2026). Biswadeep Chakraborty, Bijinepalli Surat Dhani,
Poorvai Chandrasen, Palak Kumari.

## Main result

Reading the start and the end of each judgment separately, with each task learning which part to
use (**TSR**), is the largest improvement. Combined with per-court personalisation it beats
standard federated learning, local training and centralized training on duration.

| Method (InLegalBERT, 3 seeds) | Duration MAE_log ↓ | Spearman ↑ | Disposal F1 ↑ | Case-type F1 ↑ | Worst-court MAE_log ↓ |
| --- | --- | --- | --- | --- | --- |
| Local only | 1.462 ± 0.025 | 0.579 | 0.551 | 0.894 | 1.825 |
| Centralized (pooled data) | 1.462 ± 0.021 | 0.589 | **0.561** | 0.853 | 1.852 |
| FedAvg | 1.474 ± 0.023 | 0.595 | 0.542 | 0.774 | 1.888 |
| FedAvg + fine-tuning | 1.462 ± 0.035 | 0.595 | 0.554 | 0.900 | 1.789 |
| **TSR** | **1.347** ± 0.013 | 0.632 | 0.539 | 0.844 | **1.704** |
| **TSR + PTM + fine-tuning** | **1.347** ± 0.019 | **0.635** | 0.557 | **0.916** | 1.709 |

MAE_log is the mean absolute error of log(1 + days). Full results, including FedProx, FedAvgM,
FedAdam, loss-based weighting, task-decoupled heads and continual onboarding, are in
[`results/main_results.csv`](results/main_results.csv) and [`report/methods_results.tex`](report/methods_results.tex).

**What the model learned.** Case type reads mainly the start of a judgment, disposal nature mainly
the end, and duration both ([figure](results/figures/fig_segment_weights.png)). The learned mixing
weights show that courts rely most on the shared head for duration and least for case type, whose
codes are court-specific ([figure](results/figures/fig_mixing_weights.png)).

## Method

* **Clients.** Bombay, Delhi, Karnataka, Kerala and Madhya Pradesh High Courts, one client each.
* **Model.** InLegalBERT (or RoBERTa) with LoRA adapters; heads for duration (Huber loss on
  log days), case type, and disposal nature conditioned on the predicted case type. Only adapters
  and heads (about 1% of the model) are sent to the server.
* **Leakage control.** Dates, years, multi-digit numbers and outcome words are masked.
* **TSR** (`--segments 2`): the first and last 256 tokens are encoded separately; each task has a
  learned query that weights the two.
* **PTM** (`--mix_heads dur,typ,disp`): each court keeps the shared head and its own head for each
  task and learns a blend weight per task; local heads and weights never leave the court.
* **Baselines.** Local, centralized, FedAvg, FedProx, FedAvgM, FedAdam, loss-based client weighting,
  FedAvg + per-court fine-tuning, fixed local heads (`--local_heads`), sequential EWC, and federated
  continual onboarding with EWC and replay (`--regime fed_continual`).

## Repository layout

```
fl_judicial.py          data loading, masking, model, all training regimes (main script)
run_queue.py            experiment plans (sweep, mini, confirm, novel, mix) with time budget and resume
summarize_results.py    mean ± std over seeds from runs/all_results.csv
notebooks/              Kaggle notebooks used for the experiments (see notebooks/README.md)
scripts/
  check_masking.py      re-applies the masking and reports what survives
  plot_forgetting.py    forgetting curves for continual-onboarding runs
  ner_features.py       optional OpenNyAI NER features (--ner_csv)
  linear_baseline.py    fast CPU baseline (hashed n-grams, linear model)
tests/smoke_test.py     CPU end-to-end test with a tiny random encoder
results/                main_results.csv and figures
report/                 LaTeX Methodology + Results sections for the report
data/                   put judgments_labelled.csv here (see data/README.md)
```

## Setup

```bash
git clone <this repo> && cd judicial-fl
pip install -r requirements.txt
# put the data at data/judgments_labelled.csv (see data/README.md)
python tests/smoke_test.py --data data/judgments_labelled.csv --regime fl --algo fedavg
```

The smoke test runs on CPU in about a minute and does not download a model.

## Running experiments

A GPU is needed for the real encoders (we used Kaggle T4s). Every run writes
`runs/<run_name>/per_court_test.csv`, `config.json`, `history.csv` and appends to
`runs/all_results.csv`; finished runs are skipped and interrupted federated runs resume.

```bash
DATA=data/judgments_labelled.csv
COMMON="--data $DATA --model law-ai/InLegalBERT --lr 5e-4 --max_len 256 --rounds 30 --patience 4"

# baselines
python fl_judicial.py $COMMON --regime local --epochs 15 --patience 3 --trunc headtail
python fl_judicial.py $COMMON --regime central --epochs 15 --patience 3 --trunc headtail
python fl_judicial.py $COMMON --regime fl --algo fedavg --trunc headtail --finetune_epochs 1

# proposed: TSR + PTM (+ fine-tuning, reported as a second method in the same run)
python fl_judicial.py $COMMON --regime fl --algo fedavg --segments 2 --mix_heads dur,typ,disp --finetune_epochs 1

# federated continual onboarding with EWC + replay
python fl_judicial.py $COMMON --regime fed_continual --trunc headtail --seq_epochs 3 \
    --ewc_lambda 100 --replay_size 16 --local_heads dur,typ,disp

python summarize_results.py runs/all_results.csv     # writes comparison.csv
```

Other algorithms: `--algo fedprox --mu 0.01`, `--algo fedavgm --server_lr 1.0`,
`--algo fedadam --server_lr 3e-3`, `--algo lossweight --beta -1`. Full queues used in the
project: `python run_queue.py --data $DATA --plan {sweep,mini,confirm,novel,mix}`; see the
docstring of `run_queue.py`.

## Reproducibility notes

* All settings were chosen on validation loss; test results are only reported.
* GPU training was not bit-for-bit reproducible: re-running a configuration with the same seed
  changed MAE_log by up to about 0.02. Treat single-seed differences below that as noise.
* Cost: a single-view 256-token federated round took about 95 s on one T4 (all five courts); two-segment rounds take roughly twice as long. Runs stop early after 4 rounds without validation improvement (at most 30 rounds).

## Limitations

* Local and centralized TSR runs, task-decoupled heads and continual runs are single seeds.
* TSR reads 512 tokens in total against 256 for single-view baselines; a 512-token single-view
  control was not run.
* PTM weights are learned on training data, which may bias them towards the local heads.
* Disposal and case-type labels are rule-based; the experiments use a 12,500-judgment sample of
  cases decided in 2024.
* No formal privacy mechanism (secure aggregation, differential privacy) is applied, and the replay
  buffer shares a few real cases between courts.

## Acknowledgements

Data: Indian High Court Judgments (Dattam Labs, AWS Open Data, CC-BY-4.0). Encoders:
InLegalBERT (`law-ai/InLegalBERT`) and RoBERTa (`roberta-base`) from Hugging Face; LoRA via `peft`.

## License

Code: MIT (see `LICENSE`). The dataset keeps its own CC-BY-4.0 license.
