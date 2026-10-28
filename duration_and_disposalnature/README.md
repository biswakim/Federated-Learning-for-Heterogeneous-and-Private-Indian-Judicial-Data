# Federated duration + disposal-nature prediction (Tasks 1 and 4)

Implements the planned model from the proposal (Figure 2) for Task 1 (case duration) and
Task 4 (disposal nature), with case type as the cascade input, and a menu of FL algorithms to
compare. Clients are the five High Courts.

```
case description ─► RoBERTa / InLegalBERT (+ LoRA) ─► mean-pooled embedding ─┐
                └─► OpenNyAI NER type histogram ─► MLP (optional) ────────────┤ concat
      ├─► duration head (Huber on log(1+days))
      ├─► case-type head (auxiliary)
      └─► disposal-nature head (conditioned on case-type probabilities)
```

Only LoRA adapters, the NER MLP and the heads are trained and exchanged with the server
(about 1% of RoBERTa-base), so communication stays small.

## Files
| File | Purpose |
|---|---|
| `fl_judicial.py` | main script: data prep, model, local / central / FL / continual regimes |
| `ner_features.py` | optional: OpenNyAI NER entity-type counts per case (NER branch) |
| `summarize_results.py` | compares all runs (mean ± std over seeds, worst court) |
| `linear_baseline.py` | fast CPU sanity baseline (hashed n-grams + linear model) |

## Data handling (already in the code)
* Input: `input_text` from `judgments_labelled.csv` (operative ending already removed), garbled PDFs dropped, existing `split` column used (temporal 70/15/15 per court).
* **Dates, years and multi-digit numbers are masked.** All cases were decided in 2024, so the filing year / dates in the text nearly give away the duration. In a quick check, masking them moved a text baseline from MAE_log 1.17 to 1.43; without masking, results would be inflated.
* Outcome words (allowed, dismissed, disposed of, withdrawn, …) are masked so disposal labels cannot be copied.
* Disposal labels come from `outcome_from_text` (ALLOWED / DISMISSED / DISPOSED / WITHDRAWN; UNCLEAR rows still train the other heads). `disposal_nature_metadata` is unusable for this: it is empty for every Delhi case and uses 13–38 spellings per court.
* Class weights for disposal are computed from each court's own data.


## v2: automated experiment queue for Kaggle (12 h sessions)
Run **`fl_experiments.ipynb`** (GPU T4 x2, Internet on, "Save & Run All (Commit)"):
* `PLAN = "sweep"` (session 1): one encoder per GPU (RoBERTa on GPU 0, InLegalBERT on GPU 1).
  Stage 1 picks the learning rate (2e-4 vs 5e-4) and truncation (head vs head+tail) on validation
  loss; stage 2 runs local, central, FedAvg+fine-tuning, FedProx (mu 0.01, 0.1), loss-weighting
  (beta ±1), FedAvgM, sequential EWC and FedAdam with that setting. 256 tokens, up to 30 rounds,
  early stopping after 4 rounds without validation improvement.
* `PLAN = "confirm"` (session 2, previous output attached as input): best encoder/setting and the
  top-3 FL methods (by validation loss) are rerun with seeds 1 and 2 next to local / central /
  FedAvg, plus the best method once at 512 tokens.
* Safe to interrupt: finished runs are skipped, FL runs resume from their last round (`ckpt.pt`).

New flags in `fl_judicial.py`: `--trunc headtail`, `--patience`, `--lora_targets`, `--init_from`
(fine-tune a finished FL model), `--force`; results now include `sel_val_loss` (validation loss
used for selection) and `best_step`.

## Kaggle setup (manual runs)
1. Upload `judgments_labelled.csv` (and these .py files) as a Kaggle dataset.
2. Notebook settings: **GPU T4** (or P100), **Internet ON** (to download the encoder from Hugging Face).
3. In the notebook:
```bash
!pip install -q peft            # if it complains about transformers: !pip install -q -U transformers peft
!cp /kaggle/input/<your-dataset>/*.py .
DATA=/kaggle/input/<your-dataset>/judgments_labelled.csv
```
4. Smoke test (a couple of minutes):
```bash
!python fl_judicial.py --data $DATA --regime fl --algo fedavg --rounds 2 --limit 64 --max_len 128
```
5. Check the seconds per round printed by the first real run before planning a sweep. If you run out of GPU memory use `--batch 8`. Kaggle T4×2 sessions: you can run two jobs at once with `CUDA_VISIBLE_DEVICES=0 python ... &` and `CUDA_VISIBLE_DEVICES=1 python ...`.

## Experiment plan (algorithm comparison)
Keep the training budget identical across algorithms (same `--rounds`, `--local_epochs`, `--max_len`); every regime picks its checkpoint on validation, never on test.

**Step 1: references and encoder choice** (seed 0, `--max_len 256` to keep the sweep cheap)
```bash
python fl_judicial.py --data $DATA --regime local   --epochs 6
python fl_judicial.py --data $DATA --regime central --epochs 6
python fl_judicial.py --data $DATA --regime fl --algo fedavg --rounds 15
python fl_judicial.py --data $DATA --regime fl --algo fedavg --rounds 15 --model law-ai/InLegalBERT
```
Keep the better encoder for everything below. If the printout says `best round = last round … still improving`, increase `--rounds`: FedAvg is often under-trained at a fixed small budget (our linear baseline needed ~200 rounds to converge; at 30 rounds it looked much worse than it was).

**Step 2: algorithms** (same budget)
```bash
python fl_judicial.py --data $DATA --regime fl --algo fedprox    --mu 0.01
python fl_judicial.py --data $DATA --regime fl --algo fedprox    --mu 0.1
python fl_judicial.py --data $DATA --regime fl --algo fedavgm    --server_lr 1.0
python fl_judicial.py --data $DATA --regime fl --algo fedadam    --server_lr 1e-3
python fl_judicial.py --data $DATA --regime fl --algo fedyogi    --server_lr 1e-3
python fl_judicial.py --data $DATA --regime fl --algo lossweight --beta 1     # favour well-fitting courts
python fl_judicial.py --data $DATA --regime fl --algo lossweight --beta -1    # favour struggling courts
python fl_judicial.py --data $DATA --regime fl --algo fedavg --finetune_epochs 1   # personalisation
python fl_judicial.py --data $DATA --regime seq_ewc --seq_epochs 2                # continual learning
```
FedAdam/FedYogi are sensitive to `--server_lr`; try 1e-3 and 3e-3 before discarding them.

**Step 3: confirm** the best 2–3 settings plus local, central and FedAvg with `--seed 1` and `--seed 2`, then the NER ablation (`--ner_csv ner_features.csv`) on the best setting, and finally the full `--max_len 512`.

**Compare:**
```bash
python summarize_results.py runs/all_results.csv
```
Report for each method: mean MAE_log, Spearman, 4-class F1 (duration), disposal macro-F1, and the **worst court** for both, as mean ± std over seeds. "Best" should mean best mean *without* a worse worst court, given RQ1.

## Outputs
`runs/<run_name>/per_court_test.csv`, `history.csv` (per round: validation loss, aggregation weights, mean metrics), `config.json`; every run also appends to `runs/all_results.csv`.

## Reference numbers from the linear baseline (CPU, 1 seed, same masking and splits)
| Method | Duration MAE_log | Spearman | Disposal macro-F1 |
|---|---|---|---|
| Court median / majority | 1.990 | – | 0.140 |
| Local-only | 1.542 | 0.567 | 0.553 |
| Centralized | 1.429 | 0.616 | 0.576 |
| FedAvg (300 rounds) | 1.517 | 0.566 | 0.549 |
The transformer model should beat these; if it doesn't, check the masking and the number of rounds first.
