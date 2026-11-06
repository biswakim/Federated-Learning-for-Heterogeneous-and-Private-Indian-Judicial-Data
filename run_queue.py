"""
Runs a queue of fl_judicial.py experiments on ONE GPU inside a time budget.
  * finished runs are skipped (per_court_test.csv exists), so a queue can be restarted any time;
  * an FL run interrupted by the Kaggle 12 h limit resumes from its last round (ckpt.pt);
  * hyper-parameters are chosen on VALIDATION loss only (column sel_val_loss), never on test.

Plans
  sweep   (session 1)  one encoder per GPU:
          stage 1: FedAvg with lr 2e-4 vs 5e-4, then head vs head+tail truncation (on the better lr)
          stage 2: with the chosen lr/truncation: local, central, FedProx, FedAvgM, loss-weighting
                   (beta +1 / -1), FedAvg + local fine-tuning, sequential EWC, FedAdam (higher server lr)
  confirm (session 2)  picks the best encoder + settings and the top-3 FL methods from the sweep
          (by validation loss), reruns them with seeds 1 and 2 together with local / central / FedAvg,
          and trains the overall best method once at 512 tokens.
  novel   (needs a finished sweep's all_results.csv first; ~2.5-3 h on one GPU)
          task-decoupled FL + federated continual onboarding, selected on validation throughout:
          step 1: reuse the sweep's best encoder/lr/truncation (lowest FedAvg validation loss)
          step 2: 3 FedAvg runs, one per sharing choice: all heads local, duration-only local,
                  disposal-only local (plain-shared FedAvg already exists from the sweep); the
                  best is picked on validation loss
          step 3: 4 fed_continual runs with that sharing choice -- naive (no EWC, no replay),
                  EWC only, replay only, EWC+replay -- compared on VALIDATION forgetting (read
                  from forgetting.csv, never from forgetting_summary.csv, which is test-set); the
                  best combination is re-run once more with courts onboarded in a different order
                  (the most atypical court, Madhya Pradesh, first instead of last) to check whether
                  onboarding order itself affects how much is forgotten

  mix     proposed methods (needs the sweep's all_results.csv for the base setting):
          (A) learned per-task shared/local head mixing (--mix_heads dur,typ,disp)
          (B) task-specific reading of start and end of the judgment (--segments 2)
          (C) both together; every FL run also reports its per-court fine-tuned variant (+ft).
          Seeds 0, 1, 2 (seed 0 of all three first, so a cut-off session still yields a
          full single-seed comparison), then local and centralized with --segments 2 to
          separate the architecture gain from the FL gain. Jobs are split over --n_shards
          GPUs by estimated cost. About 6-7 h per GPU on two T4s.

Usage:  python run_queue.py --data DATA --plan sweep --encoder roberta-base --budget_hours 11
        python run_queue.py --data DATA --plan confirm --shard 0 --n_shards 2 --budget_hours 11
        python run_queue.py --data DATA --plan novel --budget_hours 3
"""
import argparse, os, subprocess, sys, time, re
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--plan", choices=["sweep", "confirm", "mini", "novel", "mix"], required=True)
p.add_argument("--seeds", default="1", help="mini plan: comma-separated seeds")
p.add_argument("--encoder", default="roberta-base")
p.add_argument("--shard", type=int, default=0)
p.add_argument("--n_shards", type=int, default=1)
p.add_argument("--budget_hours", type=float, default=11.0)
p.add_argument("--out_dir", default="runs")
p.add_argument("--max_len", type=int, default=256)
p.add_argument("--rounds", type=int, default=30)
p.add_argument("--patience", type=int, default=4)
A = p.parse_args()
T0 = time.time(); os.makedirs("logs", exist_ok=True)
SHOW = re.compile(r"round|best round|early stop|mean:|already done|resumed|loaded best|Error|Traceback|CUDA out")

def base(enc, lr, trunc, max_len=None, seed=0):
    return ["--data", A.data, "--out_dir", A.out_dir, "--model", enc, "--max_len", str(max_len or A.max_len),
            "--lr", str(lr), "--trunc", trunc, "--seed", str(seed)]

def fl(enc, lr, trunc, algo, extra=(), local_heads=None, **kw):
    cmd = base(enc, lr, trunc, **kw) + ["--regime", "fl", "--algo", algo, "--rounds", str(A.rounds),
                                        "--patience", str(A.patience), "--local_epochs", "1", *extra]
    return cmd + ["--local_heads", local_heads] if local_heads else cmd

def ref(enc, lr, trunc, regime, **kw):
    return base(enc, lr, trunc, **kw) + ["--regime", regime, "--epochs", "15", "--patience", "3"]

def continual(enc, lr, trunc, ewc_lambda, replay_size, local_heads=None, order=None, **kw):
    cmd = base(enc, lr, trunc, **kw) + ["--regime", "fed_continual", "--seq_epochs", "3",
                                        "--ewc_lambda", str(ewc_lambda), "--replay_size", str(replay_size)]
    if local_heads: cmd += ["--local_heads", local_heads]
    if order: cmd += ["--onboard_order", order]
    return cmd

def run(cmd, label):
    if (time.time() - T0) / 3600 > A.budget_hours:
        print(f"[{label}] skipped: time budget used up; rerun the notebook in a new session to continue", flush=True)
        return False
    print(f"\n[{label}] start ({(time.time()-T0)/3600:.1f} h elapsed)", flush=True)
    log = open(os.path.join("logs", re.sub(r"\W+", "_", label) + ".log"), "a")
    pr = subprocess.Popen([sys.executable, "-u", "fl_judicial.py", *cmd], stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    for line in pr.stdout:
        log.write(line)
        if SHOW.search(line): print(f"[{label}] {line.rstrip()}", flush=True)
    pr.wait(); log.close()
    return pr.returncode == 0

def results():
    f = os.path.join(A.out_dir, "all_results.csv")
    return pd.read_csv(f) if os.path.exists(f) else pd.DataFrame()

def val_of(enc, regime_algo, **filt):
    """mean selection (validation) loss of the matching seed-0 runs."""
    r = results()
    if r.empty or "sel_val_loss" not in r: return float("inf")
    m = r[(r.model == enc) & (r.method == regime_algo) & (r.seed == filt.get("seed", 0))]
    for k, v in filt.items():
        if k != "seed": m = m[m[k] == v]
    return m.sel_val_loss.mean() if len(m) else float("inf")

def sweep(enc):
    # stage 1: learning rate, then truncation (FedAvg, validation loss)
    for lr in (2e-4, 5e-4): run(fl(enc, lr, "head", "fedavg"), f"{enc}|fedavg lr{lr}")
    lr = min((2e-4, 5e-4), key=lambda x: val_of(enc, "fedavg", lr=x, trunc="head", max_len=A.max_len))
    run(fl(enc, lr, "headtail", "fedavg"), f"{enc}|fedavg lr{lr} headtail")
    tr = min(("head", "headtail"), key=lambda t: val_of(enc, "fedavg", lr=lr, trunc=t, max_len=A.max_len))
    print(f"\n[{enc}] chosen on validation: lr={lr}, trunc={tr}", flush=True)
    # stage 2: references + algorithms with the chosen settings
    run(ref(enc, lr, tr, "local"), f"{enc}|local")
    run(ref(enc, lr, tr, "central"), f"{enc}|central")
    cands = [d for d in os.listdir(A.out_dir) if d.startswith("fl_fedavg_" + enc.split("/")[-1])
             and f"_L{A.max_len}_lr{lr:g}" in d and (("_ht" in d) == (tr == "headtail")) and d.endswith("_s0")
             and os.path.exists(os.path.join(A.out_dir, d, "ckpt.pt"))]
    if cands:   # personalisation reuses the finished FedAvg model instead of retraining it
        run(fl(enc, lr, tr, "fedavg", ["--finetune_epochs", "1", "--init_from",
                                       os.path.join(A.out_dir, cands[0], "ckpt.pt")]), f"{enc}|fedavg+ft")
    run(fl(enc, lr, tr, "fedprox", ["--mu", "0.01"]), f"{enc}|fedprox mu0.01")
    run(fl(enc, lr, tr, "lossweight", ["--beta", "-1"]), f"{enc}|lossweight beta-1")
    run(fl(enc, lr, tr, "lossweight", ["--beta", "1"]), f"{enc}|lossweight beta1")
    run(fl(enc, lr, tr, "fedavgm", ["--server_lr", "1.0", "--server_momentum", "0.5"]), f"{enc}|fedavgm m0.5")
    run(base(enc, lr, tr) + ["--regime", "seq_ewc", "--seq_epochs", "3"], f"{enc}|seq_ewc")
    run(fl(enc, lr, tr, "fedadam", ["--server_lr", "3e-3"]), f"{enc}|fedadam slr3e-3")
    run(fl(enc, lr, tr, "fedprox", ["--mu", "0.1"]), f"{enc}|fedprox mu0.1")

def confirm():
    r = results(); r = r[(r.seed == 0) & (r.max_len == A.max_len)]
    fa = r[r.method == "fedavg"].groupby(["model", "lr", "trunc"]).sel_val_loss.mean()
    enc, lr, tr = fa.idxmin()
    rr = r[(r.model == enc) & (r.lr == lr) & (r.trunc == tr)]
    fl_m = rr[~rr.method.isin(["local_only", "centralized", "fedavg"])].groupby(["run", "method"]).sel_val_loss.mean()
    fl_m = fl_m.groupby(level=0).min().sort_values()
    top = list(fl_m.index[:3])
    print(f"best setting on validation: {enc}, lr={lr}, trunc={tr}; top FL runs: {top}", flush=True)
    jobs = []
    for seed in (1, 2):
        jobs += [(ref(enc, lr, tr, "local", seed=seed), f"local s{seed}"),
                 (ref(enc, lr, tr, "central", seed=seed), f"central s{seed}"),
                 (fl(enc, lr, tr, "fedavg", seed=seed), f"fedavg s{seed}")]
        for run_name in top:
            cfg = __import__("json").load(open(os.path.join(A.out_dir, run_name, "config.json")))
            extra = [f"--{k}={cfg[k]}" for k in ("mu", "beta", "server_lr", "server_momentum", "seq_epochs")
                     if cfg.get(k) is not None]
            if cfg.get("finetune_epochs"): extra += ["--finetune_epochs", str(cfg["finetune_epochs"])]
            if cfg.get("regime") == "seq_ewc": extra += ["--regime", "seq_ewc"]
            jobs.append((fl(enc, lr, tr, cfg["algo"], extra, seed=seed), f"{run_name} s{seed}"))
    if top:   # best method once at 512 tokens (longer context)
        cfg = __import__("json").load(open(os.path.join(A.out_dir, top[0], "config.json")))
        extra = [f"--{k}={cfg[k]}" for k in ("mu", "beta", "server_lr", "server_momentum", "seq_epochs")
                 if cfg.get(k) is not None]
        if cfg.get("finetune_epochs"): extra += ["--finetune_epochs", str(cfg["finetune_epochs"])]
        if cfg.get("regime") == "seq_ewc": extra += ["--regime", "seq_ewc"]
        jobs.append((fl(enc, lr, tr, cfg["algo"], extra, max_len=512), f"{top[0]} @512"))
    for i, (cmd, label) in enumerate(jobs):
        if i % A.n_shards == A.shard: run(cmd, label)

def fedavg_ckpt(enc, lr, tr, seed):
    d = [x for x in os.listdir(A.out_dir) if x.startswith("fl_fedavg_" + enc.split("/")[-1])
         and f"_L{A.max_len}_lr{lr:g}" in x and (("_ht" in x) == (tr == "headtail")) and x.endswith(f"_s{seed}")
         and os.path.exists(os.path.join(A.out_dir, x, "ckpt.pt"))]
    return os.path.join(A.out_dir, d[0], "ckpt.pt") if d else None

def mini():
    """Short budget: extra seeds for the core comparison (FedAvg, FedAvg+ft, local, central)
    at the best encoder/setting found by the sweep (chosen on validation loss)."""
    r = results(); r = r[(r.seed == 0) & (r.max_len == A.max_len)]
    enc, lr, tr = r[r.method == "fedavg"].groupby(["model", "lr", "trunc"]).sel_val_loss.mean().idxmin()
    print(f"setting chosen on validation: {enc}, lr={lr}, trunc={tr}", flush=True)
    for seed in [int(x) for x in A.seeds.split(",")]:
        run(fl(enc, lr, tr, "fedavg", seed=seed), f"fedavg s{seed}")
        ck = fedavg_ckpt(enc, lr, tr, seed)
        if ck: run(fl(enc, lr, tr, "fedavg", ["--finetune_epochs", "1", "--init_from", ck], seed=seed), f"fedavg+ft s{seed}")
        run(ref(enc, lr, tr, "local", seed=seed), f"local s{seed}")
        run(ref(enc, lr, tr, "central", seed=seed), f"central s{seed}")

def local_heads_of(run_name):
    """The --local_heads a run actually used, read from its own config.json -- not guessed
    from the folder name, which also contains the (possibly dash-separated) model name and
    is not safely reversible."""
    cfg = __import__("json").load(open(os.path.join(A.out_dir, run_name, "config.json")))
    lh = cfg.get("local_heads", "none")
    return lh if lh and lh.lower() != "none" else "none"

def val_forgetting(run_name):
    """VALIDATION-set forgetting for one fed_continual run, read straight from its
    forgetting.csv (val_MAE_log / val_disp_F1 columns) -- never from forgetting_summary.csv,
    which holds the test-set numbers meant for the report, not for picking a configuration."""
    f = os.path.join(A.out_dir, run_name, "forgetting.csv")
    if not os.path.exists(f): return float("inf")
    h = pd.read_csv(f)
    if h.empty: return float("inf")
    last = h.stage.max(); onboarded_at = h.groupby("eval_court").stage.min()
    vals = []
    for c, s0 in onboarded_at.items():
        r0 = h[(h.eval_court == c) & (h.stage == s0)]; rN = h[(h.eval_court == c) & (h.stage == last)]
        if len(r0) and len(rN):
            vals.append(abs(rN.val_MAE_log.iloc[0] - r0.val_MAE_log.iloc[0]) +
                       abs(rN.val_disp_F1.iloc[0] - r0.val_disp_F1.iloc[0]))
    return sum(vals) / len(vals) if vals else float("inf")

def novel():
    r = results(); r0 = r[(r.seed == 0) & (r.max_len == A.max_len)]
    enc, lr, tr = r0[r0.method == "fedavg"].groupby(["model", "lr", "trunc"]).sel_val_loss.mean().idxmin()
    print(f"base setting (from the sweep, on validation): {enc}, lr={lr}, trunc={tr}", flush=True)

    # --- step 2: which head(s), if any, should be kept local to each court? ---
    for name, lh in [("all-local", "dur,typ,disp"), ("dur-local", "dur"), ("disp-local", "disp")]:
        run(fl(enc, lr, tr, "fedavg", local_heads=lh), f"decouple {name}")
    r = results(); r = r[(r.seed == 0) & (r.model == enc) & (r.lr == lr) & (r.trunc == tr) & (r.method == "fedavg")]
    r = r.assign(local_heads=r.run.map(local_heads_of))
    cand = r.groupby("local_heads").sel_val_loss.mean()
    best_lh = cand.idxmin()
    print(f"sharing choice on validation ({dict(cand.round(3))}): local_heads={best_lh}", flush=True)
    best_lh_arg = None if best_lh == "none" else best_lh

    # --- step 3: federated continual onboarding -- naive / EWC / replay / both ---
    combos = [("naive", 0.0, 0), ("ewc_only", 100.0, 0), ("replay_only", 0.0, 16), ("ewc_replay", 100.0, 16)]
    for name, lam, rep in combos:
        run(continual(enc, lr, tr, lam, rep, local_heads=best_lh_arg), f"continual {name}")
    r = results(); r = r[(r.model == enc) & (r.lr == lr) & (r.trunc == tr) & (r.method == "fed_continual")]
    cand = {run_name: val_forgetting(run_name) for run_name in r.run.unique()}
    if cand:
        best_combo_run = min(cand, key=cand.get)
        print(f"least VALIDATION forgetting ({ {k: round(v, 3) for k, v in cand.items()} }): {best_combo_run}", flush=True)
        cfg = __import__("json").load(open(os.path.join(A.out_dir, best_combo_run, "config.json")))
        run(continual(enc, lr, tr, cfg["ewc_lambda"], cfg["replay_size"], local_heads=best_lh_arg,
                      order="madhya_pradesh,bombay,kerala,delhi,karnataka"), "continual best-combo minority-first")

def mix_plan():
    r = results(); r0 = r[(r.seed == 0) & (r.max_len == A.max_len)]
    enc, lr, tr = r0[r0.method == "fedavg"].groupby(["model", "lr", "trunc"]).sel_val_loss.mean().idxmin()
    print(f"base setting (from the sweep, on validation): {enc}, lr={lr}, trunc={tr}", flush=True)
    MIX = ["--mix_heads", "dur,typ,disp"]; SEG = ["--segments", "2"]; FT = ["--finetune_epochs", "1"]
    jobs = []   # (label, cmd, estimated hours), in priority order
    for seed in (0, 1, 2):
        jobs += [(f"mix s{seed}", fl(enc, lr, tr, "fedavg", MIX + FT, seed=seed), 0.8),
                 (f"seg2 s{seed}", fl(enc, lr, "head", "fedavg", SEG + FT, seed=seed), 1.5),
                 (f"seg2+mix s{seed}", fl(enc, lr, "head", "fedavg", SEG + MIX + FT, seed=seed), 1.6)]
    jobs += [("local seg2 s0", ref(enc, lr, "head", "local") + SEG, 0.6),
             ("central seg2 s0", ref(enc, lr, "head", "central") + SEG, 0.7)]
    load = [0.0] * A.n_shards; mine = []
    for label, cmd, h in jobs:                     # greedy, deterministic: same split on every GPU
        k = min(range(A.n_shards), key=lambda i: (load[i], i)); load[k] += h
        if k == A.shard: mine.append((label, cmd))
    print(f"shard {A.shard}/{A.n_shards}: {[l for l, _ in mine]} (~{load[A.shard]:.1f} h)", flush=True)
    for label, cmd in mine: run(cmd, label)

if A.plan == "sweep": sweep(A.encoder)
elif A.plan == "mix": mix_plan()
elif A.plan == "mini": mini()
elif A.plan == "novel": novel()
else: confirm()
print(f"\nqueue finished after {(time.time()-T0)/3600:.2f} h", flush=True)
