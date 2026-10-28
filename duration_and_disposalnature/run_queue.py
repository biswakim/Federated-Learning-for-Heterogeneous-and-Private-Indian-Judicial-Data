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

Usage:  python run_queue.py --data DATA --plan sweep --encoder roberta-base --budget_hours 11
        python run_queue.py --data DATA --plan confirm --shard 0 --n_shards 2 --budget_hours 11
"""
import argparse, os, subprocess, sys, time, re
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--plan", choices=["sweep", "confirm"], required=True)
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

def fl(enc, lr, trunc, algo, extra=(), **kw):
    return base(enc, lr, trunc, **kw) + ["--regime", "fl", "--algo", algo, "--rounds", str(A.rounds),
                                         "--patience", str(A.patience), "--local_epochs", "1", *extra]

def ref(enc, lr, trunc, regime, **kw):
    return base(enc, lr, trunc, **kw) + ["--regime", regime, "--epochs", "15", "--patience", "3"]

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

if A.plan == "sweep": sweep(A.encoder)
else: confirm()
print(f"\nqueue finished after {(time.time()-T0)/3600:.2f} h", flush=True)
