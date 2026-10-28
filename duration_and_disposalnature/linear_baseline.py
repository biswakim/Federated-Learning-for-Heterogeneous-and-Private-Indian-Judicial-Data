"""
Task 1 (case duration) and Task 4 (disposal nature): federated text baselines.

Data   : judgments_labelled.csv (12,500 judgments from 5 High Courts; column `split`).
Leakage: every case was decided in 2024, so dates/years in the text nearly reveal the
         duration -> dates, years and multi-digit numbers are masked. For disposal nature,
         outcome words (allowed, dismissed, disposed of, withdrawn, ...) are masked too.
Feats  : HashingVectorizer word 1-2 grams (no vocabulary fitted on pooled data, so the
         same featurizer is valid when data stay at each court).
Model  : linear model trained with mini-batch Adam (numpy/scipy only), identical for every
         regime. Duration: Huber loss on log(1+days). Disposal: class-weighted softmax over
         ALLOWED / DISMISSED / DISPOSED / WITHDRAWN (UNCLEAR dropped).
Regimes: baseline (court median / court majority), local-only, centralized, FedAvg
         (client weights proportional to training size). Every regime keeps the epoch/round
         with the lowest validation loss, so differences are not just training budget.
Usage  : SEEDS=0,1,2 python task1_task4_baselines.py judgments_labelled.csv
         python task1_task4_baselines.py --summarize per_court*.csv
Outputs: task1_task4_per_court.csv, task1_task4_summary.csv
"""
import sys, os, re, warnings
import numpy as np, pandas as pd
from scipy.sparse import vstack
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import f1_score
warnings.filterwarnings("ignore")

PATH = sys.argv[1] if len(sys.argv) > 1 else "judgments_labelled.csv"
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
TASKS = os.environ.get("TASKS", "dur,disp").split(",")
EPOCHS, LOCAL_EPOCHS, BATCH, LR, L2 = 30, 2, 128, 5e-3, 1e-6
ROUNDS = int(os.environ.get("ROUNDS", 300))
NFEAT = 2**16
CLASSES = ["ALLOWED", "DISMISSED", "DISPOSED", "WITHDRAWN"]

MON = r"(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|aug(ust)?|sep(t(ember)?)?|oct(ober)?|nov(ember)?|dec(ember)?)"
def mask_dates(t):
    t = re.sub(r"\b\d{1,2}(st|nd|rd|th)?[\s./-]*" + MON + r"[\s.,/-]*(\d{2,4})?", " DATE ", t, flags=re.I)
    t = re.sub(MON + r"\s+\d{1,2}(st|nd|rd|th)?,?\s*\d{2,4}", " DATE ", t, flags=re.I)
    t = re.sub(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", " DATE ", t)
    t = re.sub(r"\b(19|20)\d{2}\b", " YEAR ", t)
    return re.sub(r"\b\d{2,}\b", " NUM ", t)
OUT = r"\b(allow(ed|s)?|dismiss(ed|al)?|dispos(ed|al)(\s+of)?|withdra(wn|w|wal)|rejected|not pressed|infructuous)\b"
def mask_outcome(t): return re.sub(OUT, " OUTCOME ", t, flags=re.I)

# ---------------- linear model + Adam ----------------
class Lin:
    def __init__(s, out, seed):
        r = np.random.default_rng(seed)
        s.W = r.normal(0, 1e-3, (NFEAT, out)); s.b = np.zeros(out)
    def state(s): return [s.W.copy(), s.b.copy()]
    def load(s, st): s.W, s.b = st[0].copy(), st[1].copy()
    def out(s, X): return np.asarray(X @ s.W) + s.b

def class_w(y):
    cnt = np.bincount(y, minlength=4).astype(float); return cnt.sum() / (4 * np.maximum(cnt, 1))

def loss_grad(m, X, y, task, cw):
    z = m.out(X); n = X.shape[0]
    if task == "dur":
        r = z[:, 0] - y; a = np.abs(r)
        loss = np.where(a <= 1, 0.5 * r**2, a - 0.5).mean()
        G = np.clip(r, -1, 1)[:, None] / n
    else:
        z = z - z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
        sw = cw[y]; loss = -(sw * np.log(p[np.arange(n), y] + 1e-12)).sum() / sw.sum()
        G = p; G[np.arange(n), y] -= 1; G = G * sw[:, None] / sw.sum()
    return loss, np.asarray(X.T @ G) + L2 * m.W, G.sum(0)

def train(m, X, y, task, epochs, rng, cw):
    mW, vW, mb, vb, t = 0, 0, 0, 0, 0; b1, b2 = 0.9, 0.999
    for _ in range(epochs):
        idx = rng.permutation(X.shape[0])
        for s in range(0, len(idx), BATCH):
            i = idx[s:s + BATCH]; _, gW, gb = loss_grad(m, X[i], y[i], task, cw); t += 1
            mW = b1 * mW + (1 - b1) * gW; vW = b2 * vW + (1 - b2) * gW**2
            mb = b1 * mb + (1 - b1) * gb; vb = b2 * vb + (1 - b2) * gb**2
            m.W -= LR * (mW / (1 - b1**t)) / (np.sqrt(vW / (1 - b2**t)) + 1e-8)
            m.b -= LR * (mb / (1 - b1**t)) / (np.sqrt(vb / (1 - b2**t)) + 1e-8)

def val_loss(m, clients, task, cw):
    tot = sum(loss_grad(m, c["Xva"], c["yva"], task, cw)[0] * len(c["yva"]) for c in clients)
    return tot / sum(len(c["yva"]) for c in clients)

def fit_early(X, y, task, out, val_clients, rng, seed):
    m = Lin(out, seed); cw = class_w(y) if task == "disp" else None; best, st = np.inf, None
    for _ in range(EPOCHS):
        train(m, X, y, task, 1, rng, cw)
        v = val_loss(m, val_clients, task, cw)
        if v < best: best, st = v, m.state()
    m.load(st); return m

def fedavg(C, task, out, rng, seed):
    g = Lin(out, seed); n = np.array([len(c["y"]) for c in C], float); w = n / n.sum()
    gcw = class_w(np.concatenate([c["y"] for c in C])) if task == "disp" else None
    best, st = np.inf, None
    best_r = 0
    for r in range(ROUNDS):
        states = []
        for c in C:
            m = Lin(out, seed); m.load(g.state())
            train(m, c["X"], c["y"], task, LOCAL_EPOCHS, rng, class_w(c["y"]) if task == "disp" else None)
            states.append(m.state())
        g.load([sum(wi * s[k] for wi, s in zip(w, states)) for k in range(2)])
        v = val_loss(g, C, task, gcw)
        if v < best: best, st, best_r = v, g.state(), r + 1
    print(f"  FedAvg {task}: best validation round {best_r}/{ROUNDS}", flush=True)
    g.load(st); return g

def bucket(d): return np.digitize(d, [30, 180, 730], right=True)
def dur_metrics(y, p):
    return {"MAE_log": np.abs(y - p).mean(),
            "Spearman": spearmanr(y, p)[0] if np.std(p) > 0 else np.nan,
            "F1_4": f1_score(bucket(np.expm1(y)), bucket(np.expm1(p)), average="macro", labels=[0, 1, 2, 3])}

def main():
    df = pd.read_csv(PATH, low_memory=False)
    df = df[~df.garbled.astype(bool)].copy()
    df["t_dur"] = df.input_text.fillna("").map(mask_dates)
    df["t_disp"] = df.t_dur.map(mask_outcome)
    hv = HashingVectorizer(n_features=NFEAT, ngram_range=(1, 2), alternate_sign=False, norm="l2")
    courts = sorted(df.court.unique()); rows = []
    for task, col, out in [t for t in [("dur", "t_dur", 1), ("disp", "t_disp", 4)] if t[0] in TASKS]:
        d = df if task == "dur" else df[df.outcome_from_text.isin(CLASSES)]
        lab = (lambda x: np.log1p(x.duration_days.values)) if task == "dur" else \
              (lambda x: x.outcome_from_text.map(CLASSES.index).values)
        C = []
        for c in courts:
            s = {p: d[(d.court == c) & (d.split == p)] for p in ["train", "val", "test"]}
            C.append({"court": c, "X": hv.transform(s["train"][col]).tocsr(), "y": lab(s["train"]),
                      "Xva": hv.transform(s["val"][col]), "yva": lab(s["val"]),
                      "Xte": hv.transform(s["test"][col]), "yte": lab(s["test"])})
        Xall, yall = vstack([c["X"] for c in C]).tocsr(), np.concatenate([c["y"] for c in C])
        for seed in SEEDS:
            rng = np.random.default_rng(seed); P = {"baseline": {}, "local_only": {}}
            for c in C:
                P["baseline"][c["court"]] = (np.full(len(c["yte"]), np.median(c["y"])) if task == "dur"
                                             else np.full(len(c["yte"]), np.bincount(c["y"]).argmax()))
                P["local_only"][c["court"]] = fit_early(c["X"], c["y"], task, out, [c], rng, seed).out(c["Xte"])
            m = fit_early(Xall, yall, task, out, C, rng, seed)
            P["centralized"] = {c["court"]: m.out(c["Xte"]) for c in C}
            g = fedavg(C, task, out, rng, seed)
            P["fedavg"] = {c["court"]: g.out(c["Xte"]) for c in C}
            for meth, pr in P.items():
                for c in C:
                    p = pr[c["court"]]
                    if task == "dur": met = dur_metrics(c["yte"], np.ravel(p) if p.ndim == 1 else p[:, 0])
                    else: met = {"MacroF1": f1_score(c["yte"], p if p.ndim == 1 else p.argmax(1),
                                                     average="macro", labels=[0, 1, 2, 3])}
                    rows.append({"task": "duration" if task == "dur" else "disposal", "seed": seed,
                                 "method": meth, "court": c["court"], **met})
            print(task, "seed", seed, "done", flush=True)
    tag = os.environ.get("TAG", "")
    pd.DataFrame(rows).to_csv(f"per_court{tag}.csv", index=False)

def summarize(files):
    out = pd.concat([pd.read_csv(f) for f in files]); out.to_csv("task1_task4_per_court.csv", index=False)
    print("\nPer court (mean over seeds):")
    print(out.groupby(["task", "method", "court"]).mean(numeric_only=True).drop(columns="seed")
          .dropna(axis=1, how="all").round(3).to_string())
    g = out.groupby(["task", "method", "seed"])
    agg = pd.concat([g[["MAE_log", "Spearman", "F1_4", "MacroF1"]].mean(),
                     g["MAE_log"].max().rename("Worst_MAE"), g["MacroF1"].min().rename("Worst_F1")], axis=1)
    summ = agg.groupby(["task", "method"]).agg(["mean", "std"]).round(3).dropna(axis=1, how="all")
    summ.to_csv("task1_task4_summary.csv"); print("\nSummary (mean, std over seeds):\n", summ.to_string())

if __name__ == "__main__":
    # python task1_task4_baselines.py judgments_labelled.csv          -> run (env: SEEDS, TASKS, ROUNDS, TAG)
    # python task1_task4_baselines.py --summarize per_court_*.csv     -> merge runs and print tables
    if sys.argv[1] == "--summarize": summarize(sys.argv[2:])
    else: main()
