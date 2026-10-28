"""
Federated multi-task model for the Indian High Court project
Task 1 (case duration) + Task 4 (disposal nature), with case type as the cascade input.

Architecture (proposal, Figure 2)
    case description ──► RoBERTa / InLegalBERT encoder (LoRA adapters) ──► mean-pooled embedding ─┐
                    └──► OpenNyAI NER entity-type histogram ──► small MLP (optional) ─────────────────┤
                                                                                                    concat
        ├─► regression head      : (1) duration, Huber loss on log(1+days)
        ├─► classification head  : (2) case type (auxiliary, cascade input)
        └─► classification head  : (4) disposal nature, conditioned on case-type probabilities

Clients = High Courts. Only trainable parameters (LoRA adapters, NER MLP, heads) are
exchanged with the server; the frozen encoder weights never move.

Regimes
    local      each court trains alone (lower bound)
    central    all training data pooled (reference upper bound, not private)
    fl         federated training with --algo:
                 fedavg     size-weighted averaging (McMahan et al. 2017)
                 fedprox    + proximal term mu/2 ||w - w_global||^2 (Li et al. 2020)
                 fedavgm    server momentum (Hsu et al. 2019)
                 fedadam    adaptive server optimizer (Reddi et al. 2021)
                 fedyogi    adaptive server optimizer (Reddi et al. 2021)
                 lossweight weights n_k * exp(-beta * L_k); beta>0 favours well-fitting
                            clients (FedLBW-style), beta<0 struggling ones (q-FFL-style)
               --finetune_epochs k adds per-court fine-tuning of the final global model
    seq_ewc    continual learning: courts visited one after another, EWC penalty
               (Kirkpatrick et al. 2017) protects what earlier courts taught the model

Leakage control
    Every case in the data was decided in 2024, so dates/years/case numbers in the text
    nearly reveal the duration: all dates, years and multi-digit numbers are masked.
    Outcome words (allowed, dismissed, disposed of, withdrawn, ...) are masked so the
    disposal label cannot be copied from the text.

Model selection
    fl: the round with the lowest size-weighted validation loss; local/central: the epoch
    with the lowest validation loss. Test metrics are reported only for that checkpoint.

Example (Kaggle, GPU):
    python fl_judicial.py --data /kaggle/input/<dataset>/judgments_labelled.csv \
        --regime fl --algo fedavg --rounds 10 --local_epochs 1 --seed 0
"""
import argparse, os, re, json, time, random, pickle, hashlib
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import AutoTokenizer, AutoModel
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

DISP = ["ALLOWED", "DISMISSED", "DISPOSED", "WITHDRAWN"]
NER_TYPES = ["COURT", "PETITIONER", "RESPONDENT", "JUDGE", "LAWYER", "ORG", "GPE",
             "STATUTE", "PROVISION", "PRECEDENT", "WITNESS", "OTHER_PERSON"]  # DATE, CASE_NUMBER excluded

# --------------------------------------------------------------------------- args
def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="judgments_labelled.csv")
    p.add_argument("--ner_csv", default=None, help="output of ner_features.py (optional NER branch)")
    p.add_argument("--out_dir", default="runs")
    p.add_argument("--run_name", default=None)
    p.add_argument("--regime", choices=["local", "central", "fl", "seq_ewc"], default="fl")
    p.add_argument("--algo", choices=["fedavg", "fedprox", "fedavgm", "fedadam", "fedyogi", "lossweight"], default="fedavg")
    # encoder
    p.add_argument("--model", default="roberta-base", help="roberta-base or law-ai/InLegalBERT")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--trunc", choices=["head", "headtail"], default="head",
                   help="headtail = first half + last half of the tokens (keeps the end of the reasoning)")
    p.add_argument("--lora_targets", default="query,value", help="e.g. query,key,value,dense")
    p.add_argument("--lora_r", type=int, default=8, help="0 = fine-tune the whole encoder (heavy to communicate)")
    p.add_argument("--freeze_encoder", action="store_true", help="no LoRA, encoder frozen: heads only")
    # optimisation
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--eval_batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=6, help="local / central regimes")
    p.add_argument("--rounds", type=int, default=10, help="fl regime")
    p.add_argument("--local_epochs", type=float, default=1.0, help="per round; fractions allowed (e.g. 0.5)")
    p.add_argument("--frac", type=float, default=1.0, help="fraction of courts sampled per round")
    p.add_argument("--patience", type=int, default=0, help="stop after this many rounds/epochs without val improvement (0 = off)")
    p.add_argument("--force", action="store_true", help="rerun even if results already exist")
    # algorithm hyper-parameters
    p.add_argument("--mu", type=float, default=0.01, help="fedprox")
    p.add_argument("--beta", type=float, default=1.0, help="lossweight")
    p.add_argument("--server_lr", type=float, default=None, help="fedavgm/fedadam/fedyogi")
    p.add_argument("--server_momentum", type=float, default=0.9)
    p.add_argument("--tau", type=float, default=1e-6, help="fedadam/fedyogi adaptivity")
    p.add_argument("--finetune_epochs", type=float, default=0.0, help="personalisation after FL")
    p.add_argument("--init_from", default=None, help="ckpt.pt of a finished FL run: skip training, only fine-tune/evaluate its best model")
    p.add_argument("--ewc_lambda", type=float, default=100.0)
    p.add_argument("--seq_epochs", type=float, default=2.0)
    p.add_argument("--seq_order", default=None, help="comma-separated courts; default largest first")
    # loss weights
    p.add_argument("--w_dur", type=float, default=1.0)
    p.add_argument("--w_type", type=float, default=0.5)
    p.add_argument("--w_disp", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="debug: max training cases per court")
    return p.parse_args(argv)

# --------------------------------------------------------------------------- text masking
MON = r"(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|aug(ust)?|sep(t(ember)?)?|oct(ober)?|nov(ember)?|dec(ember)?)"
OUT = r"\b(allow(ed|s)?|dismiss(ed|al)?|dispos(ed|al)(\s+of)?|withdra(wn|w|wal)|rejected|not pressed|infructuous)\b"
def mask_text(t):
    t = re.sub(r"\b\d{1,2}(st|nd|rd|th)?[\s./-]*" + MON + r"[\s.,/-]*(\d{2,4})?", " DATE ", t, flags=re.I)
    t = re.sub(MON + r"\s+\d{1,2}(st|nd|rd|th)?,?\s*\d{2,4}", " DATE ", t, flags=re.I)
    t = re.sub(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", " DATE ", t)
    t = re.sub(r"\b(19|20)\d{2}\b", " YEAR ", t)
    t = re.sub(r"\b\d{2,}\b", " NUM ", t)
    t = re.sub(OUT, " OUTCOME ", t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip()

# --------------------------------------------------------------------------- data
def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def load_frame(args):
    df = pd.read_csv(args.data, low_memory=False)
    if "garbled" in df: df = df[~df.garbled.astype(bool)]
    df = df.reset_index(drop=True)
    df["text"] = df.input_text.fillna("").astype(str).map(mask_text)
    df["y_dur"] = np.log1p(df.duration_days.clip(lower=0)).astype("float32")
    df["y_disp"] = df.outcome_from_text.map({c: i for i, c in enumerate(DISP)}).fillna(-100).astype(int)
    types = sorted(df.case_type_group.fillna("OTHER").unique())   # agreed label set (ten families)
    df["y_type"] = df.case_type_group.fillna("OTHER").map({c: i for i, c in enumerate(types)}).astype(int)
    ner_cols = []
    if args.ner_csv:
        ner = pd.read_csv(args.ner_csv)
        ner_cols = [f"ner_{t}" for t in NER_TYPES]
        df = df.merge(ner[["cnr"] + ner_cols], on="cnr", how="left")
        df[ner_cols] = np.log1p(df[ner_cols].fillna(0).astype("float32"))
    return df, types, ner_cols

def tokenize(df, tok, args):
    key = hashlib.md5(f"{args.model}|{args.max_len}|{args.trunc}|{len(df)}|{df.cnr.iloc[0]}".encode()).hexdigest()[:10]
    path = os.path.join(args.out_dir, f"tok_cache_{key}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    if args.trunc == "head":
        ids = tok(df.text.tolist(), truncation=True, max_length=args.max_len)["input_ids"]
    else:
        tok.model_max_length = int(1e9)                     # silence length warnings
        full = tok(df.text.tolist(), truncation=False)["input_ids"]
        h = args.max_len // 2
        ids = [x if len(x) <= args.max_len else x[:h] + x[-(args.max_len - h):] for x in full]
    os.makedirs(args.out_dir, exist_ok=True); pickle.dump(ids, open(path, "wb"))
    return ids

class CaseDS(Dataset):
    def __init__(s, d, ner_cols):
        s.ids = d.ids.tolist()
        s.ner = d[ner_cols].values.astype(np.float32) if ner_cols else np.zeros((len(d), 0), np.float32)
        s.yd, s.yt, s.yo = d.y_dur.values, d.y_type.values, d.y_disp.values
    def __len__(s): return len(s.ids)
    def __getitem__(s, i): return s.ids[i], s.ner[i], s.yd[i], s.yt[i], s.yo[i]

def make_collate(pad_id):
    def collate(items):
        L = max(len(x[0]) for x in items); B = len(items)
        ids = torch.full((B, L), pad_id, dtype=torch.long); att = torch.zeros((B, L), dtype=torch.long)
        for i, x in enumerate(items):
            ids[i, :len(x[0])] = torch.tensor(x[0]); att[i, :len(x[0])] = 1
        return {"ids": ids, "att": att,
                "ner": torch.tensor(np.stack([x[1] for x in items])),
                "yd": torch.tensor([x[2] for x in items], dtype=torch.float32),
                "yt": torch.tensor([x[3] for x in items], dtype=torch.long),
                "yo": torch.tensor([x[4] for x in items], dtype=torch.long)}
    return collate

def build_clients(df, ner_cols, args, collate):
    courts = sorted(df.court.unique()); C = {}
    for c in courts:
        part = {}
        for sp in ["train", "val", "test"]:
            d = df[(df.court == c) & (df.split == sp)]
            if sp == "train" and args.limit: d = d.iloc[:args.limit]
            if sp != "train" and args.limit: d = d.iloc[:max(32, args.limit // 4)]
            part[sp] = CaseDS(d, ner_cols)
        ytr = part["train"].yo; cnt = np.bincount(ytr[ytr >= 0], minlength=4).astype(float)
        part["cw"] = torch.tensor(cnt.sum() / (4 * np.maximum(cnt, 1)), dtype=torch.float32)  # local stats only
        part["n"] = len(part["train"])
        part["dl"] = {sp: DataLoader(part[sp], batch_size=args.batch if sp == "train" else args.eval_batch,
                                     shuffle=(sp == "train"), collate_fn=collate) for sp in ["train", "val", "test"]}
        C[c] = part
    return courts, C

# --------------------------------------------------------------------------- model
class MultiTaskModel(nn.Module):
    def __init__(s, args, n_types, ner_dim):
        super().__init__()
        enc = AutoModel.from_pretrained(args.model)
        if args.freeze_encoder:
            for p in enc.parameters(): p.requires_grad = False
        elif args.lora_r > 0:
            from peft import LoraConfig, get_peft_model
            enc = get_peft_model(enc, LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.1,
                                                 target_modules=args.lora_targets.split(","), bias="none"))
        s.enc = enc; H = enc.config.hidden_size
        s.ner = nn.Sequential(nn.Linear(ner_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()) if ner_dim else None
        D = H + (64 if ner_dim else 0)
        s.drop = nn.Dropout(0.1)
        s.dur = nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Linear(256, 1))
        s.typ = nn.Linear(D, n_types)
        s.disp = nn.Sequential(nn.Linear(D + n_types, 256), nn.GELU(), nn.Linear(256, len(DISP)))
        nn.init.constant_(s.dur[-1].bias, 4.5)        # ~ log(90 days): sensible starting point
    def forward(s, b):
        h = s.enc(input_ids=b["ids"], attention_mask=b["att"]).last_hidden_state
        m = b["att"].unsqueeze(-1).to(h.dtype); h = (h * m).sum(1) / m.sum(1).clamp(min=1)
        if s.ner is not None: h = torch.cat([h, s.ner(b["ner"].to(h.dtype))], -1)
        h = s.drop(h)
        typ = s.typ(h)
        disp = s.disp(torch.cat([h, typ.softmax(-1).detach()], -1))   # cascade: case type -> disposal
        return s.dur(h).squeeze(-1), typ, disp

def trainable(model): return [(n, p) for n, p in model.named_parameters() if p.requires_grad]
def get_state(model): return {n: p.detach().float().cpu().clone() for n, p in trainable(model)}
def set_state(model, st):
    with torch.no_grad():
        for n, p in trainable(model): p.copy_(st[n].to(p.device, p.dtype))

# --------------------------------------------------------------------------- train / eval
def to_dev(b, dev): return {k: v.to(dev, non_blocking=True) for k, v in b.items()}

def task_loss(out, b, args, cw):
    dur, typ, disp = out
    l = args.w_dur * F.huber_loss(dur.float(), b["yd"], delta=1.0)
    l = l + args.w_type * F.cross_entropy(typ.float(), b["yt"])
    m = b["yo"] >= 0
    if m.any():
        l = l + args.w_disp * F.cross_entropy(disp.float()[m], b["yo"][m], weight=None if cw is None else cw.to(disp.device))
    return l

def local_train(model, dl, args, dev, epochs, cw, prox_ref=None, ewc=None):
    params = [p for _, p in trainable(model)]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    use_amp = dev.type == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if prox_ref is not None: prox = [(p, prox_ref[n].to(dev)) for n, p in trainable(model)]
    steps = max(1, int(round(epochs * len(dl)))); done, tot, cnt = 0, 0.0, 0
    model.train()
    while done < steps:
        for b in dl:
            b = to_dev(b, dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                loss = task_loss(model(b), b, args, cw)
            if prox_ref is not None:
                loss = loss + args.mu / 2 * sum(((p.float() - g) ** 2).sum() for p, g in prox)
            if ewc is not None: loss = loss + ewc.penalty(model)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update()
            tot += loss.item(); cnt += 1; done += 1
            if done >= steps: break
    return tot / max(cnt, 1)

@torch.no_grad()
def evaluate(model, dl, args, dev):
    model.eval(); use_amp = dev.type == "cuda"
    Yd, Pd, Yt, Pt, Yo, Po, loss, n = [], [], [], [], [], [], 0.0, 0
    for b in dl:
        b = to_dev(b, dev)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
            out = model(b)
        loss += task_loss(out, b, args, None).item() * len(b["yd"]); n += len(b["yd"])
        Yd.append(b["yd"].cpu()); Pd.append(out[0].float().cpu()); Yt.append(b["yt"].cpu())
        Pt.append(out[1].float().argmax(-1).cpu()); Yo.append(b["yo"].cpu()); Po.append(out[2].float().argmax(-1).cpu())
    yd, pd_, yt, pt, yo, po = [torch.cat(x).numpy() for x in (Yd, Pd, Yt, Pt, Yo, Po)]
    bk = lambda y: np.digitize(np.expm1(y), [30, 180, 730], right=True)
    lab = yo >= 0
    return {"loss": loss / max(n, 1), "n": n,
            "MAE_log": float(np.abs(yd - pd_).mean()),
            "Spearman": float(spearmanr(yd, pd_)[0]) if np.std(pd_) > 0 else float("nan"),
            "F1_4": float(f1_score(bk(yd), bk(pd_), average="macro", labels=[0, 1, 2, 3])),
            "type_F1": float(f1_score(yt, pt, average="macro")),
            "disp_F1": float(f1_score(yo[lab], po[lab], average="macro", labels=[0, 1, 2, 3])) if lab.any() else float("nan")}

def eval_split(model, courts, C, split, args, dev):
    res = {c: evaluate(model, C[c]["dl"][split], args, dev) for c in courts}
    n = np.array([res[c]["n"] for c in courts], float)
    return res, float(np.dot([res[c]["loss"] for c in courts], n) / n.sum())

# --------------------------------------------------------------------------- FL server
class Server:
    """Aggregation + server optimisers (FedAvgM / FedAdam / FedYogi, Reddi et al. 2021)."""
    def __init__(s, args, g):
        s.a = args; s.m = {k: torch.zeros_like(v) for k, v in g.items()}
        s.v = {k: torch.full_like(v, args.tau ** 2) for k, v in g.items()}
        s.lr = args.server_lr if args.server_lr is not None else {"fedavgm": 1.0, "fedadam": 1e-3, "fedyogi": 1e-3}.get(args.algo, 1.0)
    def weights(s, n, losses):
        w = np.asarray(n, float)
        if s.a.algo == "lossweight": w = w * np.exp(-s.a.beta * (np.asarray(losses) - np.min(losses)))
        return w / w.sum()
    def step(s, g, states, w):
        avg = {k: sum(wi * st[k] for wi, st in zip(w, states)) for k in g}
        algo = s.a.algo
        if algo in ("fedavg", "fedprox", "lossweight"): return avg
        new = {}
        for k in g:
            d = avg[k] - g[k]                          # pseudo-gradient (descent direction)
            if algo == "fedavgm":
                s.m[k] = s.a.server_momentum * s.m[k] + d; new[k] = g[k] + s.lr * s.m[k]
            else:
                s.m[k] = 0.9 * s.m[k] + 0.1 * d
                if algo == "fedadam": s.v[k] = 0.99 * s.v[k] + 0.01 * d * d
                else: s.v[k] = s.v[k] - 0.01 * d * d * torch.sign(s.v[k] - d * d)
                new[k] = g[k] + s.lr * s.m[k] / (s.v[k].sqrt() + s.a.tau)
        return new

# --------------------------------------------------------------------------- EWC
class EWC:
    def __init__(s, lam): s.lam, s.F, s.anchor = lam, None, None
    def consolidate(s, model, dl, args, dev, cw, n_batches=20):
        Fc = {n: torch.zeros_like(p) for n, p in trainable(model)}; model.eval(); k = 0
        for b in dl:
            b = to_dev(b, dev); model.zero_grad()
            task_loss(model(b), b, args, cw).backward()
            for n, p in trainable(model):
                if p.grad is not None: Fc[n] += p.grad.detach().float() ** 2
            k += 1
            if k >= n_batches: break
        Fc = {n: v / k for n, v in Fc.items()}
        s.F = Fc if s.F is None else {n: s.F[n] + Fc[n] for n in Fc}
        s.anchor = {n: p.detach().float().clone() for n, p in trainable(model)}
    def penalty(s, model):
        if s.F is None: return 0.0
        return s.lam / 2 * sum((s.F[n] * (p.float() - s.anchor[n]) ** 2).sum() for n, p in trainable(model))

# --------------------------------------------------------------------------- regimes
def run_local(model, courts, C, args, dev, init):
    rows = {}
    for c in courts:
        set_state(model, init); best = (np.inf, None, 0)
        for e in range(int(args.epochs)):
            local_train(model, C[c]["dl"]["train"], args, dev, 1, C[c]["cw"])
            v = evaluate(model, C[c]["dl"]["val"], args, dev)["loss"]
            print(f"  local {c} epoch {e+1}: val loss {v:.4f}", flush=True)
            if v < best[0]: best = (v, get_state(model), e + 1)
            if args.patience and e + 1 - best[2] >= args.patience: break
        set_state(model, best[1]); rows[c] = evaluate(model, C[c]["dl"]["test"], args, dev)
        rows[c].update(sel_val_loss=best[0], best_step=best[2])
    return {"local_only": rows}

def run_central(model, courts, C, args, dev, init, collate):
    pooled = ConcatDataset([C[c]["train"] for c in courts])
    dl = DataLoader(pooled, batch_size=args.batch, shuffle=True, collate_fn=collate)
    y = np.concatenate([C[c]["train"].yo for c in courts]); cnt = np.bincount(y[y >= 0], minlength=4).astype(float)
    cw = torch.tensor(cnt.sum() / (4 * np.maximum(cnt, 1)), dtype=torch.float32)
    set_state(model, init); best = (np.inf, None, 0)
    for e in range(int(args.epochs)):
        local_train(model, dl, args, dev, 1, cw)
        _, v = eval_split(model, courts, C, "val", args, dev)
        print(f"  central epoch {e+1}: val loss {v:.4f}", flush=True)
        if v < best[0]: best = (v, get_state(model), e + 1)
        if args.patience and e + 1 - best[2] >= args.patience: break
    set_state(model, best[1]); res = eval_split(model, courts, C, "test", args, dev)[0]
    for c in res: res[c].update(sel_val_loss=best[0], best_step=best[2])
    return {"centralized": res}

def run_fl(model, courts, C, args, dev, init, out_dir):
    g = {k: v.clone() for k, v in init.items()}; server = Server(args, g)
    best = (np.inf, None, 0); hist = []; rng = np.random.default_rng(args.seed); start = 1
    ck = os.path.join(out_dir, "ckpt.pt")
    if args.init_from:                                       # reuse a finished run (e.g. FedAvg) for +ft
        st = torch.load(args.init_from, weights_only=False); best = st["best"]; hist = st["hist"]; start = args.rounds + 1
        print(f"  loaded best model (round {best[2]}) from {args.init_from}", flush=True)
    elif os.path.exists(ck):                                   # resume after a Kaggle session ended
        st = torch.load(ck, weights_only=False)
        g, server.m, server.v, best, hist, start = st["g"], st["m"], st["v"], st["best"], st["hist"], st["round"] + 1
        rng.bit_generator.state = st["rng"]
        print(f"  resumed from round {start - 1}", flush=True)
    for r in range(start, args.rounds + 1):
        t0 = time.time()
        part = [c for c in courts if rng.random() < args.frac] or [rng.choice(courts)]
        states, losses = [], []
        for c in part:
            set_state(model, g)
            L = local_train(model, C[c]["dl"]["train"], args, dev, args.local_epochs, C[c]["cw"],
                            prox_ref=g if args.algo == "fedprox" else None)
            states.append(get_state(model)); losses.append(L)
        w = server.weights([C[c]["n"] for c in part], losses)
        g = server.step(g, states, w)
        set_state(model, g)
        val, vloss = eval_split(model, courts, C, "val", args, dev)
        rec = {"round": r, "val_loss": vloss, "secs": time.time() - t0,
               **{f"w_{c}": wi for c, wi in zip(part, w)},
               **{f"{m}_mean": float(np.nanmean([val[c][m] for c in courts])) for m in ["MAE_log", "disp_F1", "type_F1"]}}
        hist.append(rec)
        print(f"  round {r:3d} | val loss {vloss:.4f} | MAE_log {rec['MAE_log_mean']:.3f} | "
              f"disp F1 {rec['disp_F1_mean']:.3f} | {rec['secs']:.0f}s", flush=True)
        if vloss < best[0]: best = (vloss, {k: v.clone() for k, v in g.items()}, r)
        torch.save({"g": g, "m": server.m, "v": server.v, "best": best, "hist": hist, "round": r,
                    "rng": rng.bit_generator.state}, ck)
        if args.patience and r - best[2] >= args.patience:
            print(f"  early stop: no val improvement for {args.patience} rounds", flush=True); break
    pd.DataFrame(hist).to_csv(os.path.join(out_dir, "history.csv"), index=False)
    last = hist[-1]["round"] if hist else 0
    print(f"  best round {best[2]}/{last}" + ("  <- still improving: add rounds" if best[2] == last == args.rounds else ""))
    set_state(model, best[1]); res = {args.algo: eval_split(model, courts, C, "test", args, dev)[0]}
    if args.finetune_epochs > 0:
        ft, fv = {}, {}
        for c in courts:
            set_state(model, best[1])
            local_train(model, C[c]["dl"]["train"], args, dev, args.finetune_epochs, C[c]["cw"])
            fv[c] = evaluate(model, C[c]["dl"]["val"], args, dev); ft[c] = evaluate(model, C[c]["dl"]["test"], args, dev)
        n = np.array([fv[c]["n"] for c in courts], float)
        ft_val = float(np.dot([fv[c]["loss"] for c in courts], n) / n.sum())
    for per in res.values():
        for c in per: per[c].update(sel_val_loss=best[0], best_step=best[2])
    if args.finetune_epochs > 0:
        for c in ft: ft[c].update(sel_val_loss=ft_val, best_step=best[2])
        res[args.algo + "+ft"] = ft
    return res

def run_seq_ewc(model, courts, C, args, dev, init):
    order = args.seq_order.split(",") if args.seq_order else sorted(courts, key=lambda c: -C[c]["n"])
    set_state(model, init); ewc = EWC(args.ewc_lambda)
    for c in order:
        L = local_train(model, C[c]["dl"]["train"], args, dev, args.seq_epochs, C[c]["cw"], ewc=ewc)
        ewc.consolidate(model, C[c]["dl"]["train"], args, dev, C[c]["cw"])
        print(f"  seq_ewc trained on {c}: train loss {L:.4f}", flush=True)
    _, v = eval_split(model, courts, C, "val", args, dev)
    res = eval_split(model, courts, C, "test", args, dev)[0]
    for c in res: res[c].update(sel_val_loss=v, best_step=len(order))
    return {"seq_ewc": res}

# --------------------------------------------------------------------------- main
def main(argv=None):
    args = get_args(argv); seed_all(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = ""
    if args.regime == "fl":
        tag = f"_{args.algo}" + {"fedprox": f"_mu{args.mu}", "lossweight": f"_beta{args.beta}"}.get(args.algo, "")
        if args.algo in ("fedavgm", "fedadam", "fedyogi") and args.server_lr is not None: tag += f"_slr{args.server_lr}"
    if args.finetune_epochs > 0: tag += f"_ft{args.finetune_epochs}"
    extra = f"_L{args.max_len}_lr{args.lr:g}" + ("_ht" if args.trunc == "headtail" else "") \
            + (f"_r{args.lora_r}" if args.lora_r != 8 else "") + ("_kvd" if args.lora_targets != "query,value" else "")
    name = args.run_name or (f"{args.regime}{tag}_{args.model.split('/')[-1]}{extra}"
                             + ("_ner" if args.ner_csv else "") + f"_s{args.seed}")
    out_dir = os.path.join(args.out_dir, name); os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, "per_court_test.csv")) and not args.force:
        print(f"{name}: already done, skipping (use --force to rerun)"); return
    json.dump(vars(args), open(os.path.join(out_dir, "config.json"), "w"), indent=2)
    print(f"device {dev} | run {name}")

    df, types, ner_cols = load_frame(args)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    df["ids"] = tokenize(df, tok, args)
    collate = make_collate(tok.pad_token_id)
    courts, C = build_clients(df, ner_cols, args, collate)
    print("clients:", {c: C[c]["n"] for c in courts}, "| NER branch:", bool(ner_cols))

    model = MultiTaskModel(args, len(types), len(ner_cols)).to(dev)
    n_tr = sum(p.numel() for _, p in trainable(model)); n_all = sum(p.numel() for p in model.parameters())
    print(f"trainable/communicated params: {n_tr:,} of {n_all:,} ({100*n_tr/n_all:.2f}%)")
    init = get_state(model)

    if args.regime == "local": res = run_local(model, courts, C, args, dev, init)
    elif args.regime == "central": res = run_central(model, courts, C, args, dev, init, collate)
    elif args.regime == "fl": res = run_fl(model, courts, C, args, dev, init, out_dir)
    else: res = run_seq_ewc(model, courts, C, args, dev, init)

    rows = [{"method": m, "court": c, "seed": args.seed, "model": args.model, "run": name,
             "max_len": args.max_len, "lr": args.lr, "trunc": args.trunc, **met}
            for m, per in res.items() for c, met in per.items()]
    out = pd.DataFrame(rows); out.to_csv(os.path.join(out_dir, "per_court_test.csv"), index=False)
    allp = os.path.join(args.out_dir, "all_results.csv")
    if os.path.exists(allp):                                  # tolerate older files with fewer columns
        prev = pd.read_csv(allp); out_all = pd.concat([prev[prev.run != name], out], ignore_index=True)
    else: out_all = out
    out_all.to_csv(allp, index=False)
    for m, d in out.groupby("method"):
        print(f"\n== {m} (test) ==")
        print(d.set_index("court")[["MAE_log", "Spearman", "F1_4", "disp_F1", "type_F1"]].round(3).to_string())
        print(f"mean: MAE_log {d.MAE_log.mean():.3f} | Spearman {d.Spearman.mean():.3f} | F1_4 {d.F1_4.mean():.3f} | "
              f"disp_F1 {d.disp_F1.mean():.3f} | worst MAE_log {d.MAE_log.max():.3f} | worst disp_F1 {d.disp_F1.min():.3f}")

if __name__ == "__main__":
    main()
