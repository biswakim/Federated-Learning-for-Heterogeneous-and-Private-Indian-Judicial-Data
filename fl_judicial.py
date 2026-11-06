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
    seq_ewc    continual learning (naive): courts visited once each, in order; evaluated
               only at the very end. Kept as a weak baseline -- see fed_continual below.
    fed_continual  federated continual onboarding: courts join one at a time. The SHARED
               part of the model (encoder + whichever heads are not in --local_heads)
               is updated at each new court and protected with EWC (Kirkpatrick et al.
               2017), optionally mixed with a small replay buffer of earlier courts'
               cases. Any head named in --local_heads gets its own permanent per-court
               copy, trained only while that court is being onboarded, so it cannot be
               forgotten by later courts. Every already-onboarded court is re-evaluated
               after every new court joins, so forgetting is actually measured, not just
               assumed (forgetting.csv, forgetting_summary.csv in the run's out_dir).

Learned per-task mixing (--mix_heads)            [proposed method, idea 1]
    For each listed head (dur, typ, disp) every court keeps BOTH the shared head (averaged
    by the server) and its own local copy, and blends their outputs with a learned weight
        out = a * shared_head(x) + (1 - a) * local_head(x),   a = sigmoid(alpha_court,task)
    alpha starts at 0 (a = 0.5) and is trained locally (own learning rate, --mix_lr); the
    local heads and alpha never leave the court. This replaces the fixed choice of
    --local_heads with a learned, per-court, per-task one. Learned weights are written to
    mixing_weights.csv and added to the per-court results (columns mix_dur, mix_typ, mix_disp).
    Related: APFL (Deng et al. 2020), FedALA (Zhang et al. 2023) blend global and local
    models per client; here the blend is per TASK in a multi-task model.

Task-specific reading of the judgment (--segments 2)   [proposed method, idea 2]
    Each case is encoded twice by the shared encoder: its first --max_len tokens and its
    last --max_len tokens. Each task head has its own learned query that weights the two
    segment embeddings (softmax over {start, end}), so the duration head can read the start
    and the disposal head the end of the judgment. Average weights on the end segment are
    reported per court (columns seg_end_dur, seg_end_typ, seg_end_disp). --trunc is ignored.

Task-decoupled sharing (--local_heads)
    "none" (default): every trainable parameter -- encoder LoRA, NER MLP, all three task
    heads -- is shared and averaged by the server, i.e. plain FedAvg/FedProx/etc as before.
    A comma list of head names (dur, typ, disp) instead keeps those heads un-averaged:
    each court keeps and trains its own copy, while the encoder (and any head not listed)
    stays shared. This is motivated by two measurements on this dataset: court explains
    only ~5% of the variance in log-duration but court x case-type explains ~17%, i.e.
    duration depends on how each court works rather than on the case text alone, so a
    single shared duration head may be fighting itself across courts; disposal-nature
    language, by contrast, looks similar across courts. Applies to --regime fl and
    --regime fed_continual. Not the same as --finetune_epochs, which personalises every
    parameter (shared and local alike) after training finishes, as a separate step.

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
import argparse, os, re, json, time, random, pickle, hashlib, copy
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import AutoTokenizer, AutoModel
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

DISP = ["ALLOWED", "DISMISSED", "DISPOSED", "WITHDRAWN"]
TASKS = ["dur", "typ", "disp"]

def parse_heads(sv, flag="--local_heads"):
    if not sv or sv.lower() == "none": return set()
    out = set(x.strip() for x in sv.split(",") if x.strip())
    bad = out - set(TASKS)
    assert not bad, f"{flag}: unknown head(s) {bad}, choose from {TASKS}"
    return out
NER_TYPES = ["COURT", "PETITIONER", "RESPONDENT", "JUDGE", "LAWYER", "ORG", "GPE",
             "STATUTE", "PROVISION", "PRECEDENT", "WITNESS", "OTHER_PERSON"]  # DATE, CASE_NUMBER excluded

# --------------------------------------------------------------------------- args
def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="judgments_labelled.csv")
    p.add_argument("--ner_csv", default=None, help="output of ner_features.py (optional NER branch)")
    p.add_argument("--out_dir", default="runs")
    p.add_argument("--run_name", default=None)
    p.add_argument("--regime", choices=["local", "central", "fl", "seq_ewc", "fed_continual"], default="fl")
    p.add_argument("--mix_heads", default="none",
                   help="comma list of head(s) with a learned per-court shared/local blend: dur,typ,disp, or 'none'")
    p.add_argument("--mix_lr", type=float, default=0.01, help="learning rate of the blend weights")
    p.add_argument("--segments", type=int, choices=[1, 2], default=1,
                   help="2 = encode start and end of the judgment separately, task-specific attention over them")
    p.add_argument("--local_heads", default="none",
                   help="comma list of head(s) kept un-averaged per court: dur,typ,disp, or 'none' (plain FL)")
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
    p.add_argument("--onboard_order", default=None,
                   help="fed_continual: comma-separated courts in join order; default = --seq_order, else largest first")
    p.add_argument("--replay_size", type=int, default=0,
                   help="fed_continual: cases kept per onboarded court for replay (0 = no replay)")
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
    trunc = "seg2" if args.segments == 2 else args.trunc
    key = hashlib.md5(f"{args.model}|{args.max_len}|{trunc}|{len(df)}|{df.cnr.iloc[0]}".encode()).hexdigest()[:10]
    path = os.path.join(args.out_dir, f"tok_cache_{key}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    if args.segments == 2:                                  # (start, end) pair per case
        tok.model_max_length = int(1e9)
        full = tok(df.text.tolist(), truncation=False)["input_ids"]
        L = args.max_len
        ids = [(x, x) if len(x) <= L else (x[:L - 1] + [x[-1]], [x[0]] + x[-(L - 1):]) for x in full]
    elif args.trunc == "head":
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
        s.ids = d.ids.tolist()                     # list of ids, or (start_ids, end_ids) pairs
        s.ner = d[ner_cols].values.astype(np.float32) if ner_cols else np.zeros((len(d), 0), np.float32)
        s.yd, s.yt, s.yo = d.y_dur.values, d.y_type.values, d.y_disp.values
    def __len__(s): return len(s.ids)
    def __getitem__(s, i): return s.ids[i], s.ner[i], s.yd[i], s.yt[i], s.yo[i]

def make_collate(pad_id):
    def pad(seqs):
        L = max(len(x) for x in seqs); B = len(seqs)
        ids = torch.full((B, L), pad_id, dtype=torch.long); att = torch.zeros((B, L), dtype=torch.long)
        for i, x in enumerate(seqs):
            ids[i, :len(x)] = torch.tensor(x); att[i, :len(x)] = 1
        return ids, att
    def collate(items):
        two = isinstance(items[0][0], tuple)
        if two:   # pad start and end segments to a common length so they share one encoder pass
            ids, att = pad([x[0][0] for x in items] + [x[0][1] for x in items]); B = len(items)
            extra = {"ids": ids[:B], "att": att[:B], "ids2": ids[B:], "att2": att[B:]}
        else:
            ids, att = pad([x[0] for x in items]); extra = {"ids": ids, "att": att}
        return {**extra,
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
        s.enc = enc; H = enc.config.hidden_size; s.H = H
        s.ner = nn.Sequential(nn.Linear(ner_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()) if ner_dim else None
        D = H + (64 if ner_dim else 0)
        s.drop = nn.Dropout(0.1)
        s.dur = nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Linear(256, 1))
        s.typ = nn.Linear(D, n_types)
        s.disp = nn.Sequential(nn.Linear(D + n_types, 256), nn.GELU(), nn.Linear(256, len(DISP)))
        nn.init.constant_(s.dur[-1].bias, 4.5)        # ~ log(90 days): sensible starting point
        # idea 2: one query per task over the {start, end} segment embeddings (zeros = 50/50 start)
        s.segments = getattr(args, "segments", 1)
        if s.segments == 2:
            s.segq = nn.ParameterDict({t: nn.Parameter(torch.zeros(H)) for t in TASKS})
        # idea 1: local copy of each mixed head + one blend logit per head (0 -> a = 0.5)
        s.mix_heads = parse_heads(getattr(args, "mix_heads", "none"), "--mix_heads")
        for t in s.mix_heads:
            setattr(s, t + "_loc", copy.deepcopy(getattr(s, t)))
        if s.mix_heads:
            s.mix = nn.ParameterDict({t: nn.Parameter(torch.zeros(1)) for t in s.mix_heads})
        s.last_seg_w = None
    def encode(s, ids, att):
        h = s.enc(input_ids=ids, attention_mask=att).last_hidden_state
        m = att.unsqueeze(-1).to(h.dtype); return (h * m).sum(1) / m.sum(1).clamp(min=1)
    def head(s, t, x):
        out = getattr(s, t)(x)
        if t in s.mix_heads:
            a = torch.sigmoid(s.mix[t]).to(out.dtype)
            out = a * out + (1 - a) * getattr(s, t + "_loc")(x)
        return out
    def forward(s, b):
        if s.segments == 2 and "ids2" in b:
            B = b["ids"].shape[0]
            hs = s.encode(torch.cat([b["ids"], b["ids2"]]), torch.cat([b["att"], b["att2"]]))
            segs = torch.stack([hs[:B], hs[B:]], 1)                           # B x 2 x H
            reps, w_end = {}, {}
            for t in TASKS:
                w = (segs @ s.segq[t].to(segs.dtype) / s.H ** 0.5).softmax(1)  # B x 2
                reps[t] = (w.unsqueeze(-1) * segs).sum(1); w_end[t] = w[:, 1].detach().float()
            s.last_seg_w = w_end
        else:
            h = s.encode(b["ids"], b["att"]); reps = {t: h for t in TASKS}; s.last_seg_w = None
        if s.ner is not None:
            nf = s.ner(b["ner"].to(reps["dur"].dtype)); reps = {t: torch.cat([r, nf], -1) for t, r in reps.items()}
        reps = {t: s.drop(r) for t, r in reps.items()}
        typ = s.head("typ", reps["typ"])
        disp = s.head("disp", torch.cat([reps["disp"], typ.softmax(-1).detach()], -1))   # cascade: type -> disposal
        return s.head("dur", reps["dur"]).squeeze(-1), typ, disp

def trainable(model): return [(n, p) for n, p in model.named_parameters() if p.requires_grad]
def get_state(model): return {n: p.detach().float().cpu().clone() for n, p in trainable(model)}
def set_state(model, st):
    with torch.no_grad():
        for n, p in trainable(model): p.copy_(st[n].to(p.device, p.dtype))

# --------------------------------------------------------------------------- task-decoupled sharing
HEAD_PREFIXES = {"dur": "dur.", "typ": "typ.", "disp": "disp."}
def param_group(name):
    """Which task head a trainable parameter belongs to, or 'shared' for the encoder/NER
    branch, which is always averaged regardless of --local_heads."""
    for g, pre in HEAD_PREFIXES.items():
        if name.startswith(pre): return g
    return "shared"

def parse_local_heads(s): return parse_heads(s, "--local_heads")

PERSONAL_PREFIXES = ("dur_loc.", "typ_loc.", "disp_loc.", "mix.")   # --mix_heads: always court-only
def is_local(name, local_heads):
    return name.startswith(PERSONAL_PREFIXES) or param_group(name) in local_heads

def split_state(state, local_heads):
    """Splits a full parameter dict into (shared, local). Local = heads listed in
    --local_heads plus, with --mix_heads, the local head copies and blend weights.
    With neither flag, everything is 'shared' -- identical to plain FedAvg."""
    shared, local = {}, {}
    for k, v in state.items():
        (local if is_local(k, local_heads) else shared)[k] = v
    return shared, local

def merge_state(shared, local): return {**shared, **local}

class ReplayBuffer:
    """Tiny per-court cache of raw training examples for continual learning. Mixed into
    later courts' batches so the shared encoder keeps seeing a trickle of earlier courts'
    cases. NOTE: this keeps a handful of real (already-masked) case texts in memory for the
    rest of the run -- a privacy/communication trade-off against the 'raw data never leaves
    the client' ideal of plain FL; report it as such if used."""
    def __init__(s, cap_per_court): s.cap, s.data = cap_per_court, {}
    def add(s, court, dataset, rng):
        n = len(dataset); idx = rng.choice(n, size=min(s.cap, n), replace=False)
        s.data[court] = [dataset[int(i)] for i in idx]
    def sample(s, k, rng, exclude=None):
        pool = [x for c, xs in s.data.items() if c != exclude for x in xs]
        if not pool or k <= 0: return None
        idx = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
        return [pool[int(i)] for i in idx]

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

def make_opt(model, args):
    """AdamW. The few scalar-like parameters of the proposed methods (blend logits of
    --mix_heads, segment queries of --segments 2) get their own larger lr (--mix_lr) and no
    weight decay, so they can move within a normal number of rounds."""
    fast = lambda n: n.startswith(("mix.", "segq."))
    mix = [p for n, p in trainable(model) if fast(n)]
    rest = [p for n, p in trainable(model) if not fast(n)]
    groups = [{"params": rest, "lr": args.lr, "weight_decay": 0.01}]
    if mix: groups.append({"params": mix, "lr": getattr(args, "mix_lr", 0.01), "weight_decay": 0.0})
    return torch.optim.AdamW(groups), rest + mix

def local_train(model, dl, args, dev, epochs, cw, prox_ref=None, ewc=None):
    opt, params = make_opt(model, args)
    use_amp = dev.type == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if prox_ref is not None:                                   # only pull parameters that HAVE a global
        prox = [(p, prox_ref[n].to(dev)) for n, p in trainable(model) if n in prox_ref]  # reference (shared ones)
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
    segw = {t: [] for t in TASKS}
    for b in dl:
        b = to_dev(b, dev)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
            out = model(b)
        if getattr(model, "last_seg_w", None):
            for t in TASKS: segw[t].append(model.last_seg_w[t].cpu())
        loss += task_loss(out, b, args, None).item() * len(b["yd"]); n += len(b["yd"])
        Yd.append(b["yd"].cpu()); Pd.append(out[0].float().cpu()); Yt.append(b["yt"].cpu())
        Pt.append(out[1].float().argmax(-1).cpu()); Yo.append(b["yo"].cpu()); Po.append(out[2].float().argmax(-1).cpu())
    yd, pd_, yt, pt, yo, po = [torch.cat(x).numpy() for x in (Yd, Pd, Yt, Pt, Yo, Po)]
    bk = lambda y: np.digitize(np.expm1(y), [30, 180, 730], right=True)
    lab = yo >= 0
    seg = {f"seg_end_{t}": float(torch.cat(segw[t]).mean()) for t in TASKS if segw[t]}
    return {**seg, "loss": loss / max(n, 1), "n": n,
            "MAE_log": float(np.abs(yd - pd_).mean()),
            "Spearman": float(spearmanr(yd, pd_)[0]) if np.std(pd_) > 0 else float("nan"),
            "F1_4": float(f1_score(bk(yd), bk(pd_), average="macro", labels=[0, 1, 2, 3])),
            "type_F1": float(f1_score(yt, pt, average="macro")),
            "disp_F1": float(f1_score(yo[lab], po[lab], average="macro", labels=[0, 1, 2, 3])) if lab.any() else float("nan")}

def eval_split(model, courts, C, split, args, dev, g=None, client_local=None):
    """Evaluates every court. With g/client_local given (task-decoupled runs), each court
    is evaluated under the shared weights plus ITS OWN local-head copy; otherwise the
    model's current weights are used unchanged for every court (plain FL, local, central)."""
    res = {}
    for c in courts:
        if g is not None: set_state(model, merge_state(g, client_local[c]))
        res[c] = evaluate(model, C[c]["dl"][split], args, dev)
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
    """group_filter(name)->bool restricts Fisher tracking and the penalty to a subset of
    trainable parameters. Default (None) covers everything, as in the original seq_ewc
    baseline. fed_continual passes a filter that keeps EWC on the SHARED parameters only:
    a court's own local head needs no protection, since nothing else ever trains it."""
    def __init__(s, lam, group_filter=None): s.lam, s.F, s.anchor, s.keep = lam, None, None, group_filter or (lambda n: True)
    def consolidate(s, model, dl, args, dev, cw, n_batches=20):
        names = [n for n, p in trainable(model) if s.keep(n)]
        Fc = {n: torch.zeros_like(p) for n, p in trainable(model) if n in names}; model.eval(); k = 0
        for b in dl:
            b = to_dev(b, dev); model.zero_grad()
            task_loss(model(b), b, args, cw).backward()
            for n, p in trainable(model):
                if n in names and p.grad is not None: Fc[n] += p.grad.detach().float() ** 2
            k += 1
            if k >= n_batches: break
        Fc = {n: v / max(k, 1) for n, v in Fc.items()}
        s.F = Fc if s.F is None else {n: s.F[n] + Fc[n] for n in Fc}
        s.anchor = {n: p.detach().float().clone() for n, p in trainable(model) if n in names}
    def penalty(s, model):
        if s.F is None: return 0.0
        return s.lam / 2 * sum((s.F[n] * (p.float() - s.anchor[n]) ** 2).sum()
                               for n, p in trainable(model) if n in s.F)

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
    local_heads = parse_local_heads(args.local_heads)
    g, local0 = split_state(init, local_heads)                 # g: shared/averaged; local0: per-court copies start here
    client_local = {c: {k: v.clone() for k, v in local0.items()} for c in courts}
    server = Server(args, g)
    best = (np.inf, None, None, 0); hist = []; rng = np.random.default_rng(args.seed); start = 1
    ck = os.path.join(out_dir, "ckpt.pt")
    if args.init_from:                                       # reuse a finished run (e.g. FedAvg) for +ft
        st = torch.load(args.init_from, weights_only=False); best = st["best"]; hist = st["hist"]; start = args.rounds + 1
        if len(best) == 3: best = (best[0], best[1], {c: {} for c in courts}, best[2])   # older ckpt, no local heads
        print(f"  loaded best model (round {best[3]}) from {args.init_from}", flush=True)
    elif os.path.exists(ck):                                   # resume after a Kaggle session ended
        st = torch.load(ck, weights_only=False)
        g, server.m, server.v, best, hist, start = st["g"], st["m"], st["v"], st["best"], st["hist"], st["round"] + 1
        client_local = st.get("client_local", client_local)
        rng.bit_generator.state = st["rng"]
        print(f"  resumed from round {start - 1}", flush=True)
    for r in range(start, args.rounds + 1):
        t0 = time.time()
        part = [c for c in courts if rng.random() < args.frac] or [rng.choice(courts)]
        states, losses = [], []
        for c in part:
            set_state(model, merge_state(g, client_local[c]))
            L = local_train(model, C[c]["dl"]["train"], args, dev, args.local_epochs, C[c]["cw"],
                            prox_ref=g if args.algo == "fedprox" else None)
            shared_c, client_local[c] = split_state(get_state(model), local_heads)
            states.append(shared_c); losses.append(L)
        w = server.weights([C[c]["n"] for c in part], losses)
        g = server.step(g, states, w)
        val, vloss = eval_split(model, courts, C, "val", args, dev, g, client_local)
        rec = {"round": r, "val_loss": vloss, "secs": time.time() - t0,
               **{f"w_{c}": wi for c, wi in zip(part, w)},
               **{f"{m}_mean": float(np.nanmean([val[c][m] for c in courts])) for m in ["MAE_log", "disp_F1", "type_F1"]}}
        hist.append(rec)
        print(f"  round {r:3d} | val loss {vloss:.4f} | MAE_log {rec['MAE_log_mean']:.3f} | "
              f"disp F1 {rec['disp_F1_mean']:.3f} | {rec['secs']:.0f}s", flush=True)
        if vloss < best[0]:
            best = (vloss, {k: v.clone() for k, v in g.items()},
                    {c: {k: v.clone() for k, v in cl.items()} for c, cl in client_local.items()}, r)
        torch.save({"g": g, "m": server.m, "v": server.v, "best": best, "hist": hist, "round": r,
                    "client_local": client_local, "rng": rng.bit_generator.state}, ck)
        if args.patience and r - best[3] >= args.patience:
            print(f"  early stop: no val improvement for {args.patience} rounds", flush=True); break
    pd.DataFrame(hist).to_csv(os.path.join(out_dir, "history.csv"), index=False)
    last = hist[-1]["round"] if hist else 0
    print(f"  best round {best[3]}/{last}" + ("  <- still improving: add rounds" if best[3] == last == args.rounds else ""))
    best_g, best_local = best[1], best[2]
    res = {args.algo: eval_split(model, courts, C, "test", args, dev, best_g, best_local)[0]}
    mixk = sorted(k for k in next(iter(best_local.values()), {}) if k.startswith("mix."))
    if mixk:                               # learned weight on the SHARED head, per court and task
        mw = pd.DataFrame([{"court": c, **{"mix_" + k.split(".")[1]: float(torch.sigmoid(best_local[c][k]))
                                           for k in mixk}} for c in courts])
        mw.to_csv(os.path.join(out_dir, "mixing_weights.csv"), index=False)
        print("  learned weight on the shared head (1 = shared only, 0 = local only):")
        print(mw.round(3).to_string(index=False))
        for c in courts: res[args.algo][c].update({k: v for k, v in mw.set_index("court").loc[c].items()})
    if args.finetune_epochs > 0:
        ft, fv = {}, {}
        for c in courts:
            set_state(model, merge_state(best_g, best_local[c]))
            local_train(model, C[c]["dl"]["train"], args, dev, args.finetune_epochs, C[c]["cw"])
            fv[c] = evaluate(model, C[c]["dl"]["val"], args, dev); ft[c] = evaluate(model, C[c]["dl"]["test"], args, dev)
        n = np.array([fv[c]["n"] for c in courts], float)
        ft_val = float(np.dot([fv[c]["loss"] for c in courts], n) / n.sum())
    for per in res.values():
        for c in per: per[c].update(sel_val_loss=best[0], best_step=best[3])
    if args.finetune_epochs > 0:
        for c in ft: ft[c].update(sel_val_loss=ft_val, best_step=best[3])
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

def local_train_continual(model, dataset, args, dev, epochs, cw, collate, rng, ewc=None, replay=None, replay_exclude=None):
    """Like local_train, but batches are built from the raw dataset (not a fixed
    DataLoader) so a replay batch can be concatenated in before collation."""
    opt, params = make_opt(model, args)
    use_amp = dev.type == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    idx_all = np.arange(len(dataset)); steps_per_epoch = max(1, len(dataset) // args.batch)
    steps = max(1, int(round(epochs * steps_per_epoch))); done, tot, cnt = 0, 0.0, 0
    model.train()
    while done < steps:
        rng.shuffle(idx_all)
        for s in range(0, len(idx_all), args.batch):
            own = [dataset[int(i)] for i in idx_all[s:s + args.batch]]
            rep = replay.sample(len(own) // 2, rng, exclude=replay_exclude) if replay is not None else None
            b = to_dev(collate(own + rep if rep else own), dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                loss = task_loss(model(b), b, args, cw)
            if ewc is not None: loss = loss + ewc.penalty(model)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update()
            tot += loss.item(); cnt += 1; done += 1
            if done >= steps: break
    return tot / max(cnt, 1)

def run_fed_continual(model, courts, C, args, dev, init, out_dir, collate):
    """Courts join one at a time. The shared encoder (+ any head not in --local_heads)
    carries over and is EWC-protected; a --local_heads head gets a fresh per-court copy
    trained only during that court's own onboarding stage, so it is never at risk of being
    forgotten by construction -- which is itself part of what this regime tests. Every
    already-onboarded court is re-evaluated after each new court joins (forgetting.csv),
    unlike --regime seq_ewc, which only checks at the very end."""
    local_heads = parse_local_heads(args.local_heads)
    order = (args.onboard_order.split(",") if args.onboard_order else
             args.seq_order.split(",") if args.seq_order else sorted(courts, key=lambda c: -C[c]["n"]))
    assert set(order) == set(courts), f"--onboard_order must list exactly the courts {courts}"
    g, local0 = split_state(init, local_heads)
    client_local = {c: {k: v.clone() for k, v in local0.items()} for c in courts}
    ewc = EWC(args.ewc_lambda, group_filter=lambda n: not is_local(n, local_heads))
    replay = ReplayBuffer(args.replay_size) if args.replay_size > 0 else None
    rng = np.random.default_rng(args.seed)
    stage_rows, onboarded_at = [], {}
    for stage, c in enumerate(order, start=1):
        set_state(model, merge_state(g, client_local[c]))
        L = local_train_continual(model, C[c]["train"], args, dev, args.seq_epochs, C[c]["cw"], collate, rng,
                                  ewc=ewc, replay=replay, replay_exclude=c)
        g, client_local[c] = split_state(get_state(model), local_heads)
        onboarded_at[c] = stage
        if replay is not None: replay.add(c, C[c]["train"], rng)
        ewc.consolidate(model, C[c]["dl"]["train"], args, dev, C[c]["cw"])    # Fisher on SHARED params only
        for i in order[:stage]:                                              # re-check every onboarded court
            set_state(model, merge_state(g, client_local[i]))
            row = {"stage": stage, "joined_this_stage": c, "eval_court": i,
                   "rounds_since_onboarded": stage - onboarded_at[i]}
            for sp in ("val", "test"):                     # both logged; forgetting is SELECTED on val, REPORTED on test
                row.update({f"{sp}_{k}": v for k, v in evaluate(model, C[i]["dl"][sp], args, dev).items()})
            stage_rows.append(row)
        mean_now = np.nanmean([row["test_MAE_log"] for row in stage_rows if row["stage"] == stage])
        print(f"  stage {stage}/{len(order)}: onboarded {c} (train loss {L:.4f}) | "
              f"mean test MAE_log over {stage} onboarded court(s): {mean_now:.3f}", flush=True)
    hist = pd.DataFrame(stage_rows); hist.to_csv(os.path.join(out_dir, "forgetting.csv"), index=False)
    # forgetting_summary.csv reports TEST-set numbers (what goes in the report/paper, like every
    # other regime's final metrics); run_queue.py's "novel" plan instead compares candidate configs
    # using the VAL columns straight from forgetting.csv, so the choice of config never looks at test.
    final_stage = len(order); forget = []
    for c in order:
        right_after = hist[(hist.eval_court == c) & (hist.stage == onboarded_at[c])].iloc[0]
        final = hist[(hist.eval_court == c) & (hist.stage == final_stage)].iloc[0]
        forget.append({"court": c, "onboarded_stage": onboarded_at[c],
                       "MAE_log_right_after": right_after.test_MAE_log, "MAE_log_final": final.test_MAE_log,
                       "MAE_log_forgetting": final.test_MAE_log - right_after.test_MAE_log,      # + = got worse
                       "disp_F1_right_after": right_after.test_disp_F1, "disp_F1_final": final.test_disp_F1,
                       "disp_F1_forgetting": right_after.test_disp_F1 - final.test_disp_F1})     # + = got worse
    fdf = pd.DataFrame(forget); fdf.to_csv(os.path.join(out_dir, "forgetting_summary.csv"), index=False)
    bwt_mae, bwt_f1 = fdf.MAE_log_forgetting.mean(), fdf.disp_F1_forgetting.mean()
    print(f"\n  forgetting summary, test set (+ = worse at the end than right after onboarding; "
          f"0 for the last-onboarded court, which is never revisited):")
    print(fdf.round(3).to_string(index=False))
    print(f"  mean forgetting: MAE_log {bwt_mae:+.3f} | disp_F1 {bwt_f1:+.3f}")
    res, _ = eval_split(model, courts, C, "test", args, dev, g, client_local)
    for c in res:
        res[c].update(sel_val_loss=float("nan"), best_step=final_stage,
                      MAE_log_forgetting=float(fdf.set_index("court").loc[c, "MAE_log_forgetting"]),
                      disp_F1_forgetting=float(fdf.set_index("court").loc[c, "disp_F1_forgetting"]))
    return {"fed_continual": res}

# --------------------------------------------------------------------------- main
def main(argv=None):
    args = get_args(argv); seed_all(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = ""
    if args.regime == "fl":
        tag = f"_{args.algo}" + {"fedprox": f"_mu{args.mu}", "lossweight": f"_beta{args.beta}"}.get(args.algo, "")
        if args.algo in ("fedavgm", "fedadam", "fedyogi") and args.server_lr is not None: tag += f"_slr{args.server_lr}"
    if args.finetune_epochs > 0: tag += f"_ft{args.finetune_epochs}"
    if args.regime == "fed_continual":                        # ewc_lambda AND replay_size must both be in the
        tag += f"_ewc{args.ewc_lambda:g}"                      # name, or e.g. "naive" and "ewc_only" collide into
        tag += f"_replay{args.replay_size}" if args.replay_size > 0 else "_noreplay"  # one folder and silently skip
    if args.regime == "fed_continual" and args.onboard_order:   # different join order = different run
        tag += "_ord-" + "".join(c[:2] for c in args.onboard_order.split(","))
    if parse_local_heads(args.local_heads): tag += "_lh-" + args.local_heads.replace(",", "-")
    if parse_heads(args.mix_heads, "--mix_heads"): tag += "_mix-" + args.mix_heads.replace(",", "-")
    both = parse_local_heads(args.local_heads) & parse_heads(args.mix_heads, "--mix_heads")
    assert not both, f"head(s) {both} cannot be both --local_heads and --mix_heads"
    extra = f"_L{args.max_len}_lr{args.lr:g}" + ("_seg2" if args.segments == 2 else "_ht" if args.trunc == "headtail" else "") \
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
    elif args.regime == "fed_continual": res = run_fed_continual(model, courts, C, args, dev, init, out_dir, collate)
    else: res = run_seq_ewc(model, courts, C, args, dev, init)

    rows = [{"method": m, "court": c, "seed": args.seed, "model": args.model, "run": name,
             "max_len": args.max_len, "lr": args.lr, "trunc": "seg2" if args.segments == 2 else args.trunc, **met}
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
