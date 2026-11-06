"""
Forgetting curves for the federated continual-onboarding runs (--regime fed_continual).

    python scripts/plot_forgetting.py --runs runs --metric test_disp_F1 --out forgetting_curves.png
    python scripts/plot_forgetting.py --runs runs --metric test_MAE_log --out forgetting_curves_duration.png

One panel per run: each court's test score after every new court joins.
"""
import argparse, glob, json, os
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(); p.add_argument("--runs", default="runs")
p.add_argument("--metric", default="test_disp_F1"); p.add_argument("--out", default="forgetting_curves.png"); a = p.parse_args()
runs = sorted(d for d in glob.glob(os.path.join(a.runs, "fed_continual_*")) if os.path.exists(os.path.join(d, "forgetting.csv")))
assert runs, f"no fed_continual runs with forgetting.csv under {a.runs}"
names = {(0, 0): "Naive", (100, 0): "EWC", (0, 16): "Replay", (100, 16): "EWC + replay"}
fig, axes = plt.subplots(1, len(runs), figsize=(3.2 * len(runs), 2.8), sharey=True, squeeze=False)
for ax, d in zip(axes[0], runs):
    cfg = json.load(open(os.path.join(d, "config.json"))); h = pd.read_csv(os.path.join(d, "forgetting.csv"))
    title = names.get((int(cfg["ewc_lambda"]), cfg["replay_size"]), os.path.basename(d))
    order = cfg.get("onboard_order")
    if order: title += " (MP last)" if order.endswith("madhya_pradesh") else " (other order)"
    for court, g in h.groupby("eval_court"):
        ax.plot(g.stage, g[a.metric], marker="o", ms=3, label=court.replace("_", " ").title())
    ax.set_title(title, fontsize=9); ax.set_xlabel("courts joined"); ax.set_xticks(range(1, int(h.stage.max()) + 1))
axes[0][0].set_ylabel(a.metric.replace("test_", "test ")); axes[0][-1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(a.out, dpi=200, bbox_inches="tight"); print("saved", a.out)
