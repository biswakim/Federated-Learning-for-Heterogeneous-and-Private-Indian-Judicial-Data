"""
Compare runs:  python summarize_results.py runs/all_results.csv
Rows = configuration (run name without the seed) x method. Columns = mean ± std over seeds of the
court-averaged test metrics, the worst court, and the validation loss used for model selection.
Sort/choose by sel_val_loss (validation), not by test numbers.
"""
import sys, pandas as pd
df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "runs/all_results.csv")
df["config"] = df.run.str.replace(r"_s\d+$", "", regex=True)
if "sel_val_loss" not in df: df["sel_val_loss"] = float("nan")
per_seed = df.groupby(["config", "method", "seed"]).agg(
    val_loss=("sel_val_loss", "mean"), MAE_log=("MAE_log", "mean"), Spearman=("Spearman", "mean"),
    F1_4=("F1_4", "mean"), disp_F1=("disp_F1", "mean"), type_F1=("type_F1", "mean"),
    worst_MAE_log=("MAE_log", "max"), worst_disp_F1=("disp_F1", "min"))
g = per_seed.groupby(["config", "method"])
mean, std, n = g.mean(), g.std(), g.size().rename("seeds")
out = mean.round(3).astype(str) + " ± " + std.round(3).fillna(0).astype(str)
out = out.join(n); out["_v"] = mean.val_loss
out = out.sort_values("_v").drop(columns="_v")
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 70)
print(out.to_string()); out.to_csv("comparison.csv"); print("\nsaved comparison.csv (sorted by validation loss)")
