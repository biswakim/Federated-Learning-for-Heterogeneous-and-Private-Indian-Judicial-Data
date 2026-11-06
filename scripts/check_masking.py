"""
Re-applies the exact masking used in training (fl_judicial.mask_text) and checks what is left.

    python scripts/check_masking.py --data data/judgments_labelled.csv --out masking_check

Writes:
  masking_check/masking_check_sample.csv  20 cases per court, original vs masked, first and last 1,200 chars
  masking_check/masked_texts_all.csv      every masked text (for searching)
and prints how often dates, numbers and outcome-like words survive masking, and whether the
surviving outcome-like words shift the disposal label distribution.
"""
import argparse, os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fl_judicial import mask_text

p = argparse.ArgumentParser(); p.add_argument("--data", required=True); p.add_argument("--out", default="masking_check")
p.add_argument("--per_court", type=int, default=20); a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
j = pd.read_csv(a.data, low_memory=False)
if "garbled" in j: j = j[~j.garbled.astype(bool)]
j = j.reset_index(drop=True); j["masked"] = j.input_text.fillna("").astype(str).map(mask_text)
pats = {"4-digit year": r"\b(?:19|20)\d{2}\b", "number with 2+ digits": r"\b\d{2,}\b",
        "ordinal (12th)": r"\b\d{1,2}(?:st|nd|rd|th)\b",
        "month name": r"\b(?:january|february|march|april|june|july|august|september|october|november|december)\b",
        "quashed": r"\bquash(?:ed|ing)?\b", "granted": r"\bgranted\b", "set aside": r"\bset aside\b",
        "allowing/dismissing/disposing": r"\b(?:allowing|dismissing|disposing|withdrawing)\b"}
has = {k: j.masked.str.contains(v, case=False, regex=True) for k, v in pats.items()}
print(pd.DataFrame({"still present in % of cases": {k: round(100 * h.mean(), 1) for k, h in has.items()}}).to_string())
print("\nmask tokens per case:", {t: round(j.masked.str.count(rf"\b{t}\b").mean(), 1) for t in ["DATE", "YEAR", "NUM", "OUTCOME"]})
lab = j.outcome_from_text != "UNCLEAR"
print("\ndisposal mix, all cases:", j[lab].outcome_from_text.value_counts(normalize=True).round(2).to_dict())
for k in ["quashed", "granted", "set aside", "allowing/dismissing/disposing"]:
    m = lab & has[k]
    print(f"  with '{k}' (n={m.sum()}):", j[m].outcome_from_text.value_counts(normalize=True).round(2).to_dict())
s = pd.concat([j[j.court == c].sample(min(a.per_court, (j.court == c).sum()), random_state=0) for c in sorted(j.court.unique())])
pd.DataFrame({"cnr": s.cnr, "court": s.court, "disposal_label": s.outcome_from_text, "duration_days": s.duration_days,
              "original_START": s.input_text.str[:1200], "masked_START": s.masked.str[:1200],
              "original_END": s.input_text.str[-1200:], "masked_END": s.masked.str[-1200:]}) \
  .to_csv(os.path.join(a.out, "masking_check_sample.csv"), index=False)
j[["cnr", "court", "split", "outcome_from_text", "duration_days", "case_type_group", "masked"]] \
  .rename(columns={"masked": "masked_text"}).to_csv(os.path.join(a.out, "masked_texts_all.csv"), index=False)
print(f"\nwrote {a.out}/masking_check_sample.csv and {a.out}/masked_texts_all.csv")
