"""
NER branch features (proposal, Figure 2): counts of OpenNyAI legal entity types per case.

Install on Kaggle first (check OpenNyAI's README for the current versions/command):
    pip install -q "spacy==3.2.4"
    pip install -q https://huggingface.co/opennyaiorg/en_legal_ner_trf/resolve/main/en_legal_ner_trf-any-py3-none-any.whl
If the spaCy pin conflicts with the Kaggle image, run this script in a separate notebook
and only upload the resulting CSV; fl_judicial.py itself does not need spaCy.

Usage:
    python ner_features.py judgments_labelled.csv ner_features.csv [max_chars]
Output columns: cnr, ner_COURT, ner_PETITIONER, ... (raw counts; fl_judicial.py applies log1p).

Leakage notes
  * Runs on input_text (operative ending already removed). DATE and CASE_NUMBER entities are
    not counted, because dates/case numbers reveal the duration in this 2024-only dataset.
  * Only entity *type counts* are kept, never the entity strings, so party names do not
    enter the model.
  * Entity counting is done separately at each court in a real deployment; here it is a
    per-document transformation, so no information is pooled across clients.
"""
import sys, pandas as pd, spacy
from collections import Counter
TYPES = ["COURT", "PETITIONER", "RESPONDENT", "JUDGE", "LAWYER", "ORG", "GPE",
         "STATUTE", "PROVISION", "PRECEDENT", "WITNESS", "OTHER_PERSON"]
src, dst = sys.argv[1], sys.argv[2]
max_chars = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
spacy.prefer_gpu()
nlp = spacy.load("en_legal_ner_trf")
df = pd.read_csv(src, usecols=["cnr", "input_text"], low_memory=False)
texts = df.input_text.fillna("").astype(str).str.slice(0, max_chars).tolist()
rows = []
for i, doc in enumerate(nlp.pipe(texts, batch_size=8)):
    c = Counter(e.label_ for e in doc.ents)
    rows.append({"cnr": df.cnr.iloc[i], **{f"ner_{t}": c.get(t, 0) for t in TYPES}})
    if (i + 1) % 500 == 0: print(f"{i+1}/{len(texts)}", flush=True)
pd.DataFrame(rows).to_csv(dst, index=False)
print("saved", dst)
