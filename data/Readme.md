# Data

The data are **not** included in this repository. Place the labelled file here as
`data/judgments_labelled.csv`.

**Source.** Indian High Court Judgments dataset (Dattam Labs), available on the AWS Open Data
Registry under CC-BY-4.0. Please credit the dataset if you use or redistribute derived files.

**What the code expects.** One row per judgment, with at least these columns:

| Column | Meaning |
| --- | --- |
| `cnr` | Unique case number (CNR) |
| `court` | One of `bombay`, `delhi`, `karnataka`, `kerala`, `madhya_pradesh` (one federated client each) |
| `split` | `train`, `val` or `test` (temporal 70/15/15 split within each court) |
| `input_text` | Judgment text with the operative (final order) part removed |
| `duration_days` | Days from filing to decision (Task 1 target) |
| `outcome_from_text` | `ALLOWED`, `DISMISSED`, `DISPOSED`, `WITHDRAWN` or `UNCLEAR` (Task 4 target; `UNCLEAR` is ignored for that task) |
| `case_type_group` | Case-type family (10 families, derived from `case_type_raw`) |
| `case_type_raw` | Court-specific case-type code from the metadata (e.g. `WP`, `BA`, `CAF`) |
| `garbled` | `True` for judgments whose PDF text extraction failed (dropped) |

The version used in the experiments has 12,500 judgments (2,500 per court), all decided in 2024.
Disposal nature and case-type families are rule-based labels.

**Masking.** At load time `fl_judicial.py` masks dates, four-digit years, multi-digit numbers and
outcome words (all cases were decided in 2024, so dates would leak the duration). Check it with
`python scripts/check_masking.py --data data/judgments_labelled.csv`.
