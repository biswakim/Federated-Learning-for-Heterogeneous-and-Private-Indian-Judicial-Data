"""
CPU smoke test: runs fl_judicial.py end to end with a tiny randomly initialised encoder,
so the pipeline can be checked without a GPU or downloading InLegalBERT / RoBERTa.
Numbers it prints are meaningless; it only checks that nothing crashes.

    python tests/smoke_test.py --data data/judgments_labelled.csv --regime fl --algo fedavg
    python tests/smoke_test.py --data data/judgments_labelled.csv --regime fl --segments 2 --mix_heads dur,typ,disp
    python tests/smoke_test.py --data data/judgments_labelled.csv --regime fed_continual --replay_size 8

Any fl_judicial.py flag can be passed; small defaults (--limit, --max_len, --rounds) are added
unless you override them.
"""
import os, sys
import pandas as pd
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaModel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import fl_judicial as F

argv = sys.argv[1:]
data = argv[argv.index("--data") + 1]
df = pd.read_csv(data, nrows=3000, low_memory=False)
tk = Tokenizer(models.WordLevel(unk_token="[UNK]")); tk.pre_tokenizer = pre_tokenizers.Whitespace()
tk.train_from_iterator(df.input_text.fillna("").map(F.mask_text).tolist(),
                       trainers.WordLevelTrainer(vocab_size=3000, special_tokens=["[PAD]", "[UNK]"]))
TOK = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="[PAD]", unk_token="[UNK]")
CFG = RobertaConfig(vocab_size=3000, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                    intermediate_size=64, max_position_embeddings=600, pad_token_id=0)
F.AutoTokenizer.from_pretrained = staticmethod(lambda *a, **k: TOK)
F.AutoModel.from_pretrained = staticmethod(lambda *a, **k: RobertaModel(CFG))

defaults = {"--limit": "40", "--max_len": "64", "--rounds": "2", "--local_epochs": "0.5", "--epochs": "1",
            "--seq_epochs": "0.5", "--batch": "8", "--out_dir": "runs_smoke"}
for k, v in defaults.items():
    if k not in argv: argv += [k, v]
F.main(argv + ["--force"])
print("\nsmoke test passed")
