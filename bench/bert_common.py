"""Stage 3b: shared BERT/DistilBERT helpers (SST-2). Loads local safetensors only; no remote code, no hub client."""
import json
from pathlib import Path
import numpy as np
from common import DATA, TEMP, RESULTS, SEED

BERT = DATA / "bert"
MODELS = {  # name -> folder, input names, card accuracy (SST-2 validation, %)
    "A": {"dir": BERT / "A_distilbert", "inputs": ["input_ids", "attention_mask"], "card_acc": 91.06,
          "hf_id": "distilbert/distilbert-base-uncased-finetuned-sst-2-english"},
    "B": {"dir": BERT / "B_bert_mini", "inputs": ["input_ids", "attention_mask", "token_type_ids"], "card_acc": 84.74,
          "hf_id": "M-FAC/bert-mini-finetuned-sst2"},
}
N_CALIB = 300
ACC_OUT = RESULTS / "accuracy_bert"


def read_split(split):
    import pandas as pd
    df = pd.read_parquet(BERT / "sst2" / f"{split}-00000-of-00001.parquet")
    return df["sentence"].tolist(), df["label"].to_numpy(dtype=np.int64)


def calib_sentences():
    """300 random SST-2 TRAIN sentences (seed 0); never validation."""
    s, y = read_split("train")
    idx = np.random.default_rng(SEED).choice(len(s), N_CALIB, replace=False)
    return [s[i] for i in idx]


def tokenizer(name):
    from transformers import AutoTokenizer   # BertTokenizerFast(vocab_file=...) ignores the file in transformers 5.x (all [UNK])
    return AutoTokenizer.from_pretrained(str(MODELS[name]["dir"]), local_files_only=True, trust_remote_code=False)


def encode(name, sentences, seq):
    """Fixed-length encoding (pad to seq, truncate); returns dict of int64 arrays (only the model's inputs) + n_truncated."""
    tok = tokenizer(name)
    full = tok(sentences, truncation=False)["input_ids"]
    n_trunc = int(sum(len(x) > seq for x in full))
    enc = tok(sentences, padding="max_length", truncation=True, max_length=seq, return_tensors="np")
    return {k: enc[k].astype(np.int64) for k in MODELS[name]["inputs"]}, n_trunc


def load_model(name):
    """FP32 model from local safetensors. attn_implementation left at the transformers default (recorded by caller)."""
    from transformers import AutoModelForSequenceClassification
    m = AutoModelForSequenceClassification.from_pretrained(str(MODELS[name]["dir"]), use_safetensors=True,
                                                           trust_remote_code=False, local_files_only=True)
    return m.eval()


class LogitsOnly:
    """torch.nn.Module wrapper returning logits only (tuple-free) so export/quantization see a plain tensor output."""
    def __new__(cls, hf_model, input_names):
        import torch
        class W(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.m = hf_model; self.names = input_names
            def forward(self, *args):
                return self.m(**dict(zip(self.names, args)), return_dict=False)[0]
        return W().eval()


def onnx_fp32_path(name):
    return TEMP / f"bert_{name}_fp32.onnx"


def make_inputs(name, seq, batch):
    """Seeded SST-2-like random-free input for timing: first `batch` validation sentences encoded at `seq`."""
    s, _ = read_split("validation")
    enc, _ = encode(name, s[:batch], seq)
    return enc
