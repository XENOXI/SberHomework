import torch
import torch.nn as nn
from pathlib import Path
from transformers import BertModel

hf = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")
batch: int = 1
seq_len: int = 32
vocab_size = hf.config.vocab_size

MODELS = [{
    "model": hf,
    "example_inputs": (torch.randint(0, vocab_size, (batch, seq_len)), torch.ones(batch, seq_len, dtype=torch.long) ),
    "name": "some",
    "weights": (Path(__file__).parent / "some.pth").resolve()}]