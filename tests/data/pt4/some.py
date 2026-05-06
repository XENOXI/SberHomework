import torch
import torch.nn as nn
from pathlib import Path
from transformers import BertModel

hf = BertModel.from_pretrained("prajjwal1/bert-tiny")

MODELS = [{
    "model": hf,
    "example_inputs": (torch.randint(0, 30522, (1, 8)),),
    "name": "some",
    "weights": (Path(__file__).parent / "some.pth").resolve()}]