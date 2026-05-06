import torch
import torch.nn as nn
from pathlib import Path

MODELS = [{
    "model": nn.Linear(10,20),
    "example_inputs": (torch.zeros(1,10), ),
    "name": "some",
    "weights": (Path(__file__).parent / "some.pth").resolve()}]