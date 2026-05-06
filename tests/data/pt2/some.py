import torch
import torch.nn as nn
from pathlib import Path

MODELS = [{
    "model": nn.Sequential(nn.Linear(20,30), nn.ReLU()),
    "example_inputs": (torch.zeros(1,20), ),
    "name": "some",
    "weights": (Path(__file__).parent / "some.pth").resolve()}]