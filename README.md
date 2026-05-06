# SberHomework - PyTorch -> StableHLO GEMM Analyser

[![Tests](https://github.com/XENOXI/SberHomework/actions/workflows/test.yml/badge.svg)](https://github.com/XENOXI/SberHomework/actions/workflows/test.yml)

A command-line utility that exports a PyTorch model to StableHLO and reports every matrix-multiplication (`dot_general`) operation found in the resulting IR - its input, weight, and output shapes.

---

## How it works

1. **Export** - the model is traced with `torch.export` and then converted to StableHLO text via `torchax`.
2. **Analyse** - the StableHLO module is parsed with the MLIR Python bindings; every `stablehlo.dot_general` op is recorded.
3. **Report** - a formatted table is printed to stdout showing the shape of each matmul's operands and result.
4. **Artefacts** - the raw HLO text and the model's state-dict (as JSON) are written to the output directory.

---

## Requirements

| Tool | Version |
|------|---------|
| Python | 3.10.x |
| uv | latest |

All Python dependencies (PyTorch, JAX, torch-xla, Transformers, MLIR, StableHLO …) are declared in `pyproject.toml` and installed automatically.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/XENOXI/SberHomework
cd SberHomework

# 2. Create a virtual environment and install dependencies
uv sync
```

> `mlir` and `stablehlo` wheels are fetched from custom indices
> ([mlir-wheels](https://github.com/makslevental/mlir-wheels) /
> [stablehlo dev-wheels](https://github.com/openxla/stablehlo)) configured in
> `pyproject.toml` — no extra steps required.

---

## Usage

```bash
uv run main.py --model <bundle_file.py> --out <output_dir> [--device cpu|cuda|mps]
```

| Argument | Default | Description |
|----------|----------|-------------|
| `--model` | - | Path to a Python file that exposes a `MODELS` list |
| `--out` | - | Directory where artefacts are written (created if absent) |
| `--device` | `cpu` | Execution device (`cpu`, `cuda`, `mps`) |

### Example

```bash
uv run main.py --model my_models.py --out ./results --device cuda
```

---

## Model bundle file

The `--model` argument must point to a plain Python file that contains a module-level list called `MODELS`. Each element is a dict whose keys match the `ModelBundle` dataclass:

```python
# my_models.py
import torch
import torch.nn as nn

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

MODELS = [
    {
        "name": "tiny_mlp",
        "model": TinyMLP(),
        "example_inputs": (torch.randn(1, 128),),
        # "weights": Path("tiny_mlp.pt"),   # optional
    }
]
```

### `ModelBundle` fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Identifier used as the artefact filename prefix |
| `model` | `nn.Module` | The PyTorch model to export |
| `example_inputs` | `tuple[Tensor, ...]` | Representative inputs used for tracing |
| `weights` | `Path \| None` | Path to a `.pt` checkpoint to load before export |

---

## Output artefacts

For each model in `MODELS` the tool writes two files to `<out>/`:

| File | Description |
|------|-------------|
| `<name>_hlo.txt` | StableHLO text representation of the exported model |
| `<name>_data.json` | Model state-dict serialised as JSON (tensors → nested lists) |

A summary table is also printed to stdout, for example:

```
========================================================================
Model: tiny_mlp
Vector-matrix multiplications found: 2
========================================================================
   #                 Input                Weights                 Output
========================================================================
   1              1x128                  128x64                   1x64
   2               1x64                   64x10                   1x10
========================================================================
```

---

## Development

```bash
# Install dev dependencies (pytest)
uv sync --group dev

# Run tests
uv run pytest
```
