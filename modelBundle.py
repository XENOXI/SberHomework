from dataclasses import field, dataclass
from torch import nn
import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelBundle:
    """A model together with concrete example inputs for tracing."""
    model: nn.Module
    example_inputs: tuple[torch.Tensor, ...]
    name: str
    weights: Path | None = field(default= None)

    def check(self) -> bool:
        errors = self._validate()
        if errors:
            for e in errors:
                logger.exception(e)
            return False
        return True

    def _validate(self) -> list[str]:
        errors = []

        if not isinstance(self.model, nn.Module):
            errors.append(f"'model' must be nn.Module, got {type(self.model).__name__}")

        if not isinstance(self.name, str) or not self.name:
            errors.append(f"'name' must be a non-empty str, got {self.name!r}")

        if not isinstance(self.example_inputs, tuple):
            errors.append(f"'example_inputs' must be a tuple, got {type(self.example_inputs).__name__}")
        elif not self.example_inputs:
            errors.append("'example_inputs' must not be empty")
        else:
            for i, inp in enumerate(self.example_inputs):
                if not isinstance(inp, torch.Tensor):
                    errors.append(f"'example_inputs[{i}]' must be Tensor, got {type(inp).__name__}")

        if self.weights is not None:
            if not isinstance(self.weights, Path):
                errors.append(f"'weights' must be Path or None, got {type(self.weights).__name__}")
            elif not self.weights.exists():
                errors.append(f"'weights' file not found: {self.weights}")
            elif not self.weights.is_file():
                errors.append(f"'weights' is not a file: {self.weights}")

        return errors