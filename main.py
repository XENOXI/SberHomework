from torch.export import export
import torch
from torchax.export import exported_program_to_stablehlo
import torch.nn as nn
import logging
from pathlib import Path
import argparse
import sys
import ast
import json
import importlib

from transformers import AutoConfig
from analyzer import analyze_matmuls, MatmulRecord
from modelBundle import ModelBundle

logger = logging.getLogger(__name__)

def export_to_stablehlo(model: nn.Module, example_inputs: tuple[torch.Tensor]) -> tuple[str, dict]:
    try:
        output = model(*example_inputs)
    except Exception as e:
        logger.exception(f"Inputs are not fits to model, description: {e}")
        sys.exit(1)
    exported = export(model, tuple(example_inputs))

    weights, stablehlo = exported_program_to_stablehlo(exported)
    return stablehlo.mlir_module(), exported.state_dict

def log_report(records: list[MatmulRecord], model_name: str):
    sep = "=" * 72
    logger.info(sep)
    logger.info("Model: %s", model_name)
    logger.info("Vector-matrix multiplications found: %d", len(records))
    logger.info(sep)

    header = (
        f"{'#':>4}  "
        f"{'Input':>20}  "
        f"{'Weights':>20}  "
        f"{'Output':>20}"
    )
    logger.info(header)
    logger.info("=" * len(header))

    fmt_shape = lambda shape: "?" if shape is None else "x".join(str(d) for d in shape)

    for i, r in enumerate(records, 1):
        line = (
            f"{i:>4}  "
            f"{fmt_shape(r.input_shapes):>20}  "
            f"{fmt_shape(r.weight_shape):>20}  "
            f"{fmt_shape(r.output_shape):>20}"
        )
        logger.info(line)

    logger.info(sep)    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple utility for export pytorch model to StableHLO and get model gemm stats")
    parser.add_argument("--model", required=True, help="Path to model bundle")
    parser.add_argument("--out", required = True, type=Path, help="Path to out files")
    parser.add_argument("--device", default="cpu", choices= ["cpu", "cuda", "mps"], type=str, help="Device for model")
    
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force = True,
        stream=sys.stdout
    )

    device: str = args.device
    if not {"cuda": torch.cuda.is_available(), "mps": torch.mps.is_available(), "cpu": True}[device]:
        logger.warning("Device %s not available, falling back to cpu", device)
        device = "cpu"

    try:
        spec = importlib.util.spec_from_file_location("module", args.model)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        models = getattr(module, "MODELS")
    except Exception as e:
        logging.exception(f"Wrong file gotted in {args.model}: {e}")
        sys.exit(1)
    
    if not isinstance(models, list):
        logging.exception(f"Expected MODELS to be list, gotted {models}")
        sys.exit(1)
    
    out_path : Path = args.out
    out_path.mkdir(parents=True, exist_ok=True)

    for mod in models:
        if not isinstance(mod, dict):
            logging.warning(f"Expected model in MODELS to be dict, gotted {mod}")
            sys.exit(1)
        try:
            mod = ModelBundle(**mod)
        except Exception as e:
            logging.warning(f"Expected model in MODELS to be dict structured like ModelBundle, gotted {mod}: {e}")
            sys.exit(1)

        if not mod.check():
            logging.exception(f"Incorrect ModelBundle")
            sys.exit(1)

        model = mod.model
        try:
            if mod.weights:
                model.load_state_dict(torch.load(mod.weights, weights_only=True))
        except Exception as e:
            logger.exception(f"Model weights import failed: {e}")
            
        model = model.to(device).eval()
        
        name = mod.name

        hlo, params = export_to_stablehlo(model, mod.example_inputs)

        with open(out_path / f"{name}_hlo.txt", mode = "w") as f:
            f.write(hlo)
        logger.info("Model StableHLO saved to: %s", out_path / f"{name}_hlo.txt")
        
        with open(out_path / f"{name}_data.json", mode = "w") as f:
            serializable_params = {k: v.detach().cpu().tolist() for k, v in params.items()}
            json.dump(serializable_params, f)
        
        logger.info("Model params saved to: %s", out_path / f"{name}_data.json")

        report = analyze_matmuls(hlo)

        log_report(report, name)
    
