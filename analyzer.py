from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp
from mlir.ir import Context, Module, RankedTensorType, Operation
from mlir.dialects import stablehlo

context = Context()
stablehlo.register_dialect(context)

logger = logging.getLogger(__name__)


ALL_MATMUL_OPS = {
    torch.matmul,
    torch.mm,
    torch.bmm,
    torch.einsum,  
    torch.nn.functional.linear,        
    torch._C._nn.linear,   
    torch.linalg.matmul,
    torch.bilinear,
    '@',
    torch.ops.aten.mm.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten.bmm.default,
    torch.ops.aten.matmul.default
}

@dataclass
class MatmulRecord:
    input_shapes: tuple[int] | None = None
    weight_shape: tuple[int] | None = None
    output_shape: tuple[int] | None = None

def _tensor_meta_to_shape(meta) -> tuple[int] | None:
    if meta is None:
        return None
    if isinstance(meta, torch.Tensor):
        return tuple(meta.shape)
    if hasattr(meta, "shape"):
        return tuple(meta.shape)
    return None


def _arg_shape(arg: torch.fx.Node) -> tuple[int] | None:
    if isinstance(arg, torch.fx.Node):
        return _tensor_meta_to_shape(arg.meta.get("tensor_meta") or arg.meta.get("val"))
    return None


def analyze_matmuls(
    stablehlo_text
) -> list[MatmulRecord]:
    
    logger.info("Парсинг модели")
    module = Module.parse(stablehlo_text,context)
    result = []
    def walk_ops(op):
        regions = op.regions if hasattr(op, "regions") else []
        for region in regions:
            for block in region:
                for child_op in block:
                    yield child_op
                    yield from walk_ops(child_op)

    for op in walk_ops(module.operation):
        if op.name != "stablehlo.dot_general":
            continue

        lhs_type = RankedTensorType(op.operands[0].type)
        rhs_type = RankedTensorType(op.operands[1].type)
        out_type  = RankedTensorType(op.results[0].type)
        result.append(MatmulRecord(tuple(lhs_type.shape), tuple(rhs_type.shape), tuple(out_type.shape)))

    return result




