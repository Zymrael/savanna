"""LoRA (Low-Rank Adaptation) implementation for Savanna's parallel linear layers.

Supports ColumnParallelLinear, RowParallelLinear, and TELinear (TransformerEngine)
with correct tensor-parallel communication patterns.
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from savanna import print_rank_0
from savanna.mpu.mappings import (
    gather_from_sequence_parallel_region,
    reduce_from_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)

logger = logging.getLogger(__name__)


class LoRAColumnParallelLinear(nn.Module):
    """LoRA wrapper for ColumnParallelLinear.

    lora_A: [r, input_size] — full (not partitioned), receives full input
    lora_B: [output_size_per_partition, r] — partitioned on dim 0, like base weight
    """

    def __init__(self, base_layer, r, alpha, dropout=0.0):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        input_size = base_layer.input_size
        output_size_per_partition = base_layer.output_size_per_partition
        self.sequence_parallel = getattr(base_layer, "sequence_parallel", False)
        self.seq_dim = getattr(base_layer, "seq_dim", 0)

        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        self.lora_A = nn.Parameter(torch.empty(r, input_size, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(output_size_per_partition, r, device=device, dtype=dtype))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_A._is_lora_param = True
        self.lora_B._is_lora_param = True

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, input_):
        base_output, base_bias = self.base_layer(input_)

        # LoRA path: mirror the base layer's input gathering for sequence parallelism
        if self.sequence_parallel:
            lora_input = gather_from_sequence_parallel_region(input_, seq_dim=self.seq_dim)
        else:
            lora_input = input_

        lora_out = F.linear(F.linear(self.lora_dropout(lora_input), self.lora_A), self.lora_B) * self.scaling
        return base_output + lora_out, base_bias


class LoRARowParallelLinear(nn.Module):
    """LoRA wrapper for RowParallelLinear.

    lora_A: [r, input_size_per_partition] — partitioned on dim 1, like base weight
    lora_B: [output_size, r] — full (not partitioned), produces full output
    """

    def __init__(self, base_layer, r, alpha, dropout=0.0):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        input_size_per_partition = base_layer.input_size_per_partition
        output_size = base_layer.output_size
        self.sequence_parallel = getattr(base_layer, "sequence_parallel", False)
        self.parallel_output = getattr(base_layer, "parallel_output", False)
        self.seq_dim = getattr(base_layer, "seq_dim", 0)

        if self.parallel_output and input_size_per_partition != base_layer.input_size:
            logger.warning(
                "LoRARowParallelLinear with parallel_output=True and TP > 1: "
                "the LoRA delta skips the all-reduce but applies full lora_B, "
                "which is mathematically incorrect. Use parallel_output=False."
            )

        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        self.lora_A = nn.Parameter(torch.empty(r, input_size_per_partition, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(output_size, r, device=device, dtype=dtype))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_A._is_lora_param = True
        self.lora_B._is_lora_param = True

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, input_):
        base_output, base_bias = self.base_layer(input_)

        # LoRA path: input_ is already partitioned across TP ranks (same as base)
        if self.base_layer.input_is_parallel:
            input_parallel = input_
        else:
            # Base layer scatters internally; we need the partitioned version
            from savanna.mpu.mappings import scatter_to_model_parallel_region

            input_parallel = scatter_to_model_parallel_region(input_)

        # lora_A produces partial sum of shape [..., r]
        lora_intermediate = F.linear(self.lora_dropout(input_parallel), self.lora_A)

        # Reduce across TP ranks (matching base layer's communication pattern)
        if self.sequence_parallel and not self.parallel_output:
            lora_intermediate = reduce_scatter_to_sequence_parallel_region(
                lora_intermediate, seq_dim=self.seq_dim
            )
        elif not self.parallel_output:
            lora_intermediate = reduce_from_model_parallel_region(lora_intermediate)

        lora_out = F.linear(lora_intermediate, self.lora_B) * self.scaling
        return base_output + lora_out, base_bias


class LoRATELinear(nn.Module):
    """LoRA wrapper for TELinear (TransformerEngine).

    Determines column vs row parallelism from the TELinear subclass type,
    then applies LoRA delta in compute dtype on top of FP8 base output.
    """

    def __init__(self, base_layer, r, alpha, dropout=0.0, parallel_mode=None):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.parallel_mode = parallel_mode

        config = base_layer.config
        self.sequence_parallel = getattr(config, "sequence_parallel", False)

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.seq_dim = getattr(config, "seq_dim", 0)

        # Determine LoRA shapes based on parallel mode
        if parallel_mode == "column":
            # Like ColumnParallelLinear: A is full input, B is partitioned output
            lora_a_size = in_features
            lora_b_size = out_features  # already partitioned by TE
        elif parallel_mode == "row":
            # Like RowParallelLinear: A is partitioned input, B is full output
            lora_a_size = in_features  # already partitioned by TE
            lora_b_size = out_features
        else:
            raise ValueError(f"Cannot determine parallel_mode for TELinear: {parallel_mode}")

        # Use bf16 for LoRA params (compute dtype) regardless of FP8 base
        dtype = torch.bfloat16
        device = next(base_layer.parameters()).device

        self.lora_A = nn.Parameter(torch.empty(r, lora_a_size, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(lora_b_size, r, device=device, dtype=dtype))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_A._is_lora_param = True
        self.lora_B._is_lora_param = True

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, input_):
        base_output, base_bias = self.base_layer(input_)

        lora_input = input_.to(self.lora_A.dtype)

        if self.parallel_mode == "column":
            # Column: TE handles sequence parallel gather internally
            # but we need to do it for our LoRA path
            if self.sequence_parallel:
                lora_input = gather_from_sequence_parallel_region(lora_input, seq_dim=self.seq_dim)
            lora_out = (
                F.linear(F.linear(self.lora_dropout(lora_input), self.lora_A), self.lora_B) * self.scaling
            )
        elif self.parallel_mode == "row":
            # Row: input is already partitioned, reduce after A
            lora_intermediate = F.linear(self.lora_dropout(lora_input), self.lora_A)
            if self.sequence_parallel:
                lora_intermediate = reduce_scatter_to_sequence_parallel_region(
                    lora_intermediate, seq_dim=self.seq_dim
                )
            else:
                lora_intermediate = reduce_from_model_parallel_region(lora_intermediate)
            lora_out = F.linear(lora_intermediate, self.lora_B) * self.scaling

        return base_output + lora_out.to(base_output.dtype), base_bias


def _get_parent_and_attr(model, dotted_name):
    """Split 'a.b.c' into (model.a.b, 'c') for setattr replacement."""
    parts = dotted_name.rsplit(".", 1)
    if len(parts) == 1:
        return model, parts[0]
    parent_name, attr_name = parts
    parent = model
    for p in parent_name.split("."):
        parent = getattr(parent, p)
    return parent, attr_name


def _detect_te_parallel_mode(module):
    """Detect whether a TELinear is column or row parallel."""
    try:
        from savanna.model.tengine import TEColumnParallelLinear, TERowParallelLinear

        if isinstance(module, TEColumnParallelLinear):
            return "column"
        if isinstance(module, TERowParallelLinear):
            return "row"
    except ImportError:
        pass

    # Fallback: check parallel_mode attribute (set by TE's Linear.__init__)
    parallel_mode = getattr(module, "parallel_mode", None)
    if parallel_mode in ("column", "row"):
        return parallel_mode

    return None


def apply_lora_to_model(model, global_config):
    """Apply LoRA adapters to target modules in the model.

    Walks model.named_modules(), matches leaf names against target_modules list,
    and wraps matching layers with the appropriate LoRA class.
    """
    from savanna.mpu.layers import ColumnParallelLinear, RowParallelLinear

    lora_cfg = global_config.lora
    r = lora_cfg.get("r", 8)
    alpha = lora_cfg.get("alpha", 16.0)
    dropout = lora_cfg.get("dropout", 0.0)
    target_modules = lora_cfg.get("target_modules", [])

    if not target_modules:
        logger.warning("LoRA enabled but target_modules is empty — no modules will be wrapped.")
        return

    # Try to import TE classes (may not be available)
    te_linear_cls = None
    try:
        from savanna.model.tengine import TELinear

        te_linear_cls = TELinear
    except ImportError:
        pass

    wrapped_count = 0
    wrapped_names = []

    # Collect all (name, module) pairs first to avoid modification during iteration
    named_modules = list(model.named_modules())

    for name, module in named_modules:
        # Check if the leaf name matches any target pattern
        leaf_name = name.rsplit(".", 1)[-1] if "." in name else name
        if leaf_name not in target_modules:
            continue

        parent, attr_name = _get_parent_and_attr(model, name)

        if isinstance(module, ColumnParallelLinear):
            lora_module = LoRAColumnParallelLinear(module, r=r, alpha=alpha, dropout=dropout)
            setattr(parent, attr_name, lora_module)
            wrapped_count += 1
            wrapped_names.append(name)

        elif isinstance(module, RowParallelLinear):
            lora_module = LoRARowParallelLinear(module, r=r, alpha=alpha, dropout=dropout)
            setattr(parent, attr_name, lora_module)
            wrapped_count += 1
            wrapped_names.append(name)

        elif te_linear_cls is not None and isinstance(module, te_linear_cls):
            parallel_mode = _detect_te_parallel_mode(module)
            if parallel_mode is None:
                logger.warning(f"Skipping TELinear '{name}': cannot determine parallel_mode.")
                continue
            lora_module = LoRATELinear(
                module, r=r, alpha=alpha, dropout=dropout, parallel_mode=parallel_mode
            )
            setattr(parent, attr_name, lora_module)
            wrapped_count += 1
            wrapped_names.append(name)

        else:
            logger.debug(f"Target module '{name}' matched but is type {type(module).__name__}, skipping.")

    if wrapped_count == 0:
        logger.warning(
            f"LoRA target_modules={target_modules} matched zero modules. "
            "Check that target names match leaf module names in the model."
        )
    else:
        print_rank_0(f"LoRA: wrapped {wrapped_count} modules: {wrapped_names}")


def get_lora_state_dict(model):
    """Extract only LoRA parameters from the model state dict."""
    return {k: v.clone() for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k}


def load_lora_state_dict(model, state_dict):
    """Load LoRA-only state dict into model (non-strict)."""
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # Filter out non-LoRA missing keys (expected since we only load LoRA params)
    lora_missing = [k for k in missing if "lora_A" in k or "lora_B" in k]
    if lora_missing:
        logger.warning(f"Missing LoRA keys when loading: {lora_missing}")
    if unexpected:
        logger.warning(f"Unexpected keys when loading LoRA state dict: {unexpected}")


def merge_lora_weights(model):
    """Merge LoRA weights into base layer weights for inference/merged checkpointing.

    After merging, the LoRA delta (B @ A * scaling) is added to base_layer.weight,
    and a flag is set to avoid double-merging.  The original weight is saved so that
    unmerge can restore it exactly (avoiding bf16 precision loss from subtract).
    """
    for name, module in model.named_modules():
        if isinstance(module, (LoRAColumnParallelLinear, LoRARowParallelLinear, LoRATELinear)):
            if getattr(module, "_lora_merged", False):
                continue
            with torch.no_grad():
                module._original_weight = module.base_layer.weight.data.clone()
                delta = (module.lora_B @ module.lora_A) * module.scaling
                module.base_layer.weight.data += delta.to(module.base_layer.weight.dtype)
            module._lora_merged = True


def unmerge_lora_weights(model):
    """Reverse the merge operation by restoring the saved original weight."""
    for name, module in model.named_modules():
        if isinstance(module, (LoRAColumnParallelLinear, LoRARowParallelLinear, LoRATELinear)):
            if not getattr(module, "_lora_merged", False):
                continue
            with torch.no_grad():
                module.base_layer.weight.data.copy_(module._original_weight)
                del module._original_weight
            module._lora_merged = False
