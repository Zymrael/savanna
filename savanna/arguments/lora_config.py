from dataclasses import dataclass

try:
    from .template import GlobalConfigTemplate
except ImportError:
    from template import GlobalConfigTemplate


@dataclass
class GlobalConfigLoRA(GlobalConfigTemplate):

    lora: dict = None
    """
    LoRA configuration dict. Keys:
      'enabled': bool (default False) — whether LoRA is active
      'r': int (default 8) — rank
      'alpha': float (default 16.0) — scaling factor (scaling = alpha / r)
      'dropout': float (default 0.0)
      'target_modules': list[str] — leaf module name patterns to wrap,
          e.g. ["dense_projection", "dense", "w1", "w2", "w3"]
      'freeze_base_model': bool (default True) — freeze all non-LoRA params
      'modules_to_save': list[str] or None — additional module name patterns
          to keep trainable (e.g. ["layernorm"])
      'save_merged': bool (default False) — merge LoRA into base weights on save
      'lora_lr': float or None — separate LR for LoRA params
      'lora_weight_decay': float (default 0.0) — weight decay for LoRA params
    """

    def __post_init__(self):
        if self.lora is not None and self.lora.get("enabled", False):
            r = self.lora.get("r", 8)
            if not isinstance(r, int) or r < 1:
                raise ValueError(f"LoRA 'r' must be a positive integer, got {r!r}")

            alpha = self.lora.get("alpha", 16.0)
            if not isinstance(alpha, (int, float)) or alpha <= 0:
                raise ValueError(f"LoRA 'alpha' must be a positive number, got {alpha!r}")

            dropout = self.lora.get("dropout", 0.0)
            if not isinstance(dropout, (int, float)) or not (0.0 <= dropout < 1.0):
                raise ValueError(f"LoRA 'dropout' must be a float in [0, 1), got {dropout!r}")

            target_modules = self.lora.get("target_modules", [])
            if not isinstance(target_modules, list):
                raise ValueError(
                    f"LoRA 'target_modules' must be a list of strings, got {type(target_modules).__name__}"
                )
