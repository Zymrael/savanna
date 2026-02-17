"""Unit tests for LoRA implementation."""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from savanna.lora import (
    LoRAColumnParallelLinear,
    LoRARowParallelLinear,
    LoRATELinear,
    apply_lora_to_model,
    get_lora_state_dict,
    load_lora_state_dict,
    merge_lora_weights,
    unmerge_lora_weights,
    _get_parent_and_attr,
)
from savanna.model.utils import _ensure_requires_grad


# ---------------------------------------------------------------------------
# Helpers: mock parallel linear layers that match the real interface
# ---------------------------------------------------------------------------


class MockColumnParallelLinear(nn.Module):
    """Mimics ColumnParallelLinear for testing without distributed init."""

    def __init__(self, input_size, output_size, tp_size=1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.output_size_per_partition = output_size // tp_size
        self.sequence_parallel = False
        self.seq_dim = 0
        self.weight = nn.Parameter(torch.randn(self.output_size_per_partition, input_size))
        self.bias = nn.Parameter(torch.randn(self.output_size_per_partition))

    def forward(self, input_):
        output = nn.functional.linear(input_, self.weight, self.bias)
        return output, None


class MockRowParallelLinear(nn.Module):
    """Mimics RowParallelLinear for testing without distributed init."""

    def __init__(self, input_size, output_size, tp_size=1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.input_size_per_partition = input_size // tp_size
        self.input_is_parallel = True
        self.sequence_parallel = False
        self.parallel_output = False
        self.seq_dim = 0
        self.weight = nn.Parameter(torch.randn(output_size, self.input_size_per_partition))
        self.bias = nn.Parameter(torch.randn(output_size))

    def forward(self, input_):
        output = nn.functional.linear(input_, self.weight, self.bias)
        return output, None


class MockTEConfig:
    """Mimics TransformerEngine config for TELinear."""

    def __init__(self, sequence_parallel=False, model_parallel_size=1, seq_dim=0):
        self.sequence_parallel = sequence_parallel
        self.model_parallel_size = model_parallel_size
        self.seq_dim = seq_dim


class MockTELinear(nn.Module):
    """Mimics TELinear for testing without TransformerEngine or distributed init.

    For column parallel: in_features is full, out_features is partitioned.
    For row parallel: in_features is partitioned, out_features is full.
    """

    def __init__(self, in_features, out_features, config, parallel_mode="column"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.parallel_mode = parallel_mode
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, input_):
        output = nn.functional.linear(input_, self.weight, self.bias)
        return output, None


def _identity(x, **kwargs):
    """No-op replacement for distributed comm functions in tests."""
    return x


# ---------------------------------------------------------------------------
# Tests: LoRA layer shapes
# ---------------------------------------------------------------------------


class TestLoRAColumnParallelLinear:
    def test_shapes(self):
        base = MockColumnParallelLinear(64, 128)
        lora = LoRAColumnParallelLinear(base, r=8, alpha=16.0)
        assert lora.lora_A.shape == (8, 64)
        assert lora.lora_B.shape == (128, 8)

    def test_shapes_with_tp(self):
        base = MockColumnParallelLinear(64, 128, tp_size=2)
        lora = LoRAColumnParallelLinear(base, r=4, alpha=8.0)
        # A is full input, B is partitioned
        assert lora.lora_A.shape == (4, 64)
        assert lora.lora_B.shape == (64, 4)  # 128 // 2 = 64

    def test_zero_init_b(self):
        base = MockColumnParallelLinear(64, 128)
        lora = LoRAColumnParallelLinear(base, r=8, alpha=16.0)
        assert torch.all(lora.lora_B == 0)

    def test_forward_matches_base_when_b_is_zero(self):
        base = MockColumnParallelLinear(32, 64)
        lora = LoRAColumnParallelLinear(base, r=4, alpha=8.0)
        x = torch.randn(2, 10, 32)
        base_out, base_bias = base(x)
        lora_out, lora_bias = lora(x)
        torch.testing.assert_close(lora_out, base_out)

    def test_forward_adds_lora_delta(self):
        base = MockColumnParallelLinear(32, 64)
        lora = LoRAColumnParallelLinear(base, r=4, alpha=8.0)
        # Set B to non-zero
        lora.lora_B.data.fill_(0.1)
        x = torch.randn(2, 10, 32)
        base_out, _ = base(x)
        lora_out, _ = lora(x)
        # Output should differ now
        assert not torch.allclose(lora_out, base_out)

    def test_scaling(self):
        base = MockColumnParallelLinear(32, 64)
        lora = LoRAColumnParallelLinear(base, r=4, alpha=16.0)
        assert lora.scaling == 16.0 / 4

    def test_is_lora_param_attribute(self):
        base = MockColumnParallelLinear(32, 64)
        lora = LoRAColumnParallelLinear(base, r=4, alpha=8.0)
        assert lora.lora_A._is_lora_param is True
        assert lora.lora_B._is_lora_param is True


class TestLoRARowParallelLinear:
    def test_shapes(self):
        base = MockRowParallelLinear(64, 128)
        lora = LoRARowParallelLinear(base, r=8, alpha=16.0)
        assert lora.lora_A.shape == (8, 64)
        assert lora.lora_B.shape == (128, 8)

    def test_shapes_with_tp(self):
        base = MockRowParallelLinear(128, 64, tp_size=2)
        lora = LoRARowParallelLinear(base, r=4, alpha=8.0)
        # A is partitioned (input_size_per_partition), B is full
        assert lora.lora_A.shape == (4, 64)  # 128 // 2 = 64
        assert lora.lora_B.shape == (64, 4)

    def test_zero_init_b(self):
        base = MockRowParallelLinear(64, 128)
        lora = LoRARowParallelLinear(base, r=8, alpha=16.0)
        assert torch.all(lora.lora_B == 0)

    @patch("savanna.lora.reduce_from_model_parallel_region", side_effect=_identity)
    def test_forward_matches_base_when_b_is_zero(self, mock_reduce):
        base = MockRowParallelLinear(32, 64)
        lora = LoRARowParallelLinear(base, r=4, alpha=8.0)
        x = torch.randn(2, 10, 32)
        base_out, base_bias = base(x)
        lora_out, lora_bias = lora(x)
        torch.testing.assert_close(lora_out, base_out)


# ---------------------------------------------------------------------------
# Tests: merge/unmerge
# ---------------------------------------------------------------------------


class TestMergeUnmerge:
    def _make_model_with_lora(self):
        model = nn.Module()
        base = MockColumnParallelLinear(32, 64)
        lora = LoRAColumnParallelLinear(base, r=4, alpha=8.0)
        # Set non-zero LoRA weights
        lora.lora_A.data.normal_()
        lora.lora_B.data.normal_()
        model.layer = lora
        return model

    def test_merge_changes_base_weight(self):
        model = self._make_model_with_lora()
        original_weight = model.layer.base_layer.weight.data.clone()
        merge_lora_weights(model)
        assert not torch.allclose(model.layer.base_layer.weight.data, original_weight)

    def test_unmerge_restores_base_weight(self):
        model = self._make_model_with_lora()
        original_weight = model.layer.base_layer.weight.data.clone()
        merge_lora_weights(model)
        unmerge_lora_weights(model)
        torch.testing.assert_close(model.layer.base_layer.weight.data, original_weight)

    def test_double_merge_is_noop(self):
        model = self._make_model_with_lora()
        merge_lora_weights(model)
        weight_after_first_merge = model.layer.base_layer.weight.data.clone()
        merge_lora_weights(model)  # should be a no-op
        torch.testing.assert_close(model.layer.base_layer.weight.data, weight_after_first_merge)

    def test_merge_flag(self):
        model = self._make_model_with_lora()
        assert not getattr(model.layer, "_lora_merged", False)
        merge_lora_weights(model)
        assert model.layer._lora_merged is True
        unmerge_lora_weights(model)
        assert model.layer._lora_merged is False


# ---------------------------------------------------------------------------
# Tests: state dict helpers
# ---------------------------------------------------------------------------


class TestStateDictHelpers:
    def test_get_lora_state_dict(self):
        model = nn.Module()
        base = MockColumnParallelLinear(32, 64)
        lora = LoRAColumnParallelLinear(base, r=4, alpha=8.0)
        model.layer = lora
        sd = get_lora_state_dict(model)
        assert "layer.lora_A" in sd
        assert "layer.lora_B" in sd
        # Should NOT contain base layer weights
        assert not any("base_layer" in k and "lora_" not in k for k in sd)

    def test_load_lora_state_dict(self):
        model = nn.Module()
        base = MockColumnParallelLinear(32, 64)
        lora = LoRAColumnParallelLinear(base, r=4, alpha=8.0)
        model.layer = lora

        # Save the original lora_A
        original_A = model.layer.lora_A.data.clone()
        sd = get_lora_state_dict(model)
        # Modify the params in the model
        model.layer.lora_A.data.fill_(42.0)
        assert torch.all(model.layer.lora_A.data == 42.0)
        # Reload from saved state dict
        load_lora_state_dict(model, sd)
        # Should be restored to original
        torch.testing.assert_close(model.layer.lora_A.data, original_A)


# ---------------------------------------------------------------------------
# Tests: apply_lora_to_model
# ---------------------------------------------------------------------------


class TestApplyLoRA:
    def _make_model_and_config(self, target_modules):
        model = nn.Module()
        model.dense_projection = MockColumnParallelLinear(32, 64)
        model.dense = MockRowParallelLinear(64, 32)
        model.layernorm = nn.LayerNorm(32)

        global_config = MagicMock()
        global_config.lora = {
            "enabled": True,
            "r": 4,
            "alpha": 8.0,
            "dropout": 0.0,
            "target_modules": target_modules,
        }
        return model, global_config

    @patch("savanna.lora.print_rank_0")
    @patch("savanna.mpu.layers.ColumnParallelLinear", MockColumnParallelLinear)
    @patch("savanna.mpu.layers.RowParallelLinear", MockRowParallelLinear)
    def test_wraps_target_modules(self, mock_print):
        model, config = self._make_model_and_config(["dense_projection", "dense"])
        apply_lora_to_model(model, config)
        assert isinstance(model.dense_projection, LoRAColumnParallelLinear)
        assert isinstance(model.dense, LoRARowParallelLinear)

    @patch("savanna.lora.print_rank_0")
    @patch("savanna.mpu.layers.ColumnParallelLinear", MockColumnParallelLinear)
    @patch("savanna.mpu.layers.RowParallelLinear", MockRowParallelLinear)
    def test_does_not_wrap_non_targets(self, mock_print):
        model, config = self._make_model_and_config(["dense_projection"])
        apply_lora_to_model(model, config)
        assert isinstance(model.dense_projection, LoRAColumnParallelLinear)
        assert isinstance(model.dense, MockRowParallelLinear)  # not wrapped
        assert isinstance(model.layernorm, nn.LayerNorm)  # not wrapped

    @patch("savanna.lora.print_rank_0")
    @patch("savanna.mpu.layers.ColumnParallelLinear", MockColumnParallelLinear)
    @patch("savanna.mpu.layers.RowParallelLinear", MockRowParallelLinear)
    def test_base_layer_preserved(self, mock_print):
        model, config = self._make_model_and_config(["dense_projection"])
        original_weight = model.dense_projection.weight.data.clone()
        apply_lora_to_model(model, config)
        torch.testing.assert_close(model.dense_projection.base_layer.weight.data, original_weight)

    @patch("savanna.lora.print_rank_0")
    @patch("savanna.mpu.layers.ColumnParallelLinear", MockColumnParallelLinear)
    @patch("savanna.mpu.layers.RowParallelLinear", MockRowParallelLinear)
    def test_nested_modules(self, mock_print):
        model = nn.Module()
        block = nn.Module()
        block.dense_projection = MockColumnParallelLinear(32, 64)
        block.dense = MockRowParallelLinear(64, 32)
        model.block = block

        config = MagicMock()
        config.lora = {
            "enabled": True,
            "r": 4,
            "alpha": 8.0,
            "dropout": 0.0,
            "target_modules": ["dense_projection", "dense"],
        }
        apply_lora_to_model(model, config)
        assert isinstance(model.block.dense_projection, LoRAColumnParallelLinear)
        assert isinstance(model.block.dense, LoRARowParallelLinear)


# ---------------------------------------------------------------------------
# Tests: _get_parent_and_attr helper
# ---------------------------------------------------------------------------


class TestGetParentAndAttr:
    def test_simple_name(self):
        model = nn.Module()
        model.layer = nn.Linear(10, 10)
        parent, attr = _get_parent_and_attr(model, "layer")
        assert parent is model
        assert attr == "layer"

    def test_dotted_name(self):
        model = nn.Module()
        model.block = nn.Module()
        model.block.layer = nn.Linear(10, 10)
        parent, attr = _get_parent_and_attr(model, "block.layer")
        assert parent is model.block
        assert attr == "layer"


# ---------------------------------------------------------------------------
# Tests: config parsing
# ---------------------------------------------------------------------------


class TestLoRAConfig:
    def test_config_defaults(self):
        from savanna.arguments.lora_config import GlobalConfigLoRA

        config = GlobalConfigLoRA()
        assert config.lora is None

    def test_config_from_dict(self):
        from savanna.arguments.lora_config import GlobalConfigLoRA

        config = GlobalConfigLoRA(
            lora={
                "enabled": True,
                "r": 16,
                "alpha": 32.0,
                "target_modules": ["dense"],
            }
        )
        assert config.lora["enabled"] is True
        assert config.lora["r"] == 16
        assert config.lora["alpha"] == 32.0
        assert config.lora["target_modules"] == ["dense"]


# ---------------------------------------------------------------------------
# Tests: dropout
# ---------------------------------------------------------------------------


class TestDropout:
    def test_no_dropout(self):
        base = MockColumnParallelLinear(32, 64)
        lora = LoRAColumnParallelLinear(base, r=4, alpha=8.0, dropout=0.0)
        assert isinstance(lora.lora_dropout, nn.Identity)

    def test_with_dropout(self):
        base = MockColumnParallelLinear(32, 64)
        lora = LoRAColumnParallelLinear(base, r=4, alpha=8.0, dropout=0.1)
        assert isinstance(lora.lora_dropout, nn.Dropout)


# ---------------------------------------------------------------------------
# Tests: integration - forward/backward pass
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_backward_pass(self):
        base = MockColumnParallelLinear(32, 64)
        lora = LoRAColumnParallelLinear(base, r=4, alpha=8.0)
        lora.lora_B.data.fill_(0.1)

        x = torch.randn(2, 10, 32, requires_grad=True)
        out, _ = lora(x)
        loss = out.sum()
        loss.backward()

        assert lora.lora_A.grad is not None
        assert lora.lora_B.grad is not None

    def test_only_lora_params_require_grad_after_freeze(self):
        model = nn.Module()
        model.dense = MockColumnParallelLinear(32, 64)

        # Wrap with LoRA
        lora = LoRAColumnParallelLinear(model.dense, r=4, alpha=8.0)
        model.dense = lora

        # Freeze base
        for name, param in model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        assert "dense.lora_A" in trainable
        assert "dense.lora_B" in trainable
        assert all("lora_" in n for n in trainable)


# ---------------------------------------------------------------------------
# Tests: LoRATELinear
# ---------------------------------------------------------------------------


class TestLoRATELinear:
    def test_column_shapes_tp1(self):
        config = MockTEConfig(model_parallel_size=1)
        base = MockTELinear(64, 128, config, parallel_mode="column")
        lora = LoRATELinear(base, r=8, alpha=16.0, parallel_mode="column")
        # A: full input, B: partitioned output (= out_features with TP=1)
        assert lora.lora_A.shape == (8, 64)
        assert lora.lora_B.shape == (128, 8)

    def test_column_shapes_tp2(self):
        config = MockTEConfig(model_parallel_size=2)
        # With TP=2 column: TE stores out_features = full / tp_size
        base = MockTELinear(64, 64, config, parallel_mode="column")  # 128/2 = 64 out
        lora = LoRATELinear(base, r=4, alpha=8.0, parallel_mode="column")
        assert lora.lora_A.shape == (4, 64)  # full input
        assert lora.lora_B.shape == (64, 4)  # partitioned output

    def test_row_shapes_tp1(self):
        config = MockTEConfig(model_parallel_size=1)
        base = MockTELinear(64, 128, config, parallel_mode="row")
        lora = LoRATELinear(base, r=8, alpha=16.0, parallel_mode="row")
        # A: partitioned input (= in_features with TP=1), B: full output
        assert lora.lora_A.shape == (8, 64)
        assert lora.lora_B.shape == (128, 8)

    def test_row_shapes_tp2(self):
        config = MockTEConfig(model_parallel_size=2)
        # With TP=2 row: TE stores in_features = full / tp_size, out_features = full
        base = MockTELinear(32, 128, config, parallel_mode="row")  # 64/2 = 32 in
        lora = LoRATELinear(base, r=4, alpha=8.0, parallel_mode="row")
        assert lora.lora_A.shape == (4, 32)  # partitioned input
        assert lora.lora_B.shape == (128, 4)  # full output

    def test_zero_init_b(self):
        config = MockTEConfig()
        base = MockTELinear(64, 128, config)
        lora = LoRATELinear(base, r=8, alpha=16.0, parallel_mode="column")
        assert torch.all(lora.lora_B == 0)

    def test_lora_params_are_bf16(self):
        config = MockTEConfig()
        base = MockTELinear(64, 128, config)
        lora = LoRATELinear(base, r=8, alpha=16.0, parallel_mode="column")
        assert lora.lora_A.dtype == torch.bfloat16
        assert lora.lora_B.dtype == torch.bfloat16

    @patch("savanna.lora.reduce_from_model_parallel_region", side_effect=_identity)
    def test_column_forward_matches_base_when_b_is_zero(self, mock_reduce):
        config = MockTEConfig()
        base = MockTELinear(32, 64, config)
        lora = LoRATELinear(base, r=4, alpha=8.0, parallel_mode="column")
        x = torch.randn(2, 10, 32)
        base_out, _ = base(x)
        lora_out, _ = lora(x)
        torch.testing.assert_close(lora_out, base_out)

    @patch("savanna.lora.reduce_from_model_parallel_region", side_effect=_identity)
    def test_row_forward_matches_base_when_b_is_zero(self, mock_reduce):
        config = MockTEConfig()
        base = MockTELinear(32, 64, config)
        lora = LoRATELinear(base, r=4, alpha=8.0, parallel_mode="row")
        x = torch.randn(2, 10, 32)
        base_out, _ = base(x)
        lora_out, _ = lora(x)
        torch.testing.assert_close(lora_out, base_out)

    @patch("savanna.lora.reduce_from_model_parallel_region", side_effect=_identity)
    def test_column_forward_adds_lora_delta(self, mock_reduce):
        config = MockTEConfig()
        base = MockTELinear(32, 64, config)
        lora = LoRATELinear(base, r=4, alpha=8.0, parallel_mode="column")
        lora.lora_B.data.fill_(0.1)
        x = torch.randn(2, 10, 32)
        base_out, _ = base(x)
        lora_out, _ = lora(x)
        assert not torch.allclose(lora_out, base_out)

    @patch("savanna.lora.reduce_from_model_parallel_region", side_effect=_identity)
    def test_row_forward_adds_lora_delta(self, mock_reduce):
        config = MockTEConfig()
        base = MockTELinear(32, 64, config)
        lora = LoRATELinear(base, r=4, alpha=8.0, parallel_mode="row")
        lora.lora_B.data.fill_(0.1)
        x = torch.randn(2, 10, 32)
        base_out, _ = base(x)
        lora_out, _ = lora(x)
        assert not torch.allclose(lora_out, base_out)

    @patch("savanna.lora.reduce_from_model_parallel_region", side_effect=_identity)
    def test_backward_pass_column(self, mock_reduce):
        config = MockTEConfig()
        base = MockTELinear(32, 64, config)
        lora = LoRATELinear(base, r=4, alpha=8.0, parallel_mode="column")
        lora.lora_B.data.fill_(0.1)
        x = torch.randn(2, 10, 32, requires_grad=True)
        out, _ = lora(x)
        loss = out.sum()
        loss.backward()
        assert lora.lora_A.grad is not None
        assert lora.lora_B.grad is not None

    @patch("savanna.lora.reduce_from_model_parallel_region", side_effect=_identity)
    def test_backward_pass_row(self, mock_reduce):
        config = MockTEConfig()
        base = MockTELinear(32, 64, config)
        lora = LoRATELinear(base, r=4, alpha=8.0, parallel_mode="row")
        lora.lora_B.data.fill_(0.1)
        x = torch.randn(2, 10, 32, requires_grad=True)
        out, _ = lora(x)
        loss = out.sum()
        loss.backward()
        assert lora.lora_A.grad is not None
        assert lora.lora_B.grad is not None

    def test_is_lora_param_attribute(self):
        config = MockTEConfig()
        base = MockTELinear(32, 64, config)
        lora = LoRATELinear(base, r=4, alpha=8.0, parallel_mode="column")
        assert lora.lora_A._is_lora_param is True
        assert lora.lora_B._is_lora_param is True

    def test_invalid_parallel_mode_raises(self):
        config = MockTEConfig()
        base = MockTELinear(32, 64, config)
        with pytest.raises(ValueError, match="Cannot determine parallel_mode"):
            LoRATELinear(base, r=4, alpha=8.0, parallel_mode=None)

    def test_seq_dim_from_config(self):
        config = MockTEConfig(seq_dim=1)
        base = MockTELinear(32, 64, config)
        lora = LoRATELinear(base, r=4, alpha=8.0, parallel_mode="column")
        assert lora.seq_dim == 1


# ---------------------------------------------------------------------------
# Tests: LoRARowParallelLinear additional tests
# ---------------------------------------------------------------------------


class TestLoRARowParallelLinearExtended:
    @patch("savanna.lora.reduce_from_model_parallel_region", side_effect=_identity)
    def test_backward_pass(self, mock_reduce):
        base = MockRowParallelLinear(32, 64)
        lora = LoRARowParallelLinear(base, r=4, alpha=8.0)
        lora.lora_B.data.fill_(0.1)
        x = torch.randn(2, 10, 32, requires_grad=True)
        out, _ = lora(x)
        loss = out.sum()
        loss.backward()
        assert lora.lora_A.grad is not None
        assert lora.lora_B.grad is not None

    @patch("savanna.lora.reduce_from_model_parallel_region", side_effect=_identity)
    def test_forward_adds_lora_delta(self, mock_reduce):
        base = MockRowParallelLinear(32, 64)
        lora = LoRARowParallelLinear(base, r=4, alpha=8.0)
        lora.lora_B.data.fill_(0.1)
        x = torch.randn(2, 10, 32)
        base_out, _ = base(x)
        lora_out, _ = lora(x)
        assert not torch.allclose(lora_out, base_out)

    def test_parallel_output_warning(self):
        """parallel_output=True with TP > 1 should emit a warning."""
        base = MockRowParallelLinear(128, 64, tp_size=2)
        base.parallel_output = True
        with patch("savanna.lora.logger") as mock_logger:
            LoRARowParallelLinear(base, r=4, alpha=8.0)
            mock_logger.warning.assert_called_once()
            assert "parallel_output=True" in mock_logger.warning.call_args[0][0]


# ---------------------------------------------------------------------------
# Tests: _ensure_requires_grad
# ---------------------------------------------------------------------------


class TestEnsureRequiresGrad:
    def test_enables_grad_on_first_float_tensor(self):
        t = torch.randn(4, 4)
        assert not t.requires_grad
        result = _ensure_requires_grad((t,))
        assert result[0].requires_grad

    def test_only_enables_first_tensor(self):
        t1 = torch.randn(4)
        t2 = torch.randn(4)
        result = _ensure_requires_grad((t1, t2))
        assert result[0].requires_grad
        assert not result[1].requires_grad

    def test_skips_non_float_tensors(self):
        int_t = torch.tensor([1, 2, 3])
        float_t = torch.randn(4)
        result = _ensure_requires_grad((int_t, float_t))
        assert not result[0].requires_grad  # int tensor skipped
        assert result[1].requires_grad  # first float tensor gets grad

    def test_noop_if_already_requires_grad(self):
        t = torch.randn(4, requires_grad=True)
        result = _ensure_requires_grad((t,))
        assert result[0] is t  # same object, not detached

    def test_noop_for_non_tensor_args(self):
        result = _ensure_requires_grad((None, "string", 42))
        assert result == (None, "string", 42)

    def test_mixed_args(self):
        non_tensor = "hello"
        int_t = torch.tensor([1, 2])
        float_t = torch.randn(3)
        result = _ensure_requires_grad((non_tensor, int_t, float_t))
        assert result[0] == "hello"
        assert not result[1].requires_grad
        assert result[2].requires_grad


# ---------------------------------------------------------------------------
# Tests: config validation
# ---------------------------------------------------------------------------


class TestLoRAConfigValidation:
    def test_r_zero_raises(self):
        from savanna.arguments.lora_config import GlobalConfigLoRA

        with pytest.raises(ValueError, match="positive integer"):
            GlobalConfigLoRA(lora={"enabled": True, "r": 0, "target_modules": ["dense"]})

    def test_r_float_raises(self):
        from savanna.arguments.lora_config import GlobalConfigLoRA

        with pytest.raises(ValueError, match="positive integer"):
            GlobalConfigLoRA(lora={"enabled": True, "r": 8.5, "target_modules": ["dense"]})

    def test_r_negative_raises(self):
        from savanna.arguments.lora_config import GlobalConfigLoRA

        with pytest.raises(ValueError, match="positive integer"):
            GlobalConfigLoRA(lora={"enabled": True, "r": -1, "target_modules": ["dense"]})

    def test_alpha_zero_raises(self):
        from savanna.arguments.lora_config import GlobalConfigLoRA

        with pytest.raises(ValueError, match="positive number"):
            GlobalConfigLoRA(lora={"enabled": True, "r": 8, "alpha": 0, "target_modules": ["dense"]})

    def test_alpha_negative_raises(self):
        from savanna.arguments.lora_config import GlobalConfigLoRA

        with pytest.raises(ValueError, match="positive number"):
            GlobalConfigLoRA(lora={"enabled": True, "r": 8, "alpha": -1.0, "target_modules": ["dense"]})

    def test_dropout_out_of_range_raises(self):
        from savanna.arguments.lora_config import GlobalConfigLoRA

        with pytest.raises(ValueError, match="dropout"):
            GlobalConfigLoRA(
                lora={"enabled": True, "r": 8, "dropout": 1.0, "target_modules": ["dense"]}
            )

    def test_dropout_negative_raises(self):
        from savanna.arguments.lora_config import GlobalConfigLoRA

        with pytest.raises(ValueError, match="dropout"):
            GlobalConfigLoRA(
                lora={"enabled": True, "r": 8, "dropout": -0.1, "target_modules": ["dense"]}
            )

    def test_target_modules_string_raises(self):
        from savanna.arguments.lora_config import GlobalConfigLoRA

        with pytest.raises(ValueError, match="list of strings"):
            GlobalConfigLoRA(lora={"enabled": True, "r": 8, "target_modules": "dense"})

    def test_valid_config_passes(self):
        from savanna.arguments.lora_config import GlobalConfigLoRA

        config = GlobalConfigLoRA(
            lora={"enabled": True, "r": 8, "alpha": 16.0, "dropout": 0.1, "target_modules": ["dense"]}
        )
        assert config.lora["r"] == 8

    def test_disabled_config_skips_validation(self):
        from savanna.arguments.lora_config import GlobalConfigLoRA

        # Should not raise even with bad r, because enabled=False
        config = GlobalConfigLoRA(lora={"enabled": False, "r": 0})
        assert config.lora["r"] == 0


# ---------------------------------------------------------------------------
# Tests: unmerge numerical stability (bf16)
# ---------------------------------------------------------------------------


class TestUnmergeStability:
    def test_unmerge_exact_in_bf16(self):
        """Verify unmerge restores the exact original weight in bf16."""
        base = MockColumnParallelLinear(32, 64)
        base.weight.data = base.weight.data.to(torch.bfloat16)
        lora = LoRAColumnParallelLinear(base, r=4, alpha=8.0)
        lora.lora_A.data = lora.lora_A.data.to(torch.bfloat16)
        lora.lora_B.data.normal_()
        lora.lora_B.data = lora.lora_B.data.to(torch.bfloat16)

        model = nn.Module()
        model.layer = lora

        original_weight = base.weight.data.clone()
        merge_lora_weights(model)
        unmerge_lora_weights(model)
        # Should be bit-exact since we restore from saved original
        assert torch.equal(base.weight.data, original_weight)
