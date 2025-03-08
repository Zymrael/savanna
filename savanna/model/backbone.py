# Copyright (c) 2021 EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch
import torch.nn as nn
from collections import defaultdict

from functools import partial
from savanna.model.utils import Lambda, SequentialWrapper, recursive_setattr
from savanna.model.operators.local.norms import get_norm
from savanna.model.init_functions import get_init_methods

from savanna import mpu
from savanna.utils import make_upper_case
from savanna.mpu import ParallelRelativePositionBias
from savanna.model.block import (
    ParallelBlockPipe,
    NormPipe,
    ParallelLinearPipe,
    parallel_lm_logits,
    ParallelLinear,
)
from savanna.model.operators.word_embeddings import EmbeddingPipe, SoftEmbedding

# Pipeline parallelism
import deepspeed.pipe as pipe
from typing import Union, List


def gpt2_attention_mask_func(attention_scores, ltor_mask):
    attention_scores.masked_fill_(ltor_mask, -10000.0)
    return attention_scores


def cross_entropy(output, labels, _fp16=False, reduce=True):
    """From pretrain_gpt2:forward_step()"""
    """
    if self.fp16_lm_cross_entropy:
        assert output.dtype == torch.half
        loss = mpu.vocab_parallel_cross_entropy(output, labels)
    else:
        loss = mpu.vocab_parallel_cross_entropy(output.float(), labels)
        return loss
    
    if reduce, performs mean reduction and applies loss_mask
    if not reduce, returns losses without loss_mask
    """
    labels, loss_mask = labels[0], labels[1]
    if _fp16:
        assert output.dtype == torch.half and loss_mask.dtype == torch.half
        losses = mpu.vocab_parallel_cross_entropy(output.contiguous(), labels)
    else:
        losses = mpu.vocab_parallel_cross_entropy(output.float().contiguous(), labels)

    if reduce:
        loss_mask = loss_mask.view(-1)
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        return loss
    else:
        return losses


def uppercase_ce(output, labels, _fp16=False, reduce=True):
    labels, loss_mask = labels[0], labels[1]
    labels, lowercase_mask = make_upper_case(labels)

    upper_loss_mask = loss_mask.bool() & (~lowercase_mask.bool())
    upper_loss_mask = upper_loss_mask.to(labels.dtype)

    return cross_entropy(output, (labels, upper_loss_mask), _fp16=_fp16, reduce=reduce)


def lowercase_ce(output, labels, _fp16=False, reduce=True):
    labels, loss_mask = labels[0], labels[1]
    labels, lowercase_mask = make_upper_case(labels)

    lower_loss_mask = loss_mask.bool() & (lowercase_mask.bool())
    lower_loss_mask = lower_loss_mask.to(labels.dtype)

    return cross_entropy(output, (labels, lower_loss_mask), _fp16=_fp16, reduce=reduce)


def reweighted_cross_entropy(output, labels, _fp16=False, reduce=True, lowercase_weight=1.0):
    """From pretrain_gpt2:forward_step()
    Modified for lower case loss reweighting"""
    """
    if self.fp16_lm_cross_entropy:
        assert output.dtype == torch.half
        loss = mpu.vocab_parallel_cross_entropy(output, labels)
    else:
        loss = mpu.vocab_parallel_cross_entropy(output.float(), labels)
        return loss
    """
    labels, loss_mask = labels[0], labels[1]

    labels, lowercase_mask = make_upper_case(labels)
    upper_loss_mask = loss_mask.bool() & (~lowercase_mask.bool())
    lower_loss_mask = loss_mask.bool() & lowercase_mask.bool()

    upper_loss_mask = upper_loss_mask.to(labels.dtype)
    lower_loss_mask = lower_loss_mask.to(labels.dtype)

    if _fp16:
        assert output.dtype == torch.half and upper_loss_mask.dtype == torch.half
        losses = mpu.vocab_parallel_cross_entropy(output.contiguous(), labels)
    else:
        losses = mpu.vocab_parallel_cross_entropy(output.float().contiguous(), labels)

    upper_case_loss = torch.sum(losses.view(-1) * upper_loss_mask.view(-1))
    lower_case_loss = torch.sum(losses.view(-1) * lower_loss_mask.view(-1))

    combined_loss = (upper_case_loss + lowercase_weight * lower_case_loss) / (
        upper_loss_mask.sum() + lower_loss_mask.sum()
    )

    return combined_loss


def oadm_loss(output, labels, _fp16=False, pad_token=1):
    """
    Loss for Order-agnostic Autoregressive Diffusion Modeling (Hoodgeboom et al., ICLR, 2021). Reimplementation based on EvoDiff (Almadri, et al., 2023)
    `output` contains values from a forward pass.
    `target` is a tuple containing `labels` (the actual token ids), and `loss_mask`.
    `pad_token` is used for reweighting
    """
    losses = cross_entropy(output, labels, _fp16=_fp16, reduce=False)

    labels, loss_mask = labels[0], labels[1]

    reweighting_term = 1.0 / (loss_mask).sum(dim=-1)
    non_pad_tokens = (labels != pad_token).sum(dim=-1)

    reweighted_loss = non_pad_tokens[:, None] * reweighting_term[:, None] * losses

    loss_mask = loss_mask.view(-1)
    loss = torch.sum(reweighted_loss.view(-1) * loss_mask) / loss_mask.sum()

    return loss


def dpo_loss(output, target, _fp16=False, beta=1.0):
    """
    Loss for Direct Preference Optimization (DPO) (Rafailov et al., NeurIPS, 2023).

    `output` is the outputted values from a forward pass.
    `target` is a tuple containing `labels` (the actual token ids), `logprobs` (the
        reference logprobs for each text), and the `loss_mask`.

    Refer to `_get_batch_dpo()` in `savanna/training.py` for how the `labels` are
    packed.
    """
    labels, loss_mask = target

    tokens = labels[:, :-1].long()
    ref_logprobs = labels[:, -1]

    if _fp16:
        assert output.dtype == torch.half and loss_mask.dtype == torch.half
        losses = mpu.vocab_parallel_cross_entropy(output.contiguous(), tokens)
    else:
        losses = mpu.vocab_parallel_cross_entropy(output.float().contiguous(), tokens)

    # `losses` should be of shape [2 * batch, length].
    losses = loss_mask * losses

    # Each loss is -log prob_\theta(x).
    losses = losses.mean(dim=-1)

    accept_logp, reject_logp = -losses.reshape(2, -1)
    accept_ref_logp, reject_ref_logp = ref_logprobs.reshape(2, -1)

    return -torch.nn.functional.logsigmoid(
        beta * ((accept_logp - accept_ref_logp) - (reject_logp - reject_ref_logp))
    ).mean()


def _pre_transformer_block(args):
    # data format change for hidden_states to avoid explicit tranposes : [b s h] --> [s b h]
    assert len(args) == 2, "Incorrect number of arguments to _pre_transformer_block"
    fn = lambda _args: (_args[0].transpose(0, 1).contiguous(), *_args[1:])
    return fn(args)


def _post_transformer_block(args):
    # from (hidden_states, attention_mask)
    # to (hidden_states.T)
    assert len(args) == 2, "Incorrect number of arguments to _post_transformer_block"
    fn = lambda _args: (_args[0].transpose(0, 1).contiguous())
    return fn(args)


class BackbonePipe(pipe.PipelineModule, torch.nn.Module):
    def __init__(
        self,
        global_config,
        num_tokentypes=0,
        parallel_output=True,
        topology=None,
        use_cache=False,
    ):
        self.global_config = global_config

        self.use_cache = use_cache
        self.parallel_output = parallel_output
        self.hidden_size = self.global_config.hidden_size
        self.num_tokentypes = num_tokentypes
        self.init_method, self.output_layer_init_method = get_init_methods(self.global_config)
        self.__topology__ = topology

        self.specs = []
        self.init_specs()  # initializes the layer specs (basically a fancy nn.Sequential)

        if self.global_config.alignment_method is None:
            if self.global_config.pretraining_strategy == "OADM":
                loss_fn = partial(
                    oadm_loss,
                    _fp16=self.global_config.fp16_lm_cross_entropy,
                    pad_token=global_config.tokenizer.pad,
                )
            else:
                if (self.global_config.to_upper == "weighted") or (self.global_config.to_upper == "weighted"):
                    loss_fn = partial(
                        reweighted_cross_entropy,
                        _fp16=self.global_config.fp16_lm_cross_entropy,
                        lowercase_weight=self.global_config.lowercase_loss_reweighting,
                    )
                else:
                    loss_fn = partial(cross_entropy, _fp16=self.global_config.fp16_lm_cross_entropy)

        elif self.global_config.alignment_method == "dpo":
            loss_fn = partial(
                dpo_loss,
                _fp16=self.global_config.fp16_lm_cross_entropy,
                beta=self.global_config.dpo_beta,
            )
        else:
            raise ValueError(f"Invalid alignment_method {self.global_config.alignment_method}.")

        super().__init__(
            layers=self.specs,
            loss_fn=loss_fn,
            topology=topology,
            activation_checkpoint_interval=(
                self.global_config.checkpoint_num_layers if self.global_config.checkpoint_activations else 0
            ),
            partition_method=global_config.pipe_partition_method,
            checkpointable_layers=["ParallelBlockPipe"],
            num_stages=global_config.pipe_parallel_size,
        )

    def insert_layers(self, layers: Union[nn.Module, nn.ModuleList, nn.Sequential, List], idx):
        """
        inserts the layers in `layers` into the pipe model at `idx`.
        """
        if isinstance(layers, nn.Module):
            self.specs.insert(idx, layers)
        elif any([isinstance(layers, nn.ModuleList), isinstance(layers, nn.Sequential)]):
            self.specs[idx:idx] = layers
        elif isinstance(layers, list):
            assert all([hasattr(l, "__call__") for l in layers]), "all items in `layers` must be Callables"
            self.specs[idx:idx] = layers
        else:
            raise ValueError(
                f"layer passed into {self.__class__.__name__}.insert_layer() should be either an nn.Module, an nn.ModuleList, an nn.Sequential object, or a list of callables not a {type(layers)}"
            )

        # re-initialize parent class
        super().__init__(
            layers=self.specs,
            loss_fn=self.loss_fn,
            topology=self.__topology__,
            activation_checkpoint_interval=self.activation_checkpoint_interval,
            partition_method=self.global_config.pipe_partition_method,
            checkpointable_layers=["ParallelBlockPipe"],
        )

    def init_specs(self):
        weight_tying = not self.global_config.no_weight_tying
        self.specs = []

        # Embedding layer
        # input will be (input_ids, position_ids, attention_mask)

        if weight_tying:
            self.specs.append(
                pipe.TiedLayerSpec(
                    "embed",
                    EmbeddingPipe,
                    self.global_config,
                    self.hidden_size,
                    self.global_config.padded_vocab_size,
                    self.global_config.max_position_embeddings,
                    self.global_config.hidden_dropout,
                    self.init_method,
                    self.num_tokentypes,
                    tied_weight_attr="word_embeddings_weight",
                )
            )
        else:
            self.specs.append(
                pipe.LayerSpec(
                    EmbeddingPipe,
                    self.global_config,
                    self.hidden_size,
                    self.global_config.padded_vocab_size,
                    self.global_config.max_position_embeddings,
                    self.global_config.hidden_dropout,
                    self.init_method,
                    self.num_tokentypes,
                )
            )

        # NB: the attention mask always needs to be the *last* item in the args when being passed from
        # one stage to the next, because deepspeed is hacks on top of hacks.
        #
        # outputs are now (hidden_states,  attention_mask)

        self.specs.append(_pre_transformer_block)

        # T5 RPE positional embedding
        if self.global_config.pos_emb == "rpe":
            hidden_size_per_attention_head = mpu.divide(
                self.global_config.hidden_size, self.global_config.num_attention_heads
            )
            rpe_scale = math.sqrt(hidden_size_per_attention_head)
            rpe_emb = ParallelRelativePositionBias(
                global_config=self.global_config,
                scale=rpe_scale,
                causal=True,
                num_buckets=self.global_config.rpe_num_buckets,
                max_distance=self.global_config.rpe_max_distance,
                heads=self.global_config.num_attention_heads,
            )

        # Transformer layers
        for i in range(self.global_config.num_layers):
            layer_type = self.global_config.operator_config[i]
            self.specs.append(
                pipe.LayerSpec(
                    ParallelBlockPipe,
                    global_config=self.global_config,
                    attention_mask_func=gpt2_attention_mask_func,
                    init_method=self.init_method,
                    output_layer_init_method=self.output_layer_init_method,
                    layer_number=i,
                    rpe=rpe_emb if self.global_config.pos_emb == "rpe" else None,
                    rotary="rotary" in self.global_config.pos_emb,  # check for any rotary type
                    use_cache=self.use_cache,
                )
            )

        # used to drop attention mask + reshape hidden states
        self.specs.append(_post_transformer_block)

        # NormPipe is a (deprecated) helper class that used to be used to pass presents along the pipeline - since presents are now cached to the `TransformerLayer` class this is no longer needed
        norm, eps = get_norm(self.global_config)
        self.specs.append(pipe.LayerSpec(NormPipe, norm, self.global_config.hidden_size, eps=eps))

        # outputs are now a single tensor: hidden_states

        def _logits_helper(embedding, lm_output):
            """Just a wrapper to massage inputs/outputs from pipeline."""
            if self.global_config.use_mup:
                # Since we're using pipeline parallelism, we can't directly use MuReadout. Instead, use this workaround that does the same thing as MuReadout.
                # https://github.com/microsoft/mup/issues/6#issuecomment-1082156274
                lm_output = lm_output / self.tied_modules.embed.word_embeddings.weight.infshape.width_mult()

            logits = parallel_lm_logits(
                lm_output,
                embedding.word_embeddings_weight,
                self.parallel_output,
                seq_parallel=self.global_config.sequence_parallel
            )
            return logits

        if weight_tying:
            self.specs.append(
                pipe.TiedLayerSpec(
                    "embed",
                    EmbeddingPipe,
                    self.global_config,
                    self.hidden_size,
                    self.global_config.padded_vocab_size,
                    self.global_config.max_position_embeddings,
                    self.global_config.hidden_dropout,
                    self.init_method,
                    self.num_tokentypes,
                    forward_fn=_logits_helper,
                    tied_weight_attr="word_embeddings_weight",
                )
            )
        else:
            self.specs.append(
                pipe.LayerSpec(
                    ParallelLinearPipe,
                    global_config=self.global_config,
                    init_method=self.init_method,
                    parallel_output=self.parallel_output,
                    is_last_layer=True,
                )
            )

    def _set_parallel_output(self, value):
        # sets the parallel output value of the final layer to value
        final_layer = list(self.forward_funcs)[-1]
        if isinstance(final_layer, (ParallelLinearPipe, ParallelLinear)):
            final_layer.final_linear.set_parallel_output(value)

    def inference_mode(self, use_cache=True):
        """
        Sets up the model for inference by turning on k/v caching (if specified) and setting `parallel output` of the final layer to false,
        so logits are gathered across model parallel ranks.

        :param cache: (bool) True if you want to use caching during inference, False otherwise
        """
        # first set caching to true if specified
        recursive_setattr(self.forward_funcs, "use_cache", use_cache, assert_type=bool)
        # then set parallel output of the final layer to false so we don't have to gather the output manually
        self._set_parallel_output(False)

    def train_mode(self):
        """
        Sets up the model for training by turning off k/v caching and setting `parallel output` of the final layer to True,
        so logits are not gathered across model parallel ranks, and loss is computed in parallel (more efficient).
        """
        # set caching to false
        recursive_setattr(self.forward_funcs, "use_cache", False)
        # then set parallel output to true (more efficient training)
        self._set_parallel_output(True)

    def clear_cache(self):
        """
        Recursively clears the kv cache on all layers
        """
        recursive_setattr(self.forward_funcs, "layer_past", None)

    def to_sequential(self):
        """
        Transforms the PipelineModule to a plain nn.Sequential module
        :return:
        """
        layers = []
        for i, func in enumerate(self.forward_funcs):
            if isinstance(func, torch.nn.Module):
                layers.append(func)
            elif hasattr(func, "__call__"):
                # check that it's a callable function
                layers.append(Lambda(func))
            else:
                raise ValueError(f"Layer number {i} ({func}) Not recognized")
        model = SequentialWrapper(
            layers,
            self.activation_checkpoint_interval,
            self.activation_checkpoint_func,
            parent_class_name=self.__class__.__name__,
        )
        return model

    # def get_additional_losses(self):
    #     lowercase_ce_loss = partial(lowercase_ce, _fp16=self.global_config.fp16_lm_cross_entropy)
    #     uppercase_ce_loss = partial(uppercase_ce, _fp16=self.global_config.fp16_lm_cross_entropy)

    #     return {"lowercase": lowercase_ce_loss(), "uppercase": uppercase_ce_loss()}
