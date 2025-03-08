# Copyright (c) 2023, Michael Poli.
# Checkpoint conversion code from safari-neox or savanna to this repo's format


import os
from typing import Any, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

import glob
from collections import OrderedDict
from src.utils import dotdict
from src.mpu import initialize_model_parallel, print_rank_0
from src.model import StripedHyena
from src.tokenizer import HFAutoTokenizer
from src.generation import Generator

# import rearrange
from einops import rearrange


KEY_UPDATE_DICT_HYENA = {
    "attention.mixer.filter.kernel.B": "filter.B",
    "attention.mixer.filter.kernel.C": "filter.C",
    "attention.mixer.long_conv_bias": "filter.D",
    "attention.mixer.filter.kernel.log_dt": "filter.log_dt",
    "attention.mixer.filter.kernel.inv_A_real": "filter.inv_A_real",
    "attention.mixer.filter.kernel.A_imag": "filter.A_imag",
    # short conv
    "attention.hyena_proj_conv.short_conv_weight": "filter.short_filter_weight",
    "attention.hyena_proj_conv.short_conv_bias": "filter.short_filter_bias",
    "attention.mixer.short_conv_weight": "",
    "attention.hyena_proj_conv.weight": "filter.short_filter_weight",
    "attention.hyena_proj_conv.bias": "filter.short_filter_bias",
    # rope
    "attention.rotary_emb.inv_freq": "rotary_emb.inv_freq",
    # qkv proj
    "attention.dense_projection.weight": "projections.weight",
    "attention.dense_projection.bias": "projections.bias",
    # mlp
    "mlp.w1.weight": "mlp.l1.weight",
    "mlp.w2.weight": "mlp.l2.weight",
    "mlp.w3.weight": "mlp.l3.weight",
    # dense layers
    "attention.dense.weight": "out_filter_dense.weight",
    "attention.dense.bias": "out_filter_dense.bias",
    # to scrap
    "mlp.w1._extra_state": "",
    "mlp.w2._extra_state": "",
    "mlp.w3._extra_state": "",
    "attention.dense._extra_state": "",
    "post_attention_layernorm.scale": "",
    "outer_mlp_layernorm.scale": "",
    "attention.dense_projection._extra_state": "",
    #
    "input_layernorm.weight": "pre_norm.scale",
    "input_layernorm.scale": "pre_norm.scale",
    "pre_mlp_layernorm.scale": "post_norm.scale",
    #
    "attention.mixer.filter.act.freq": "",
    "attention.mixer.filter.pos_emb.t": "",
    "attention.mixer.filter.pos_emb.z": "",
    "attention.mixer.filter.implicit_filter.0.weight": "",
    "attention.mixer.filter.implicit_filter.0.bias": "",
    "attention.mixer.filter.implicit_filter.1.freq": "",
    "attention.mixer.filter.implicit_filter.2.weight": "",
    "attention.mixer.filter.implicit_filter.2.bias": "",
    "attention.mixer.filter.implicit_filter.3.freq": "",
    "attention.mixer.filter.implicit_filter.4.weight": "",
    "attention.mixer.filter.implicit_filter.4.bias": "",
    "attention.mixer.filter.implicit_filter.5.freq": "",
    "attention.mixer.filter.final_filter.weight": "",
    "attention.mixer.filter.modulation.weight": "",
    #
    "mlp.gate_proj.weight": "mlp.l1.weight",
    "mlp.up_proj.weight": "mlp.l2.weight",
    "mlp.down_proj.weight": "mlp.l3.weight",
}


KEY_UPDATE_DICT_ATTENTION = {
    "attention.dense_projection.weight": "inner_mha_cls.Wqkv.weight",
    "attention.dense_projection.bias": "inner_mha_cls.Wqkv.bias",
    "attention.dense.weight": "inner_mha_cls.out_proj.weight",
    "attention.dense.bias": "inner_mha_cls.out_proj.bias",
    "attention.o_proj.weight": "inner_mha_cls.out_proj.weight",
    "attention.o_proj.bias": "inner_mha_cls.out_proj.bias",
    # rope
    # "attention.rotary_emb.inv_freq": "inner_mha_cls.rotary_emb.inv_freq",
    "attention.rotary_emb.inv_freq": "",
    # to scrap
    "mlp.w1._extra_state": "",
    "mlp.w2._extra_state": "",
    "mlp.w3._extra_state": "",
    "attention.dense._extra_state": "",
    "post_attention_layernorm.scale": "",
    "outer_mlp_layernorm.scale": "",
    "attention.dense_projection._extra_state": "",
    "attention.q_proj.weight": "",
    "attention.k_proj.weight": "",
    "attention.v_proj.weight": "",
    # mlp
    "mlp.w1.weight": "mlp.l1.weight",
    "mlp.w2.weight": "mlp.l2.weight",
    "mlp.w3.weight": "mlp.l3.weight",
    #
    "input_layernorm.weight": "pre_norm.scale",
    "post_attention_layernorm.weight": "post_norm.scale",
    "input_layernorm.weight": "pre_norm.scale",
    "input_layernorm.scale": "pre_norm.scale",
    "pre_mlp_layernorm.scale": "post_norm.scale",
    #
    "mlp.gate_proj.weight": "mlp.l1.weight",
    "mlp.up_proj.weight": "mlp.l2.weight",
    "mlp.down_proj.weight": "mlp.l3.weight",
    # misc
    "final_linear.weight": "word_embeddings.weight",
    "norm.weight": "norm.scale",
}


def filter_state_dict(state_dict):
    # replace keys according to mapping above
    # for q,k,v in attention, merge the weights in single tensor
    # 'attention.q_proj.weight', 'attention.k_proj.weight', 'attention.v_proj.weight'

    # automatic (brittle) detection of attention vs hyena
    keys = list(state_dict.keys())
    print(keys)
    if "attention.hyena_proj_conv.short_conv_weight" in keys:
        KEY_UPDATE_DICT = KEY_UPDATE_DICT_HYENA
    else:  # embeddings are grouped with attention for convenience
        KEY_UPDATE_DICT = KEY_UPDATE_DICT_ATTENTION

    params_to_merge = ["attention.q_proj.weight", "attention.k_proj.weight", "attention.v_proj.weight"]

    new_state_dict = OrderedDict()
    params_to_merge_values = []
    for k in state_dict.keys():
        if k.startswith("module."):
            k = k[7:]

        if k in params_to_merge:
            print("merging params for k:", k)
            params_to_merge_values.append(state_dict[k])

        print(f"Pre update: {k}")
        new_k = KEY_UPDATE_DICT.get(k, k)
        print(f"Post update: {new_k}")

        if new_k != "":
            print(state_dict[k].shape)
            new_state_dict[new_k] = state_dict[k]

        # # this is to handle the different convention of shapes in FlashAttention module in inference code
        # if 'inner_mha_cls.Wqkv.weight' in new_k:

        #     # rearrange "inner_mha_cls.Wqkv.weight" from [(head * three * dim) x dim] to [three x heads x dim x dim]
        #     new_Wqkv = rearrange(new_state_dict[new_k], '(head three d) d -> head three d d', head=1, three=3)

        #     # permute 1st and 2nd dims
        #     new_Wqkv = rearrange(new_Wqkv, 'head three d d -> three head d d', head=1, three=3)

        #     # recombine the 1st three dims into a single dim
        #     new_Wqkv = rearrange(new_Wqkv, 'three head d d -> (three head d) d', head=1, three=3)

        #     # set the new_k value to this new tensor
        #     new_state_dict[new_k] = new_Wqkv

    if len(params_to_merge_values) > 0:
        params_to_merge_values = torch.cat(params_to_merge_values, dim=0)
        new_state_dict["inner_mha_cls.Wqkv.weight"] = params_to_merge_values

    # convert modal parameters to pole / residue form
    if "filter.inv_A_real" in new_state_dict.keys():
        print("Detected alternative parametrization")
        A_real = -torch.exp(new_state_dict["filter.inv_A_real"])
        A_imag = new_state_dict["filter.A_imag"]

        dt = torch.exp(new_state_dict["filter.log_dt"])[:, None]
        dt_A = dt * (A_real + 1j * A_imag)
        C = new_state_dict["filter.C"]
        C = torch.view_as_complex(C.to(torch.float32))
        B = new_state_dict["filter.B"]
        B = torch.view_as_complex(B.to(torch.float32))

        # pop old keys
        new_state_dict.pop("filter.inv_A_real")
        new_state_dict.pop("filter.A_imag")
        new_state_dict.pop("filter.log_dt")
        new_state_dict.pop("filter.B")
        new_state_dict.pop("filter.C")

        residues = 2 * B * C * (1.0 - dt_A / 2).reciprocal() * dt
        poles = (1.0 + dt_A / 2) / (1.0 - dt_A / 2)

        new_state_dict["filter.poles"] = torch.view_as_real(poles).squeeze()[:, :, None]
        new_state_dict["filter.residues"] = torch.view_as_real(residues).squeeze()[:, :, None]

    return new_state_dict


def checkpoint_conversion(checkpoint_path, new_checkpoint_path):
    # loads checkpoint in deepspeed format ("layer-{idx}-model_00-model_states.pt")
    # assumes model parallel 1
    files = glob.glob(os.path.join(checkpoint_path, "layer*states.pt"))
    files = sorted(files, key=lambda x: int(x.split("/")[-1].split("_")[1].split("-")[0]))
    for idx, file in enumerate(files):
        state_dict = torch.load(file)
        print(f"Loading {file}, keys: {state_dict.keys()}", end="\n\n")

        state_dict = filter_state_dict(state_dict)

        new_file = file.split("/")[-1]
        torch.save(state_dict, os.path.join(new_checkpoint_path, f"layer_{idx:02d}.pt"))
