"""
Checkpoint conversion code from savanna to vortex

python3 convert_checkpoint_to_vortex.py

@mp: This works with the latest version of Evo2 models (SHC, trained post onboarding on NVIDIA cluster). It does NOT work with SHC 1.5 models (e.g. VP run)

Also computes a dummy set of logits to test correctness of the conversion

Make sure to check whether the GatedMLP is using the right activation after layer idx 1 when computing logits. If the activation func is still a lambda, force it to be F.gelu
"""

import os
from typing import Any, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from savanna.ops.vandermonde import log_vandermonde_naive as log_vandermonde
from opt_einsum import contract
import glob
from collections import OrderedDict

from savanna.model.backbone import ParallelBlockPipe, EmbeddingPipe, NormPipe, Lambda

from tools.load_checkpoint_from_deepspeed_raw import load_savanna_checkpoint

# import rearrange
from einops import rearrange


KEY_UPDATE_DICT_HYENA = {
    # hyena
    "mixer.mixer.filter.h": "filter.h",
    "mixer.mixer.filter.kernel.B": "filter.B",
    "mixer.mixer.filter.kernel.C": "filter.C",
    "mixer.mixer.conv_bias": "filter.D",
    "mixer.mixer.filter.decay": "",
    "mixer.mixer.filter.gamma": "",
    "mixer.mixer.filter.kernel.log_dt": "filter.log_dt",
    "mixer.mixer.filter.kernel.inv_A_real": "filter.inv_A_real",
    "mixer.mixer.filter.kernel.A_imag": "filter.A_imag",
    # short conv
    "mixer.hyena_proj_conv.short_conv_weight": "filter.short_filter_weight",
    "mixer.hyena_proj_conv.short_conv_bias": "filter.short_filter_bias",
    "mixer.mixer.short_conv_weight": "",
    "mixer.mixer.short_conv.short_conv_weight": "",
    # "mixer.hyena_proj_conv.weight": "filter.short_filter_weight",
    "mixer.hyena_proj_conv.bias": "filter.short_filter_bias",
    # rope
    "mixer.rotary_emb.inv_freq": "rotary_emb.inv_freq",
    # qkv proj
    "mixer.dense_projection.weight": "projections.weight",
    "mixer.dense_projection.bias": "projections.bias",
    # mlp
    "mlp.w1.weight": "mlp.l1.weight",
    "mlp.w2.weight": "mlp.l2.weight",
    "mlp.w3.weight": "mlp.l3.weight",
    # dense layers
    "mixer.dense.weight": "out_filter_dense.weight",
    "mixer.dense.bias": "out_filter_dense.bias",
    # to scrap
    "mlp.w1._extra_state": "",
    "mlp.w2._extra_state": "",
    "mlp.w3._extra_state": "",
    "mixer.dense._extra_state": "",
    "post_attention_layernorm.scale": "",
    "outer_mlp_layernorm.scale": "",
    "mixer.dense_projection._extra_state": "",
    "input_layernorm.weight": "pre_norm.scale",
    "post_attention_layernorm.weight": "post_norm.scale",
    "input_layernorm.weight": "pre_norm.scale",
    "input_layernorm.scale": "pre_norm.scale",
    "pre_mlp_layernorm.scale": "post_norm.scale",
    #
    "input_layernorm.weight": "pre_norm.scale",
    "input_layernorm.scale": "pre_norm.scale",
    "pre_mlp_layernorm.scale": "post_norm.scale",
    "pre_mlp_layernorm.weight": "post_norm.scale",
    "outer_mlp_layernorm.weight": "",
    #
    "mixer.mixer.filter.act.freq": "",
    "mixer.mixer.filter.pos_emb.t": "",
    "mixer.mixer.filter.pos_emb.z": "",
    "mixer.mixer.filter.implicit_filter.0.weight": "",
    "mixer.mixer.filter.implicit_filter.0.bias": "",
    "mixer.mixer.filter.implicit_filter.1.freq": "",
    "mixer.mixer.filter.implicit_filter.2.weight": "",
    "mixer.mixer.filter.implicit_filter.2.bias": "",
    "mixer.mixer.filter.implicit_filter.3.freq": "",
    "mixer.mixer.filter.implicit_filter.4.weight": "",
    "mixer.mixer.filter.implicit_filter.4.bias": "",
    "mixer.mixer.filter.implicit_filter.5.freq": "",
    "mixer.mixer.filter.final_filter.weight": "",
    "mixer.mixer.filter.modulation.weight": "",
    #
    "mlp.w1.weight": "mlp.l1.weight",
    "mlp.w2.weight": "mlp.l2.weight",
    "mlp.w3.weight": "mlp.l3.weight",

    "norm.weight": "norm.scale",
}

KEY_UPDATE_DICT_ATTENTION = {
    "mixer.dense_projection.weight": "inner_mha_cls.Wqkv.weight",
    "mixer.dense_projection.bias": "inner_mha_cls.Wqkv.bias",
    "mixer.dense.weight": "inner_mha_cls.out_proj.weight",
    "mixer.dense.bias": "inner_mha_cls.out_proj.bias",
    "mixer.o_proj.weight": "inner_mha_cls.out_proj.weight",
    "mixer.o_proj.bias": "inner_mha_cls.out_proj.bias",
    # rope
    # "attention.rotary_emb.inv_freq": "inner_mha_cls.rotary_emb.inv_freq",
    "mixer.rotary_emb.inv_freq": "inner_mha_cls.rotary_emb.inv_freq",
    # to scrap
    "mlp.w1._extra_state": "",
    "mlp.w2._extra_state": "",
    "mlp.w3._extra_state": "",
    "attention.dense._extra_state": "",
    "post_attention_layernorm.scale": "",
    "outer_mlp_layernorm.weight": "",
    "outer_mlp_layernorm.scale": "",
    "mixer.dense_projection._extra_state": "",
    "mixer.q_proj.weight": "",
    "mixer.k_proj.weight": "",
    "mixer.v_proj.weight": "",
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
    "pre_mlp_layernorm.weight": "post_norm.scale",
    #
    "mlp.gate_proj.weight": "mlp.l1.weight",
    "mlp.up_proj.weight": "mlp.l2.weight",
    "mlp.down_proj.weight": "mlp.l3.weight",
    # misc
    "final_linear.weight": "word_embeddings.weight",
    "norm.weight": "norm.scale",
    #mlp
}


KEY_UPDATE_DICT_EMBEDDING = {
    "word_embeddings.weight": "embedding_layer.weight",
}

KEY_UPDATE_DICT_NORM = {
    "norm.scale": "norm.scale",
    "norm.weight": "norm.scale",
}

def remove_state_dict_prefixes(state_dict):
    for k in list(state_dict.keys()):
        if k.startswith("module."):
            state_dict[k[7:]] = state_dict.pop(k)
        elif k.startswith("sequential."):
            state_dict[k[10:]] = state_dict.pop(k)
    return state_dict


def detect_module_cls(module):
    print(f"Module: {module}")
    if isinstance(module, ParallelBlockPipe):
        return module.operator_type
    elif isinstance(module, EmbeddingPipe):
        return "embedding"
    elif isinstance(module, NormPipe):
        return "norm"
    elif isinstance(module, Lambda):
        return "lambda"
    else:
        print(f"Module {module} is not a ParallelBlockPipe, defaulting to attention")
        return "attention"

def convert_module_state_dict(state_dict, module):
    """
    Convert a pretrained savanna checkpoint state_dict to stripedhyena format

    Note:
    - Keys are replaced according to handrolled mapping
    - QKV weights are concatenated into a single tensor
    - Modal parametrization converted to pole / residue form
    """
    operator_type = detect_module_cls(module)
    print(f"Operator type: {operator_type}")
    if operator_type == "hyena":
        KEY_UPDATE_DICT = KEY_UPDATE_DICT_HYENA
    elif operator_type == "hyena_medium_conv":
        KEY_UPDATE_DICT = KEY_UPDATE_DICT_HYENA
    elif operator_type == "hyena_short_conv":
        KEY_UPDATE_DICT = KEY_UPDATE_DICT_HYENA
    elif operator_type == "embedding":
        KEY_UPDATE_DICT = KEY_UPDATE_DICT_EMBEDDING
    elif operator_type == "norm":
        KEY_UPDATE_DICT = KEY_UPDATE_DICT_NORM
    elif operator_type == "lambda":
        return state_dict
    else:
        KEY_UPDATE_DICT = KEY_UPDATE_DICT_ATTENTION

    params_to_merge = ["attention.q_proj.weight", "attention.k_proj.weight", "attention.v_proj.weight"]
    new_state_dict = OrderedDict()
    params_to_merge_values = []

    state_dict = remove_state_dict_prefixes(state_dict)
    
    print(f"State dict keys: {state_dict.keys()}")

    for k in state_dict.keys():

        if k in params_to_merge:
            print("merging params for k:", k)
            params_to_merge_values.append(state_dict[k])

        print(f"Pre update: {k}")
        new_k = KEY_UPDATE_DICT.get(k, k)
        print(f"Post update: {new_k}")

        if new_k != "":
            # Check if the value is a tensor before trying to access .shape #Changed
            if hasattr(state_dict[k], 'shape'):
                print(state_dict[k].shape)
                if "filter.short_filter_weight" in new_k:
                    new_state_dict[new_k] = state_dict[k][:,None]
                else:
                    new_state_dict[new_k] = state_dict[k]
            else:
                print(f"{k}: {type(state_dict[k])} (no shape attribute)")

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

    # We extract pole and residue representation for long hyena, medium and short use h instead
    if operator_type == "hyena" or operator_type == "hyena_medium_conv":

        if operator_type == "hyena":
            p = state_dict["mixer.mixer.filter.p"].reshape(4096, 16).to(torch.float32)
            R = state_dict["mixer.mixer.filter.R"].reshape(4096, 16).to(torch.float32)
            
            new_state_dict.pop("mixer.mixer.filter.p")
            new_state_dict.pop("mixer.mixer.filter.R")

            gamma = state_dict["mixer.mixer.filter.gamma"].to(torch.float32)

            logp = -torch.exp(p)
            logp = (logp * torch.exp(gamma))[..., None]


            new_state_dict["filter.log_poles"] = logp
            new_state_dict["filter.residues"] = R

            # @mp: code for previous parametrizations of long hyena
            # new_state_dict["filter.poles"] = new_state_dict["mixer.mixer.filter.p"].reshape(4096, 8, 1, 2)
            # new_state_dict["filter.residues"] = new_state_dict["mixer.mixer.filter.R"].reshape(4096, 8, 1, 2)

            # new_state_dict.pop("mixer.mixer.filter.p")
            # new_state_dict.pop("mixer.mixer.filter.R")

            # print(f"filter.poles shape: {new_state_dict['filter.poles'].shape}")
            # print(f"filter.residues shape: {new_state_dict['filter.residues'].shape}")
            
        elif operator_type == "hyena_medium_conv":
            # print(dir(state_dict))
            # parametrization in medium_conv matches! need to slice h with hyena_medium_conv_len
            h = state_dict["mixer.mixer.filter.h"]
            decay = state_dict["mixer.mixer.filter.decay"]
            L = module.mixer.mixer.hyena_medium_conv_len
            print(f"h shape: {h.shape}")
            h = h[:, :L] * decay[:, :L]
            new_state_dict["filter.h"] = h.unsqueeze(1)
        
    elif operator_type == "hyena_short_conv":
        print(state_dict["mixer.mixer.short_conv.short_conv_weight"].shape)
        h = state_dict["mixer.mixer.short_conv.short_conv_weight"]
        print(f"h shape: {h.shape}")
        new_state_dict["filter.h"] = h

    return new_state_dict


def checkpoint_conversion(checkpoint_path, new_checkpoint_path):
    # loads checkpoint in deepspeed format ("layer-{idx}-model_00-model_states.pt")
    # assumes model parallel 1
    "Deprecated: we use deepspeed raw checkpoint loading to convert the entire model"
    files = glob.glob(os.path.join(checkpoint_path, "layer*states.pt"))
    files = sorted(files, key=lambda x: int(x.split("/")[-1].split("_")[1].split("-")[0]))
    for idx, file in enumerate(files):
        state_dict = torch.load(file)
        print(f"Loading {file}, keys: {state_dict.keys()}", end="\n\n")

        state_dict = convert_module_state_dict(state_dict)

        new_file = file.split("/")[-1]
        torch.save(state_dict, os.path.join(new_checkpoint_path, f"layer_{idx:02d}.pt"))



if __name__ == "__main__":

    # config_path = "/home/zymrael/workspace/savanna/configs/model/evo2/7b_2t.yml"
    # option to either pass here or in config
    # checkpoint_path = "/scratch/hielab/brianhie/checkpoint/evo2/7b_13h_8m_8s_3a_cascade15"
    # new_checkpoint_path = "/home/zymrael/checkpoints/evo2/7b_13h_8m_8s_3a_cascade15_inference"
    # iteration = 457_500
    checkpoint_path = '/scratch/hielab/brianhie/checkpoint/evo2/7b-ablations-n32/7b_stripedhyena2_base_4M_resume/202410210618'
    config_path='/scratch/hielab/brianhie/checkpoint/evo2/7b-ablations-n32/7b_stripedhyena2_base_4M_resume/202410210618/global_step205000/configs/7b_stripedhyena2_base_4M_resume.yml'
    iteration = 205_000
    new_checkpoint_path = '/home/zymrael/checkpoints/evo2/7b-ablations-n32/7b_stripedhyena2_base_4M_resume/'   #'/scratch/hielab/gbrixi/evo2/nvidia/7b_stripedhyena2_base_4M_resume/'

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["SLURM_NTASKS"] = "1"
    os.environ["SLURM_NTASKS_PER_NODE"] = "1"
    os.environ["GLOBAL_NUM_GPUS"] = "1"

    model, tokenizer = load_savanna_checkpoint(
        config_path,
        checkpoint_path=checkpoint_path,
        iteration=iteration,
    )

    sequential = model.sequential


    device = sequential[0].word_embeddings.weight.device

    inputs = tokenizer.tokenize("ACTGACTGACTGACTG")
    inputs = torch.tensor(inputs)[None].to(device).long()

    outputs = sequential((inputs, None, None))
    
    print(inputs, inputs.shape)
    print(outputs, outputs.shape)

    os.makedirs(new_checkpoint_path, exist_ok=True)
    torch.save(outputs, os.path.join(new_checkpoint_path, "logits_test.pt"))
    
    new_state_dict = OrderedDict()
    layer_counter = 0
    for idx, module in enumerate(sequential):
        converted_state_dict = convert_module_state_dict(module.state_dict(), module)

        if isinstance(module, ParallelBlockPipe):
            for k, v in converted_state_dict.items():
                new_state_dict[f"blocks.{layer_counter}.{k}"] = v
            layer_counter += 1
        elif isinstance(module, EmbeddingPipe):
            for k, v in converted_state_dict.items():
                new_state_dict[f"{k}"] = v
                new_state_dict["unembed.weight"] = v
        elif isinstance(module, NormPipe):
            for k, v in converted_state_dict.items():
                new_state_dict[f"{k}"] = v
  
    
    print(new_state_dict.keys())
    checkpoint_file = f"iter_{iteration}.pt"
    torch.save(new_state_dict, os.path.join(new_checkpoint_path, checkpoint_file))