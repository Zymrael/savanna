
from functools import lru_cache

# Mapping from evo2 savanna model params to vortex (stripedhyena) model params
# IMPORTANT: ONLY tested on evo2 architecture

# Concatenate these params when mapping to new state dict
PARAMS_TO_MERGE = ["attention.q_proj.weight", "attention.k_proj.weight", "attention.v_proj.weight"]
MERGE_KEY = "inner_mha_cls.Wqkv.weight"

# Mapping from savanna.param -> vortex.param
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

KEY_UPDATE_DICT_EMBEDDING = {
    "word_embeddings.weight": "embedding_layer.weight",
}

KEY_UPDATE_DICT_NORM = {
    "norm.scale": "norm.scale",
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

# Add operator to expected params
#Add mapping from savanna.param -> vortex.param
SAVANNA_TO_VORTEX = {
        "hyena": KEY_UPDATE_DICT_HYENA,
        "embedding": KEY_UPDATE_DICT_EMBEDDING,
        "norm": KEY_UPDATE_DICT_NORM,
        "flash_v2": KEY_UPDATE_DICT_ATTENTION
}
def remove_empty_values(d: dict):
    return {k: v for k,v in d.items() if v != ""}

def get_savanna_to_vortex_map(remove_empty=True):
    clean_map = {}

    for operator_type, mapping in SAVANNA_TO_VORTEX.items():
        clean_map[operator_type] = remove_empty_values(mapping) if remove_empty else mapping
    return clean_map