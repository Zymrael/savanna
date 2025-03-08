# Generate Context Extension Configs


import argparse
import math
import os
from pathlib import Path

from ruamel.yaml import YAML

yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)

CURRENT_DIR = Path(__file__).resolve().parent
CONTEXT_CONFIGS_DIR = CURRENT_DIR.parent
SAVANNA_ROOT = CONTEXT_CONFIGS_DIR.parent.parent
DEFAULT_CONFIG = CONTEXT_CONFIGS_DIR / "model_configs" / "7b_stripedhyena2_base_4M_32k.yml"
DEFAULT_OUTPUT_DIR = CONTEXT_CONFIGS_DIR / "model_configs" / "generated"
DEFAULT_CONTEXT_LENS = [5, 6, 7]
KILO = 2**10
ROPE_BASE_KEY = "rotary_emb_base"
ROPE_SCALING_KEY = "rotary_emb_scaling_factor"
POS_EMB_KEY = "pos_emb"
SEQLEN_KEY = "seq_length"
CONTEXT_LEN_BASE = 8 * KILO
DEFAULT_GENERATED_PREFIX = "7b"
DEFAULT_ROPE_BASE = 10000
DEFAULT_ROPE_SCALING_FACTOR = 1.0
ROPE_SCALES = ["linear", "log", "evo1", "5x"]

# Model Config
OPERATOR_CONFIG_KEY = "operator-config"
RING_ATTN_KEY = "ring"
MP_KEY = "model_parallel_size"
CONTEXT_PARALLEL_KEY = "context_parallel_size"
MBS_KEY = "train_micro_batch_size_per_gpu"
AC_KEY = "checkpoint-num-layers"  # activation checkpointing
TRAIN_ITERS_KEY = "train-iters"
LR_DECAY_ITERS_KEY = "lr-decay-iters"
ASYNC_SAVE_KEY = "async_save"
DEFAULT_ASYNC_SAVE = False
SAVE_RETAIN_INTERVAL_KEY = "save_retain_interval"
CHECKPOINT_KEY = "load"
DEFAULT_TRAIN_ITERS = 12500
DEFAULT_SAVE_RETAIN_INTERVAL = 12500
CHECKPOINT_32K = (
    "/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/7b_base_evo2_converted/interleaved/32K/MP2/"
)
CHECKPOINT_64K = "/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/7b-context-extension-v3/hybrid-log_evo1/64K/MP8/"
CHECKPOINT_128K = "/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/7b-context-extension-v3/hybrid-log_evo1/128K/MP8/"
CHECKPOINT_8K_to_128K = (
    "/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/7b_base_evo2_converted/interleaved/128K/MP8"
)
CHECKPOINT_256K = "/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/7b-context-extension-v3/hybrid-log_evo1/256K/MP16/"
DEFAULT_CHECKPOINT = "XXX"


def parse_int_list(str_list):
    return [int(item) for item in str_list.split(",")]


def generate_context_len(context_len: int):
    return 2**context_len * KILO


def generate_rope_base(
    rope_base: int, context_len: int, rope_scale: str, context_base_len: int = CONTEXT_LEN_BASE
):
    multiplier = context_len // context_base_len
    if rope_scale == "linear":
        return multiplier * rope_base
    elif rope_scale == "log":
        multiplier = int(math.log(multiplier, 2))
        return (10**multiplier) * rope_base
    elif rope_scale == "5x":
        multiplier = int(math.log(multiplier, 2))
        return (5**multiplier) * rope_base
    elif rope_scale == "evo1":
        return DEFAULT_ROPE_BASE
    else:
        raise ValueError(f"Invalid RoPE scale: {rope_scale}")


def generate_rope_scaling_factor(
    rope_base: int, context_len: int, rope_scale: str, context_base_len: int = CONTEXT_LEN_BASE
):
    multiplier = context_len // context_base_len
    if rope_scale == "evo1":
        return multiplier
    return DEFAULT_ROPE_SCALING_FACTOR


def read_config(config_path: Path):
    with open(config_path, "r") as f:
        return yaml.load(f)


def generate_output_path(
    output_dir: Path,
    rope_scale: str,
    context_len: int,
    base_name: str = DEFAULT_GENERATED_PREFIX,
    subdir: str = None,
):
    suffix = f"{rope_scale}-{context_len}K"
    filename = f"{base_name}-{suffix}.yml"
    output_dir = output_dir / f"{context_len}K"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if subdir is not None:
        output_dir = output_dir / subdir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    return output_dir / filename

def add_ring_attention(config: dict):
    operator_config  = config.get(OPERATOR_CONFIG_KEY, None)
    assert operator_config is not None, "Operator config not found"
    for i in range(len(operator_config)):
        [op_type], num_ops = operator_config[i]
        assert num_ops == 1, "Only one op per layer should be provided"
        if 'flash' in op_type:
            operator_config[i][0][0] = RING_ATTN_KEY
    return config

def generate_config(
    context_len: list[int],
    base_config: dict,
    rope_scale: str,
    output_path: Path,
    rotary_emb_base: int = DEFAULT_ROPE_BASE,
    rotary_emb_scaling_factor: float = DEFAULT_ROPE_SCALING_FACTOR,
    **kwargs,
):

    config = base_config.copy()

    config[ROPE_BASE_KEY] = rotary_emb_base
    config[ROPE_SCALING_KEY] = rotary_emb_scaling_factor

    # Update position embeddings
    if rope_scale == "evo1" or "hybrid" in rope_scale:
        pos_emb = "rotary_linear_scaled"
    else:
        pos_emb = "rotary"
    config[POS_EMB_KEY] = pos_emb

    # Seq length
    config[SEQLEN_KEY] = context_len

    # Update parallelism and mbs
    if context_len == 32 * KILO:
        config[MBS_KEY] = 1
        config[MP_KEY] = 2
        config[AC_KEY] = 4
    elif context_len == 64 * KILO:
        config[MBS_KEY] = 2
        config[MP_KEY] = 8
        config[AC_KEY] = 4
    elif context_len == 128 * KILO:
        config[MBS_KEY] = 1
        config[MP_KEY] = 8
        config[AC_KEY] = 2
    elif context_len == 256 * KILO:
        config[MBS_KEY] = 1
        config[MP_KEY] = 16
        config[AC_KEY] = 2
        config[CONTEXT_PARALLEL_KEY] = 1
        config[CHECKPOINT_KEY] = CHECKPOINT_256K
    print(
        f"Generating config for context length {context_len}:\n MBS: {config[MBS_KEY]} MP_SIZE: {config[MP_KEY]} CP_SIZE: {config[CONTEXT_PARALLEL_KEY]}\n {rope_scale=} {pos_emb=} {rotary_emb_base=} {rotary_emb_scaling_factor=} {kwargs=}"
    )
    
    for k, v in kwargs.items():
        if k in config:
            print(f" -> Overwriting {k} with {v}")
            config[k] = v
    context_parallel_size = config.get(CONTEXT_PARALLEL_KEY, None)
    if context_parallel_size is not None and context_parallel_size > 1:
        config = add_ring_attention(config)
        
    with open(output_path, "w") as f:
        yaml.dump(config, f)
    print(f"Generated config at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate model configs based on a template config",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base_config", type=Path, default=DEFAULT_CONFIG, help="Path to the base model config file"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Path to the output directory"
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=5,
#        choices=DEFAULT_CONTEXT_LENS,
        help="Context length as power of 2.  E.g., 5,6,7 -> 32,64,128",
    )
    parser.add_argument(
        "--rope_scale",
        type=str,
        #choices=ROPE_SCALES + ["all", "v3", "v3-8K_128K"],
        default=None,
        help="Rope scale for the context length",
    )
    parser.add_argument(
        "--context_base_len", type=int, default=CONTEXT_LEN_BASE, help="Base length of the context"
    )
    parser.add_argument("--rope_base", type=int, default=DEFAULT_ROPE_BASE, help="RoPE base frequency")
    parser.add_argument(
        "--prefix", type=str, default=DEFAULT_GENERATED_PREFIX, help="Prefix for the generated config files"
    )
    parser.add_argument(
        "--no_clean_generated_dir",
        action="store_true",
        help="Clear the generated directory before generating new configs",
    )
    args = parser.parse_args()

    base_config = read_config(args.base_config)

    # Get base RoPE freq
    if args.rope_base is None:
        args.rope_base = base_config.get(ROPE_BASE_KEY, None)
    if args.rope_base is None:
        raise ValueError(f"RoPE base not found in the base config or provided as an argument")
    # print(f"Using RoPE base: {args.rope_base}")

    # Generate actual context lens
    context_len = generate_context_len(args.context_length)
    if args.rope_scale == "v3-128K":
        context_len = 128 * KILO

    # print(f"Generating model configs for context length: {context_len}")

    # Clear the output directory
    # if not args.no_clean_generated_dir:
    #     shutil.rmtree(args.output_dir / context_len, ignore_errors=True)
    #     print(f"Cleaned the output directory: {args.output_dir}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    extra_kwargs = {}
   # extra_kwargs[CHECKPOINT_KEY] = DEFAULT_CHECKPOINT
    extra_kwargs[ASYNC_SAVE_KEY] = DEFAULT_ASYNC_SAVE
    extra_kwargs[SAVE_RETAIN_INTERVAL_KEY] = DEFAULT_SAVE_RETAIN_INTERVAL

    if args.rope_scale == "all":
        rope_scales = ROPE_SCALES
    elif args.rope_scale == "v3":
        rope_scales = ["hybrid-log"]
        if context_len == 32 * KILO:
            extra_kwargs[CHECKPOINT_KEY] = CHECKPOINT_32K
        elif context_len == 64 * KILO:
            extra_kwargs[CHECKPOINT_KEY] = CHECKPOINT_64K
        elif context_len == 128 * KILO:
            extra_kwargs[CHECKPOINT_KEY] = CHECKPOINT_128K
    elif args.rope_scale == "v3-8K_128K":
        rope_scales = ["log", "hybrid-log"]
        extra_kwargs[CHECKPOINT_KEY] = CHECKPOINT_8K_to_128K
    else:
        rope_scales = [args.rope_scale]

    #Match beginning of string with v1, v2, v3, etc. and assign as subdir
    # it no match, no subdir
    import re
    pattern = r"v\d"
    if re.match(pattern, args.rope_scale):
        subdir = args.rope_scale
    else:
        subdir = None
    #subdir = args.rope_scale if "v3" in args.rope_scale else "v1"

    for rope_scale in rope_scales:
        if args.rope_scale == "v3-8K_128K":
            context_len = 128 * KILO
            extra_kwargs[TRAIN_ITERS_KEY] = DEFAULT_TRAIN_ITERS * 3
            extra_kwargs[LR_DECAY_ITERS_KEY] = DEFAULT_TRAIN_ITERS * 3

        if "hybrid" in rope_scale:
            scale = rope_scale.split("-")[1]
            rope_scaling_factor = generate_rope_scaling_factor(
                args.rope_base, context_len, "evo1", args.context_base_len
            )
            rope_base = generate_rope_base(args.rope_base, context_len, scale, args.context_base_len)
            scale_label = f"{rope_scale}_evo1"
            output_path = generate_output_path(
                args.output_dir, scale_label, context_len // KILO, subdir=subdir
            )

            generate_config(
                context_len,
                base_config,
                scale_label,
                output_path,
                rotary_emb_base=rope_base,
                rotary_emb_scaling_factor=rope_scaling_factor,
                **extra_kwargs,
            )

        else:
            rope_base = generate_rope_base(args.rope_base, context_len, rope_scale, args.context_base_len)
            rope_scaling_factor = generate_rope_scaling_factor(
                args.rope_base, context_len, rope_scale, args.context_base_len
            )
            output_path = generate_output_path(
                args.output_dir, rope_scale, context_len // KILO, subdir=subdir
            )

            generate_config(
                context_len,
                base_config,
                rope_scale,
                output_path,
                rotary_emb_base=rope_base,
                rotary_emb_scaling_factor=rope_scaling_factor,
                **extra_kwargs,
            )
