import argparse

import yaml

from savanna.utils import FP8_SHAPE, pad_to_multiple

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--multiple_of", type=int, default=128) # default 128       
    args = parser.parse_args()

    global_config = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)
    multiple_of = args.multiple_of
    hidden_size = global_config.get("hidden_size", global_config.get("hidden-size", None))
    assert hidden_size is not None, f"hidden_size not found in {args.config_path}"
    model_parallel_size = global_config.get("model_parallel_size", global_config.get("model-parallel-size", None))
    assert model_parallel_size is not None, f"model_parallel_size not found in {args.config_path}"
    
    ff_dim = int(2 * hidden_size * 4 / 3)
    ff_dim = multiple_of * ((ff_dim + multiple_of - 1) // multiple_of)
    # print(f"DEBUG::ParallelGLU::ff_dim:before padding {ff_dim}")
    ff_dim = pad_to_multiple(ff_dim, FP8_SHAPE[1] * model_parallel_size)
    # print(f"DEBUG::ParallelGLU::ff_dim:after padding {ff_dim}")


