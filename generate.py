#!/usr/bin/env python
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

from savanna.utils import print_rank_0, setup_for_inference_or_eval

from savanna.text_generation_utils import (
    generate_samples_input_from_file,
    generate_samples_from_prompt,
    generate_samples_unconditional,
    generate_samples_interactive,
)


def main():
    """
    Generate text/sample model
    """
    model, global_config = setup_for_inference_or_eval(use_cache=False)
    if global_config.recompute:
        model.module.inference_mode(
            use_cache=False
        )  # don't use kv cache if recomputing
    if global_config.text_gen_type == "unconditional":
        print_rank_0(
            f"Generating samples unconditionally and saving results to {global_config.sample_output_file}"
        )
        generate_samples_unconditional(
            global_config=global_config,
            model=model,
            number_of_samples=global_config.num_samples,
            output_file=global_config.sample_output_file,
            maximum_tokens=global_config.maximum_tokens,
            recompute=global_config.recompute,
            temperature=global_config.temperature,
            top_k=global_config.top_k,
            top_p=global_config.top_p,
        )

    elif global_config.text_gen_type == "input-file":
        print_rank_0(
            f"Generating samples from input file {global_config.sample_input_file}"
        )
        assert global_config.sample_input_file is not None
        generate_samples_input_from_file(
            global_config=global_config,
            model=model,
            input_file=global_config.sample_input_file,
            output_file=global_config.sample_output_file,
            maximum_tokens=global_config.maximum_tokens,
            prompt_end=global_config.prompt_end,
            recompute=global_config.recompute,
            temperature=global_config.temperature,
            top_k=global_config.top_k,
            top_p=global_config.top_p,
        )

    elif global_config.text_gen_type == "interactive":
        generate_samples_interactive(
            global_config=global_config,
            model=model,
            recompute=global_config.recompute,
            temperature=global_config.temperature,
            maximum_tokens=global_config.maximum_tokens,
            prompt_end=global_config.prompt_end,
            top_k=global_config.top_k,
            top_p=global_config.top_p,
        )

    else:
        raise ValueError(
            f"`text-gen-type` either not specified or not recognised: {global_config.text_gen_type}"
        )


if __name__ == "__main__":
    main()
