# Evo2 7b Context Extension

Documentation for various conversions needed to extend the 7B 8K pretrained model for longer context lengths.

## 8K -> 32K
1. Interleave 7b pretrained checkpoint
- Due to incorrect splitting of input tensors (`x1`, `x2`, and `v`) to hyena `mixer` layers for `model_parallel_size > 1`, a one time interleaving of the pretrained 7B tensors was needed for warm starting the model at a different `mp` size.  Note this will be needed for the 40B pretrained model as well. 
    - The [dense_projection](https://github.com/Zymrael/savanna/blob/137d34b50ad2015070906ca9f5412db083b7da6a/savanna/model/block.py#L586-L590) and [short_conv_proj](https://github.com/Zymrael/savanna/blob/137d34b50ad2015070906ca9f5412db083b7da6a/savanna/model/block.py#L595) weights directly preceding the hyena mixer operators in `ParallelSequenceMixer` were interleaved using this [script](https://github.com/Zymrael/savanna/blob/jeromeku/7b-context-ext/tools/7b_context_extension/conversion/interleave_model_states.py).
    - This was accompanied by a change in [block.py](https://github.com/Zymrael/savanna/blob/137d34b50ad2015070906ca9f5412db083b7da6a/savanna/model/block.py#L610-L613) that implemented the necessary tensor reshaping needed for correctly splitting input tensors to the hyena mixer operators across `mp` sizes.
    - The interleaved model checkpoint was then checked using this [script](https://github.com/Zymrael/savanna/blob/jeromeku/7b-context-ext/tools/7b_context_extension/checks/check_interleaved_model_states.py), which compares the original checkpoint with the interleaved checkpoint to confirm that the interleaved tensors were correctly rearranged.

2. Extend filter lengths for hyena medium layers
- The parametrized filter (`ExplicitDecayFilter`) used for evo2 hyena medium conv layers has [two parameters](https://github.com/Zymrael/savanna/blob/137d34b50ad2015070906ca9f5412db083b7da6a/savanna/model/operators/hyena/parametrization/explicit_filter.py#L27-L49) that are dependent on sequence length, though only use the first convolution length of these filters (`128` for evo2) are used.  When extending to new context lengths, these checkpointed params need to be manually extended to the target sequence length, otherwise the checkpoint will not be able to instantiated at the longer context length. 
    - The extension of `mixer.filter.h` and `mixer.decay` params of the parametrized filter was performed using this [script](https://github.com/Zymrael/savanna/blob/jeromeku/7b-context-ext/tools/7b_context_extension/conversion/extend_filter.py).
    - The extended filters were checked using this [script](https://github.com/Zymrael/savanna/blob/jeromeku/7b-context-ext/tools/7b_context_extension/checks/check_filter_lens.py).

3. Convert interleaved, extended model checkpoint to MP2
    - The 7b pretrained model was trained using `model_parallel_size=1`, and the context extension configs required `mp > 1`.  
    - Originally, `DeepSpeed`'s `universal checkpoint` was used to do this conversion.  However, due to divergent losses in the converted checkpoint at higher `mp` sizes, we resorted to a manual conversion script (h/t @bhie)
    - The [conversion script](https://github.com/Zymrael/savanna/blob/jeromeku/7b-context-ext/tools/7b_context_extension/conversion/convert_checkpoint_model_parallel_evo2.py) enables conversion of any model trained using `ZeRO-1` from any source `model_parallel_size` to a new target `model_parallel_size`. 
    - Note it only converts the `model_state` and not the `optimizer_states`, and thus, is intended for use only for warm starting / finetuning existing checkpoints.

Runs that confirm that losses are within range after these conversions [here](https://api.wandb.ai/links/hyena/30n4ejvs).  Also see this Slack [thread](https://hsu-laboratory.slack.com/archives/C07EZHB8W64/p1731181298611229?thread_ts=1731006753.001399&cid=C07EZHB8W64).

## 32K -> 64K
1. Start from RoPE-scale-specific 32K checkpoint
2. Merge from MP2 -> MP1
3. Extend filters from 32K -> 64K
4. Convert extended checkpoint from MP2 -> MP8
5. Repeat steps 1-5 from each RoPE scale 

## 64K -> 128K
Same as `32K -> 64K`, replacing with respective context lengths.