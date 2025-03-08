## Hyena MLP

### Description

Triton kernels that fuse the elementwise gating and permutation ops surrounding the main convolution in `hyena short conv / hyena mlp` while relying on either `causal_conv` or `F.conv1d` for the convolution.

Compared to the torch-native implementation, it ensures that all output tensors are contiguous and achieves speedups by fusing the shape manipulations with memory-bound gating kernel.

The kernels can be either manually configured or autotuned, which is automatically handled based on the arguments passed to the primary API `hyena_mlp.interface.hyena_mlp`.

Note that further speedups can likely be achieved by devising an efficient depthwise convolution triton implementation to fully fuse the entire layer (WIP).

### Usage
See `tests/test_interface.py` and `benchmarks/bench_hyena_mlp.py`, which contain additional documentation and examples of usage.