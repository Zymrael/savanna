## Kernel Tests

### TLDR
Primary testing is done in `test_fwd.py` and `test_bwd.py`.  

Other tests are for smaller device kernels and other utilities used by the primary kernels.

### Overview

- `test_fwd.py` tests forward triton kernel
- `test_bwd.py` has 3 test suites:
    - `test_two_pass_bwd` - unit test for backwards kernel
    - `test_two_pass_fwd_bwd - unit test for forwards kernel followed by backwards kernel where intermediate activations are manually passed to backwards.
    - `test_interface` - e2e test for `torch.nn.autograd.Function` `cgcg.interface.two_pass_chunked_gate_conv_gate`.

### Usage
Two utility scripts for easier testing given number of test configurations especially for `bwd` tests:
- `test_runner.sh`:
    ```
    ./test_runner.sh test_bwd.py
    ```
    - Will run the tests first on GPU and then re-run any failed tests on CPU using triton interpreter.  
    - Note that TMA tests will be skipped when running in interpreter.
- `test_debugger.sh`:
    ```
    ./test_debugger.sh test_bwd.py`
    ```
    - Runs the tests then saves debugging outputs for post-hoc analysis
    - See `test_log_analysis.ipynb` for post-processing and analysis.
- `Correctness` - to run only float32 tests to verify algorithmic correctness:
    ```
    pytest -sv -k "float32" test_bwd.py
    ```