## Tests
`savanna/launcher/tests/{torch,deepspeed,srun}.launcher.test.sh` 
- This test script runs the pipeline with a "mock" train script `savanna/launcher/tests/check_args.py` that:
  - checks that `GlobalConfig` can be correctly instantiated
  - checks that distributed env vars are set (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`)
  - attempts to initiate torch distributed env `torch.distributed.init_process_group`, which completes only if the distributed env is set up correctly.
- Added debug messages and assertions in `train.py` to check that the distributed env: `RANK`, `LOCAL_RANK`, `WORLD_SIZE` are set correctly when using the distributed launcher.

`savanna/launcher/tests/deepspeed-rank-log.sh`
- This tests that per-rank logging works when using the `deepspeed` launcher
- More specifically, you should see a `rank_logs` folder in the output directory with files for `stdout/stderr` for each rank after running the test.