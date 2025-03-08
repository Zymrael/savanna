import sys
import time
from pathlib import Path

from savanna.memory_stats import _make_alloc_keys, _make_count_keys

TEST_DIR = Path(__file__).parent.parent
SAVANNA_DIR = TEST_DIR.parent
TEST_CONFIGS_DIR = TEST_DIR / "test_configs"

sys.path.insert(0, str(SAVANNA_DIR))
from tests.utils import get_timestamp, get_wandb_run, run_program


def test_mem_stats_logging():
    model_config_path = TEST_CONFIGS_DIR / "mem_stats.yml"
    ts = get_timestamp()
    wandb_project = "functional_tests"
    wandb_group = f"mem_stats-test-{ts}"
    run_name = f"{ts}"
    run_program(
        model_config_path,
        wandb_project=wandb_project,
        wandb_group=wandb_group,
        wandb_run_name=run_name,
    )
    # wait for run to finish and wandb logs to be uploaded, alternatively, can run in offline mode
    time.sleep(10)
    run = get_wandb_run(wandb_project=wandb_project, wandb_run_name=run_name, summaries_only=True)
    logged_data_keys = run.columns
    
    _alloc_keys = _make_alloc_keys()
    _count_keys = _make_count_keys()

    assert all(k in logged_data_keys for k in _alloc_keys)
    assert all(k in logged_data_keys for k in _count_keys)
    
if __name__ == "__main__":
    test_mem_stats_logging()