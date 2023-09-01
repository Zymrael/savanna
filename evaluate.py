"""Evaluation tasks - modified from https://github.com/EleutherAI/lm-evaluation-harness"""
import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from megatron.training import forward_step
from megatron.utils import setup_for_inference_or_eval, init_wandb
from megatron.logging import tb_wandb_log
from eval_tasks import run_eval_harness
from pprint import pprint
from datetime import datetime
import json


def main():
    model, global_config = setup_for_inference_or_eval(use_cache=False)
    results = run_eval_harness(
        model,
        forward_step,
        global_config,
        eval_tasks=global_config.eval_tasks,
        bootstrap_iters=10000,
    )
    if global_config.rank == 0:
        init_wandb(global_config=global_config)
        # log to wandb
        for k, v in results["results"].items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    k3 = "_".join([k, k2])
                    tb_wandb_log(
                        f"eval/{k3}",
                        v2,
                        global_config.iteration,
                        use_wandb=global_config.use_wandb,
                    )
            else:
                tb_wandb_log(
                    f"eval/{k}",
                    v,
                    global_config.iteration,
                    use_wandb=global_config.use_wandb,
                )

        pprint(results)
        results_path = (
            f'eval_results_{datetime.now().strftime("%m-%d-%Y-%H-%M-%S")}.json'
        )
        if global_config.eval_results_prefix:
            results_path = f"{global_config.eval_results_prefix}_{results_path}"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
