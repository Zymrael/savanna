import logging
import os

import numpy as np
from attrdict import AttrMap

# from eval_tasks.eval_adapter import run_eval_harness
from savanna.neox_arguments import GlobalConfig
from torch.utils.tensorboard import SummaryWriter

from determined.pytorch import MetricReducer, PyTorchCallback


def get_global_config(context):
    args = AttrMap(context.get_hparams())
    exp_config = context.get_experiment_config()

    # Gather overrides.
    overwrite_values = args.pop("overwrite_values", {})
    # We are going to overwrite certain global_config with determined config values
    # from the experiment config to ensure consistency.
    assert (
        "batches" in exp_config["searcher"]["max_length"]
    ), "Please specify max_length in batches."
    assert (
        "batches" in exp_config["min_validation_period"]
    ), "Please specify min_validation_period in batches."
    overwrite_values.update(
        {
            "checkpoint_factor": exp_config["min_validation_period"]["batches"],
            "eval_interval": exp_config["min_validation_period"]["batches"],
            "hostfile": os.environ.get("DET_DEEPSPEED_HOSTFILE_PATH"),
            "seed": context.env.trial_seed,
        }
    )
    for k, v in overwrite_values.items():
        logging.info(f"Setting global_config.{k} to {v}")
    print(args)
    global_config = GlobalConfig.consume_parsed_deepy_args(
        args, overwrite_values=overwrite_values
    )
    return global_config


class TensorboardWriter(PyTorchCallback):
    def __init__(self, writer: SummaryWriter):
        self.tb_writer = writer

    def on_validation_end(self, metrics):
        self.tb_writer.flush()

    def trial_cleanup(self) -> None:
        self.tb_writer.flush()
        self.tb_writer.close()


class EarlyStoppingCallback(PyTorchCallback):
    def __init__(self, trial):
        self.trial = trial

    def on_validation_start(self):
        if self.trial.reported_flops:
            self.trial.context.set_stop_requested(True)


class LMReducers(MetricReducer):
    def __init__(self, global_config):
        self.char_level_ppl = global_config.char_level_ppl
        self.token_count = 0
        self.char_count = 0
        self.lm_losses = []

    def update(self, lm_loss, token_count=None, char_count=None):
        self.lm_losses.append(lm_loss)
        if self.char_level_ppl:
            self.token_count += token_count
            self.char_count += char_count

    def reset(self):
        self.lm_losses = []
        self.token_count = 0
        self.char_count = 0

    def per_slot_reduce(self):
        return self.lm_losses, self.token_count, self.char_count

    def cross_slot_reduce(self, per_slot_metrics):
        lm_losses, token_count, char_count = zip(*per_slot_metrics)
        lm_losses = [item for sublist in lm_losses for item in sublist]

        metrics = {"lm_loss": np.mean(lm_losses)}
        metrics["lm_loss_ppl"] = np.exp(metrics["lm_loss"])
        if self.char_level_ppl:
            tokens_per_char = sum(token_count) / sum(char_count)
            metrics["lm_loss_char_lvl_ppl"] = np.exp(
                metrics["lm_loss"] * tokens_per_char
            )
        return metrics


class EvalHarness(PyTorchCallback):
    def __init__(self, model, forward_step_fn, global_config):
        self.model = model
        self.forward_step_fn = forward_step_fn
        self.global_config = global_config

    def on_validation_end(self, metrics):
        pass
        # TODO: This hangs with pipeline parallel.
        # metrics.update(
        #     run_eval_harness(
        #         self.model,
        #         self.forward_step_fn,
        #         self.global_config,
        #         eval_tasks=self.global_config.eval_tasks,
        #     )
        # )
