import logging
import pathlib
import traceback
from datetime import datetime

import deepspeed
import savanna.training as megatron_train
import savanna.utils as megatron_utils
import torch
from attrdict import AttrMap
from det_utils import (
    EarlyStoppingCallback,
    EvalHarness,
    LMReducers,
    TensorboardWriter,
    get_global_config,
)
from savanna import mpu
from savanna.checkpointing import load_checkpoint, save_checkpoint
from savanna.data.data_utils import build_datasets_from_global_config

from determined import LOG_FORMAT, InvalidHP
from determined.pytorch import DataLoader
from determined.pytorch.deepspeed import (
    DeepSpeedTrial,
    DeepSpeedTrialContext,
    ModelParallelUnit,
)

import wandb

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


class GPT2Trial(DeepSpeedTrial):
    def __init__(self, context: DeepSpeedTrialContext) -> None:
        """
        Trial classes contain timers, model, and collect metrics.
        These work similarly to LightningModules.
        """
        self.context = context
        self.exp_config = self.context.get_experiment_config()
        self.args = AttrMap(self.context.get_hparams())

        # Initalize and get arguments, timers, and Tensorboard writer.
        try:
            self.global_config = get_global_config(self.context)
        except:
            traceback.print_exc()
            raise InvalidHP("Could not parse global_config.")
        logging.info(self.global_config)
        self.writer = self.context.get_tensorboard_writer()
        self.global_config.tensorboard_writer = self.writer
        self.global_config.configure_distributed_args()
        # The tokenizer needs to be built before model initialization in order to set the
        # required padded_vocab_size argument.
        self.global_config.build_tokenizer()
        megatron_train.initialize_megatron(global_config=self.global_config)
        wandb.init(project="hessian-hyena", config=self.global_config)
        self.timers = megatron_utils.Timers(
            use_wandb=True, tensorboard_writer=self.global_config.tensorboard_writer
        )

        # Model, optimizer, and learning rate.
        self.timers("model and optimizer").start()
        with deepspeed.zero.Init(
            enabled=self.global_config.zero_optimization["stage"] == 3
        ):
            (
                model,
                self.optimizer,
                self.lr_scheduler,
            ) = megatron_train.setup_model_and_optimizer(
                global_config=self.global_config
            )
        self.model = self.context.wrap_model_engine(model)
        self.context.set_mpu(
            ModelParallelUnit(
                mpu.get_data_parallel_rank(),
                mpu.get_data_parallel_world_size(),
                should_report_metrics=True,
                should_build_data_loader=self.should_build_data_loader(),
            )
        )
        self.timers("model and optimizer").stop()

        # Print setup timing.
        megatron_utils.print_rank_0("done with setups ...")
        self.timers.log(["model and optimizer"])
        megatron_utils.print_rank_0("training ...")

        # For tracking.
        if not self.args.search_world_size:
            self.reducer = self.context.wrap_reducer(
                LMReducers(self.global_config), for_training=False, for_validation=True
            )
        self.report_memory_flag = True
        self.total_train_loss_dict = {}
        self.total_val_loss_dict = {}
        self.tflops = 0
        self.reported_flops = False
        self.noise_scale_logger = megatron_utils.get_noise_scale_logger(
            self.global_config
        )
        self.timers("interval time").start()

    def should_build_data_loader(self):
        if self.global_config.is_pipe_parallel:
            is_first_stage = mpu.get_pipe_parallel_rank() == 0
            is_last_stage = (
                mpu.get_pipe_parallel_rank() == mpu.get_pipe_parallel_world_size() - 1
            )
            pipe_load = is_first_stage or is_last_stage
        else:
            pipe_load = True
        return mpu.get_model_parallel_rank() == 0 and pipe_load

    def build_callbacks(self):
        callbacks = {"tb": TensorboardWriter(self.writer)}
        if self.global_config.eval_tasks:
            callbacks["eval_tasks"] = EvalHarness(
                self.model, megatron_train.forward_step, self.global_config
            )
        if self.args.search_world_size:
            callbacks["early_stopping"] = EarlyStoppingCallback(self)
        return callbacks

    def train_batch(self, data_iterator, epoch_idx, batch_idx):
        if self.global_config.is_pipe_parallel:
            reduced_loss = megatron_train.train_step_pipe(
                global_config=self.global_config,
                timers=self.timers,
                model=self.model,
                data_iterator=data_iterator,
            )
        else:
            losses = []
            for _ in range(self.global_config.gradient_accumulation_steps):
                self.timers("forward").start()
                loss = megatron_train.forward_step(
                    global_config=self.global_config,
                    timers=self.timers,
                    data_iterator=data_iterator,
                    model=self.model,
                )
                self.timers("forward").stop()
                losses.append(loss)
                # Calculate gradients, reduce across processes, and clip.
                self.timers("backward").start()
                megatron_train.backward_step(
                    global_config=self.global_config,
                    timers=self.timers,
                    optimizer=self.optimizer,
                    model=self.model,
                    loss=loss,
                )
                self.timers("backward").stop()
                # Update parameters.
                self.timers("optimizer").start()
                if self.global_config.deepspeed:
                    self.model.step()
                else:
                    raise ValueError("Must be using deepspeed to run neox")
                self.timers("optimizer").stop()
            reduced_loss = {"lm_loss": megatron_utils.reduce_losses(losses).mean()}
        self.global_config.iteration += 1

        if self.global_config.log_gradient_noise_scale:  # log noise scale if applicable
            self.noise_scale_logger.update()

        # get learning rate (if present) - if doing soft prompt tuning + pipe parallel, you
        # may have no tunable parameters on a specific rank
        if self.optimizer.param_groups:
            lr = self.optimizer.param_groups[0].get("lr", 0)
        else:
            lr = 0

        # Logging.
        self.report_memory_flag = megatron_train.training_log(
            global_config=self.global_config,
            timers=self.timers,
            loss_dict=reduced_loss,
            total_loss_dict=self.total_train_loss_dict,
            learning_rate=lr,
            iteration=self.global_config.iteration,
            loss_scale=self.optimizer.cur_scale
            if self.global_config.precision == "fp16"
            else None,
            report_memory_flag=self.report_memory_flag,
            model=self.model,
            skipped_iter=0,
            optimizer=self.optimizer,
            noise_scale_logger=self.noise_scale_logger,
        )
        # if (
        #     additional_metrics is not None
        #     and additional_metrics["num_nans"] == 0
        #     and additional_metrics["num_skipped"] == 0
        # ):
        #     self.tflops = additional_metrics["flops_per_sec_per_gpu"] / 10**12

        if (
            self.global_config.exit_interval
            and self.global_config.iteration % self.global_config.exit_interval == 0
        ):
            torch.distributed.barrier()
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            megatron_utils.print_rank_0(
                "time: {} | exiting the program at iteration {}".format(
                    time_str, self.global_config.iteration
                )
            )
            self.context.set_stop_requested(True)
        return reduced_loss

    def evaluate_batch(self, data_iterator, batch_idx):
        """
        Calculate validation metrics for a batch and return them as a dictionary.
        This method is not necessary if the user defines evaluate_full_dataset().
        """
        if self.args.search_world_size:
            if self.tflops > 0:
                self.reported_flops = True
            return {"tflops": self.tflops}

        if data_iterator is not None:
            if self.global_config.char_level_ppl:
                data_iterator = megatron_utils.CharCounter(
                    data_iterator, self.global_config.tokenizer
                )

        loss = megatron_train.forward_step(
            model=self.model,
            data_iterator=data_iterator,
            global_config=self.global_config,
            timers=self.timers,
        )

        if data_iterator is not None:
            if self.global_config.char_level_ppl:
                self.reducer.update(
                    loss.item(), data_iterator.token_count, data_iterator.char_count
                )
            else:
                self.reducer.update(loss.item())

        if self.global_config.deepspeed and self.global_config.checkpoint_activations:
            deepspeed.checkpointing.reset()

        return {"lm_loss": loss}

    def build_training_data_loader(self):
        # Data stuff.
        self.timers("train/valid/test data dataset").start()
        (
            self.train_data,
            self.valid_data,
            self.test_data,
        ) = build_datasets_from_global_config(self.global_config)
        self.timers("train/valid/test data dataset").stop()
        self.timers.log(["train/valid/test data dataset"])
        return DataLoader(
            self.train_data,
            batch_size=self.global_config.train_micro_batch_size_per_gpu,
            shuffle=True,
            num_workers=self.global_config.num_workers,
            drop_last=True,
            pin_memory=False,
        )

    def build_validation_data_loader(self):
        return DataLoader(
            self.valid_data,
            batch_size=self.global_config.train_micro_batch_size_per_gpu,
            num_workers=self.global_config.num_workers,
            drop_last=True,
            pin_memory=False,
        )

    def save(self, context: DeepSpeedTrialContext, path: pathlib.Path) -> None:
        self.global_config.save = str(path)
        save_checkpoint(
            global_config=self.global_config,
            iteration=self.global_config.iteration,
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
        )

    def load(self, context: DeepSpeedTrialContext, path: pathlib.Path) -> None:
        self.global_config.load = str(path)
        self.global_config.iteration = load_checkpoint(
            global_config=self.global_config,
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            inference=False,
        )
        megatron_utils.print_rank_0(
            f"Loading checkpoint and starting from iteration {self.global_config.iteration}"
        )
