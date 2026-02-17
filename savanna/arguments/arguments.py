import argparse
import base64
import copy
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from socket import gethostname
from typing import Dict, List

import torch
import yaml

try:
    from typing import Literal, Union
except ImportError:
    from typing_extensions import Literal, Union

import deepspeed.launcher.runner as runner

from savanna.logging import Tee
from savanna.tokenizer import build_tokenizer
from savanna.utils import expand_operator_types, obtain_resource_pool

from .deepspeed_args import GlobalConfigDeepspeedConfig, GlobalConfigDeepspeedRunner
from .global_config import (
    GlobalConfigLogging,
    GlobalConfigLRScheduler,
    GlobalConfigModel,
    GlobalConfigOptimizer,
    GlobalConfigOther,
    GlobalConfigParallelism,
    GlobalConfigProfiler,
    GlobalConfigTextgen,
    GlobalConfigTokenizer,
    GlobalConfigTraining,
    operator_type_CHOICES,
)
from .lora_config import GlobalConfigLoRA

ZERO_DEFAULTS = {
    "stage": 0,
    "allgather_partitions": True,
    "reduce_scatter": True,
    "allgather_bucket_size": int(5e8),
    "overlap_comm": False,
    "reduce_scatter": True,
    "reduce_bucket_size": int(5e8),
    "contiguous_gradients": False,
}

OPT_DEFAULT = "Adam"
OPT_PARAMS_DEFAULTS = {
    "lr": 0.001,
    "betas": [0.9, 0.999],
    "eps": 1.0e-8,
    "weight_decay": 0,
    "freeze_step": 400,
    "momentum": 0.0,
    "cuda_aware": False,
}


AUTOTUNING_ARGS = (
    "train_batch_size",
    "train_micro_batch_size_per_gpu",
    "gradient_accumulation_steps",
    "zero_optimization",
    "autotuning",
)

BASE_CLASSES = [
    GlobalConfigDeepspeedRunner,
    GlobalConfigDeepspeedConfig,
    GlobalConfigModel,
    GlobalConfigLRScheduler,
    GlobalConfigOptimizer,
    GlobalConfigTokenizer,
    GlobalConfigTraining,
    GlobalConfigParallelism,
    GlobalConfigLogging,
    GlobalConfigTextgen,
    GlobalConfigOther,
    GlobalConfigProfiler,
    GlobalConfigLoRA,
]

DEEPSPEED_ARG_CLASSES = [GlobalConfigDeepspeedRunner, GlobalConfigDeepspeedConfig]
NEOX_ARG_CLASSES = [i for i in BASE_CLASSES if i not in DEEPSPEED_ARG_CLASSES]

if "DLTS_HOSTFILE" in os.environ:
    DLTS_HOSTFILE = os.environ["DLTS_HOSTFILE"]


@dataclass
class GlobalConfig(*BASE_CLASSES):
    """
    data class containing all configurations

    GlobalConfig inherits from a number of small configuration classes
    """

    ############################################################################################################################
    # start of instantiation

    def __post_init__(self):
        """
        after initialization of default or loaded values
        a number of functions are performed in order to
        calculate values, assert consistency and do typechecking.
        """
        if not GlobalConfig.validate_keys():
            raise ValueError(
                self.__class__.__name__ + ".__post_init__() GlobalConfig keys cannot be validated"
            )

        self.enable_logging()

        self.calculate_derived()

        if not self.validate_types():
            raise ValueError(
                self.__class__.__name__ + ".__post_init__() GlobalConfig types cannot be validated"
            )

        if not self.validate_values():
            raise ValueError(
                self.__class__.__name__ + ".__post_init__() GlobalConfig values cannot be validated"
            )
        
    def build_tokenizer(self):
        self.tokenizer = build_tokenizer(self)

    def initialize_tensorboard_writer(self):
        if self.tensorboard_dir and self.rank == 0:
            try:
                from torch.utils.tensorboard import SummaryWriter

                print("> setting tensorboard ...")
                self.tensorboard_writer = SummaryWriter(log_dir=self.tensorboard_dir)
            except (ModuleNotFoundError, ImportError):
                print(
                    "WARNING: TensorBoard writing requested but is not "
                    "available (are you using PyTorch 1.1.0 or later and do you have tensorboard installed?), "
                    "no TensorBoard logs will be written.",
                    flush=True,
                )

    @classmethod
    def from_ymls(cls, paths_to_yml_files: List[str], overwrite_values: Dict = None):
        """
        instantiates GlobalConfig while reading values from yml files

        paths_to_yml_files: list of paths to yml files

        overwrite_values: If provided, overwrite any values in the yamls with these values
        """

        print(cls.__name__ + ".from_ymls() " + str(paths_to_yml_files), flush=True)

        # initialize an empty config dictionary to be filled by yamls
        config = dict()
        config_files = dict()
        # iterate of all to be loaded yaml files
        for conf_file_name in paths_to_yml_files:
            # load file
            with open(conf_file_name) as conf_file:
                conf = yaml.load(conf_file, Loader=yaml.FullLoader)

            # check for key duplicates and load values
            for conf_key, conf_value in conf.items():
                if conf_key in config:
                    raise ValueError(
                        f"Conf file {conf_file_name} has the following duplicate keys with previously loaded file: {conf_key}"
                    )

                conf_key_converted = conf_key.replace(
                    "-", "_"
                )  # TODO remove replace and update configuration files?
                config[conf_key_converted] = conf_value

            # load original config files to save unchanged with checkpoint
            # saving the original config retains comments
            filename = os.path.basename(conf_file_name)
            assert (
                filename not in config_files
            ), "At least two config files have the same filename. This will result in conflicts when saving out configs with the checkpoint in one single directory. Please use unique names for configs."
            config_files[filename] = open(conf_file_name).read()

        # add config file content to neox args to make them accessible in code
        # this is used when saving checkpoints
        config["config_files"] = config_files

        # Configuration parameters not specified
        params_not_in_config = sorted(list(set(cls.__dataclass_fields__.keys()) - set(config.keys())))
        if len(params_not_in_config) > 0:
            logging.debug(
                cls.__name__
                + ".from_ymls() Configuration parameters not specified (using defaults): "
                + ", ".join(params_not_in_config)
            )

        if overwrite_values is not None:
            for k, v in overwrite_values.items():
                config[k] = v

        # instantiate class and return
        # duplicate values and unrecognized keys are again checked upon instantiation
        return cls(**config)

    @classmethod
    def from_dict(cls, args_dict: Dict):
        """
        instantiates GlobalConfig while reading values from input dict
        """
        return cls(**args_dict)

    @classmethod
    def consume_parsed_deepy_args(cls, args, overwrite_values: Dict = None):
        conf_files = args["conf_file"]
        conf_dir = args["conf_dir"]
        paths_to_yml_files = []
        if conf_files is not None:
            for conf_file in conf_files:
                paths_to_yml_files.append(os.path.join(conf_dir, conf_file))

        # load args
        global_config = cls.from_ymls(
            paths_to_yml_files=paths_to_yml_files, overwrite_values=overwrite_values
        )
        return global_config

    ############################################################################################################################
    # start of command line args interface
    @classmethod
    def consume_deepy_args(cls, return_extra_args=False):
        """
        entry point for deepy.py configuring and consuming command line arguments.

        We can use `--wandb_group` / `--wandb_team` to overwrite those args from the command line, otherwise the value from the config is taken.

        @jeromeku
        Additional args for `srun` launcher:
        --hostlist: text file with hostnames, one per line, distinct from `hostfile`
        --train-args-output: output path for base encoded parsed train args
        """

        parser = argparse.ArgumentParser(description="GPT-NeoX Configuration", allow_abbrev=False)

        group = parser.add_argument_group(title="Training Configuration")

        group.add_argument(
            "user_script",
            type=str,
            help="User script to launch, followed by any required " "arguments.",
        )

        group.add_argument(
            "--conf_dir",
            "-d",
            type=str,
            default=None,
            help="Directory to prefix to all configuration file paths",
        )

        group.add_argument(
            "conf_file",
            type=str,
            nargs="+",
            help="Configuration file path. Multiple files can be provided and will be merged.",
        )

        group = parser.add_argument_group(title="Weights and Biases monitoring args")

        group.add_argument(
            "--wandb_project", type=str, help="Weights & Biases project name.", default="savanna"
        )

        group.add_argument(
            "--wandb_group",
            type=str,
            default=None,
            help='Weights & Biases group name - used to group together "runs".',
        )

        group.add_argument(
            "--wandb_run_name",
            type=str,
            default=None,
            help="Weights & Biases run name",
        )

        group.add_argument(
            "--wandb_team",
            type=str,
            default=None,
            help="Weights & Biases team name.",
        )

        group = parser.add_argument_group(title="Eval args")

        group.add_argument(
            "--eval_tasks",
            type=str,
            nargs="+",
            default=None,
            help="Optionally overwrite eval tasks to run for evaluate.py",
        )
        group.add_argument(
            "--iteration",
            type=int,
            default=None,
            help="Iteration to load checkpoint from in evaluate.py / generate.py. If None is provided, uses the latest iteration.",
        )
        group.add_argument(
            "--eval_results_prefix",
            type=str,
            default=None,
            help="prefix to append to eval results file",
        )
        parser.add_argument(
            "-H",
            "--hostfile",
            type=str,
            help="Hostfile path (in MPI style) that defines the "
            "resource pool available to the job (e.g., "
            "worker-0 slots=4)",
        )
        group = parser.add_argument_group(title="Generation args")
        group.add_argument(
            "-i",
            "--sample_input_file",
            type=str,
            default=None,
            help="Optionally overwrite `sample_input_file` for generate.py",
        )
        group.add_argument(
            "-o",
            "--sample_output_file",
            type=str,
            default=None,
            help="Optionally overwrite `sample_output_file` for generate.py",
        )

        tuning = parser.add_argument_group(title="DeepSpeed Autotuning")
        tuning.add_argument(
            "--autotuning",
            type=str,
            default=None,
            choices=("tune", "run"),
            help="Use DeepSpeed's autotuning feature to optimize certain hyperparameters. For more details refer to documentation here: https://www.deepspeed.ai/tutorials/autotuning/",
        )

        launcher = parser.add_argument_group(title="Launch args")
        launcher.add_argument(
            "--hostlist",
            type=str,
            default="hostlist",
            help="Path to of hostnames for launching",
        )
        launcher.add_argument(
            "--train-args-output",
            type=str,
            default="train_args.txt",
            help="Path to save train args output",
        )

        args_parsed = parser.parse_args()
        hostlist = args_parsed.hostlist
        train_args_output = args_parsed.train_args_output
        # Delete launcher args, otherwise will interfere with GlobalConfig parsing downstream
        delattr(args_parsed, "hostlist")
        delattr(args_parsed, "train_args_output")

        # Validate user_script exists
        assert os.path.exists(
            args_parsed.user_script
        ), f"User script could not be found: {args_parsed.user_script}"

        # load config files
        conf_files = args_parsed.conf_file
        if args_parsed.conf_dir:
            conf_files = [os.path.join(args_parsed.conf_dir, f) for f in conf_files]

        # enables us to pass in `125M` instead of `125M.yml`
        conf_files = [
            (cf if (cf.endswith(".yml") or cf.endswith(".json")) else cf + ".yml") for cf in conf_files
        ]

        # determine overwrite values
        overwrite_values = dict()
        for k, v in vars(args_parsed).items():
            if k == "autotuning" and v is not None:
                overwrite_values["autotuning_run"] = v
            elif k not in ["conf_dir", "conf_file"] and v is not None:
                overwrite_values[k] = v

        # load args
        global_config = cls.from_ymls(paths_to_yml_files=conf_files, overwrite_values=overwrite_values)

        if global_config.use_wandb:
            try:
                import wandb

                # Check if the W&B group name is configured
                if global_config.wandb_group is None:
                    # Set a randomized string as group name if no group name is provided
                    global_config.wandb_group = wandb.sdk.lib.runid.generate_id()
                else:
                    print(f"W&B group name: {global_config.wandb_group}")
                    # @jeromeku: I commented this out because I found it easier to organize the runs by group name (without attaching a randomized id)
                    # otherwise a random id is always attached to the user-provided group name
                #     # Concatenate the W&B group name with a randomized string to ensure uniqueness.
                #     global_config.wandb_group += "_" + wandb.sdk.lib.runid.generate_id()
            except ModuleNotFoundError as e:
                if e.name == "wandb":
                    e.msg += "\nWeights & Biases monitoring was requested but `wandb` was not found. Install `wandb` to use Weights & Biases, or set the `use_wandb` configuration option to a boolean false to disable Weights & Biases logging."
                raise e

        global_config.print()

        if return_extra_args:
            return args_parsed.conf_file, hostlist, train_args_output, global_config
        return global_config

    @classmethod
    def consume_global_config(cls, overwrite_values=None):
        """
        Deepspeed launcher needs to pass the arguments for `pretrain_gpt2.py` across to all machines.

        In order not to have any problems with different configs being mismatched across machines, we instead read the .yaml configuration file from the main rank,
        then serialize the arguments to a dictionary, which the deepspeed launcher broadcasts to all machines (`--megatron_config`).

        We then instantiate a new GlobalConfig from the dictionary (`.from_dict`). This should ensure args are never inconsistent across machines.
        """

        parser = argparse.ArgumentParser(description="GPT-NeoX Configuration", allow_abbrev=False)
        parser.add_argument(
            "--megatron_config",
            type=str,
            default=None,
            help="json dict dumped as string in GlobalConfig.get_deepspeed_main_args()",
        )
        parser.add_argument(
            "--deepspeed_config",
            type=str,
            default=None,
            help="Only need this (at this stage) for autotuning",
        )
        args_parsed, _ = parser.parse_known_args()
        megatron_config = json.loads(base64.urlsafe_b64decode(args_parsed.megatron_config).decode("utf-8"))
        if args_parsed.deepspeed_config is not None:
            overwrite_values = cls.set_up_autotuning(args_parsed.deepspeed_config, overwrite_values)
        if overwrite_values is not None:
            megatron_config.update(overwrite_values)
        return cls.from_dict(args_dict=megatron_config)

    @staticmethod
    def set_up_autotuning(encoded_config, overwrite_values):

        config = json.loads(base64.urlsafe_b64decode(encoded_config).decode("utf-8"))
        overwrite_values = overwrite_values if overwrite_values else {}
        for tuning_param in AUTOTUNING_ARGS:
            # TODO: This is for autotuning specifically, may cause surprises for someone with a weird setup
            if tuning_param in config:
                overwrite_values[tuning_param] = config[tuning_param]
        return overwrite_values

    @staticmethod
    def convert_key_value_to_command_line_arg(k, v):
        if isinstance(v, bool):
            if v:
                return [f"--{k}"]
            else:
                return []
        if v is None:
            return []
        return [f"--{k}", str(v)]

    def get_extra_deepspeed_args(self):
        """
        Sets up the extra arguments for deepspeed. This is done by reading in the `deepspeed_extra_args` dictionary from
            the configuration file, and then adding any arguments where values differ from those specified in the dataclass.
        """
        global_config = self.get_parent_class_value_dict(*self.__class__.__bases__, only_non_defaults=True)

        extra_ds_args = dict()

        for key, value in self.deepspeed_extra_args.items():
            # Check to make sure the key is not already changed from defaults, and raise an exception if it is
            # This is to prevent users from accidentally writing arguments both in deepspeed_extra_args and in the base level
            # of the configuration file
            if hasattr(global_config, key):
                raise ValueError(
                    f"Key {key} is already specified elsewhere. Reading in a different value from the 'deepspeed_extra_args' option in the configuration file will cause undefined behavior."
                )
            extra_ds_args[key] = value

        return extra_ds_args

    def get_deepspeed_main_args(self):
        args_list = list()

        if self.autotuning_run is not None:
            args_list.extend(self.convert_key_value_to_command_line_arg("autotuning", self.autotuning_run))

        # get deepspeed runner args, and only pass them in to deepspeed launcher if they differ from defaults
        for key, default_value in GlobalConfigDeepspeedRunner().defaults():
            if key == "autotuning_run":
                continue
            configured_value = getattr(self, key)

            # Only needed for deepspeed.launcher.runner, we are doing this manually
            # if key == "force_multi":
            #     if self.deepspeed_slurm or self.deepspeed_mpi:
            #         configured_value = True
            if configured_value != default_value:
                args_list.extend(self.convert_key_value_to_command_line_arg(key, configured_value))

        if self.deepspeed_slurm:
            comment = getattr(self, "comment")
            if comment:
                args_list.extend(self.convert_key_value_to_command_line_arg("comment", comment))
            # This is handled in config_parser
            # master_address = os.environ["SLURM_JOB_NODELIST"].split("\n")[0]
            # args_list.extend(self.convert_key_value_to_command_line_arg("master_addr", master_address))

        if "DLTS_HOSTFILE" in os.environ:
            args_list.extend(
                self.convert_key_value_to_command_line_arg("hostfile", os.environ["DLTS_HOSTFILE"])
            )

        # jeromeku: IMPORTANT, when using srun_launcher MASTER_ADDR and MASTER_PORT are set within the SLURM script
        if "MASTER_ADDR" in os.environ and not self.use_srun_launcher:
            args_list.extend(
                self.convert_key_value_to_command_line_arg("master_addr", os.environ["MASTER_ADDR"])
            )

        if "MASTER_PORT" in os.environ and not self.use_srun_launcher:
            args_list.extend(
                self.convert_key_value_to_command_line_arg("master_port", os.environ["MASTER_PORT"])
            )

        if ("--include" in args_list or "--exclude" in args_list) and "--num_gpus" in args_list:
            print(
                "WARNING: both --include/--exclude and num_gpus were specified simultaneously - overriding num_gpus with --include/--exclude"
            )
            # cannot specify these both simultaneously, remove num_gpus from list
            idx = args_list.index("--num_gpus")
            # pop twice, once for the arg, once for its value
            args_list.pop(idx)
            args_list.pop(idx)

        # add user script
        args_list.append(self.user_script)

        self.configure_distributed_args()
        cwd = Path.cwd()

        # get deepspeed_config
        args_list.append("--deepspeed_config")

        if self.autotuning_run is not None:
            ds_fp = cwd / Path("ds_config.json")
            if self.rank == 0:
                with open(ds_fp, mode="w") as ds_file:
                    json.dump(self.deepspeed_config, ds_file)
            args_list.append(str(ds_fp))
        else:
            encoded_ds_config = base64.urlsafe_b64encode(
                json.dumps(self.deepspeed_config).encode("utf-8")
            ).decode("utf-8")
            args_list.append(encoded_ds_config)

        # get all config values
        args_list.append("--megatron_config")
        global_config = self.get_parent_class_value_dict(*self.__class__.__bases__, only_non_defaults=True)
        encoded_mega_config = base64.urlsafe_b64encode(json.dumps(global_config).encode("utf-8")).decode(
            "utf-8"
        )
        args_list.append(str(encoded_mega_config))
        return args_list

    ############################################################################################################################
    # start of calculated properties

    @property
    def deepspeed_config(self) -> dict:
        """
        returns a dict containing variables within deepspeed config
        """
        config = self.get_parent_class_value_dict_extra_ds(
            GlobalConfigDeepspeedConfig, only_non_defaults=True
        )
        return config

    @property
    def deepspeed_runner(self) -> dict:
        """
        returns variables within deepspeed runner
        """
        return self.get_parent_class_value_dict(GlobalConfigDeepspeedRunner)

    @property
    def megatron_config(self) -> dict:
        """
        returns variables within megatron args
        """
        return self.get_parent_class_value_dict(*NEOX_ARG_CLASSES)

    @property
    def all_config(self) -> dict:
        """
        returns variables of all args
        """
        return self.get_parent_class_value_dict(*BASE_CLASSES)

    def get_parent_class_value_dict(self, *parent_classes, only_non_defaults=False) -> dict:
        """
        takes a sequence of parent classes and returns corresponding values (with defaults set)
        """
        # TODO no Nones or non-defaults
        result = dict()
        for parent in parent_classes:
            for key, default_value in parent().defaults():
                if key in ["tokenizer", "tensorboard_writer", "adlr_autoresume_object"]:
                    continue
                if only_non_defaults:
                    value = getattr(self, key)
                    if value == default_value:
                        continue
                result[key] = getattr(self, key)
        return result

    def get_parent_class_value_dict_extra_ds(self, *parent_classes, only_non_defaults=False) -> dict:
        """
        Takes a sequence of parent classes and returns corresponding values (with defaults set).
        Also adds in any extra deepspeed arguments that are specified in the configuration file.

        Args:
            parent_classes: sequence of parent classes
            only_non_defaults: if True, only returns values that differ from defaults

        Returns:
            dict of arguments and values

        """
        # TODO no Nones or non-defaults
        result = dict()
        for parent in parent_classes:
            for key, default_value in parent().defaults():
                if key in [
                    "tokenizer",
                    "tensorboard_writer",
                    "adlr_autoresume_object",
                    "deepspeed_extra_args",
                ]:
                    continue
                if only_non_defaults:
                    value = getattr(self, key)
                    if value == default_value:
                        continue
                result[key] = getattr(self, key)

        if self.deepspeed_extra_args is not None:
            extra_ds_args = self.get_extra_deepspeed_args()
            result.update(extra_ds_args)

        return result

    @property
    def params_dtype(self):
        """
        returns the datatype on the basis of configured precision
        """
        if self.precision == "fp16":
            return torch.half
        elif self.precision == "bfloat16":
            return torch.bfloat16
        else:
            return torch.float

    ############################################################################################################################
    # start of logging and output

    def enable_logging(self):
        """
        enable Tee logs based on the configured logdir
        """
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            hostname = gethostname()
            file_prefix = os.path.join(self.log_dir, hostname)
            Tee(file_prefix + "_stdout.txt", err=False)
            Tee(file_prefix + "_stderr.txt", err=True)

    def print(self):
        """Print arguments."""
        if self.rank == 0 or self.rank is None:
            print("-------------------- arguments --------------------", flush=True)
            str_list = []
            for arg in vars(self):
                # add arg + value
                dots = "." * (32 - len(arg))
                value = getattr(self, arg)
                print_str = "  {} {} {}".format(arg, dots, value)

                # add info 'default or updated'
                field_def = self.__dataclass_fields__.get(arg)
                if field_def is not None:
                    default_info = "default" if value == field_def.default else "updated"
                else:
                    default_info = ""
                dots = "." * (64 - len(print_str))
                print_str += dots
                str_list.append({"print_str": print_str, "default_info": default_info})

            for arg in sorted(
                sorted(str_list, key=lambda x: x["print_str"].lower()),
                key=lambda x: x["default_info"],
                reverse=True,
            ):
                print(arg["print_str"] + arg["default_info"], flush=True)
            print("---------------- end of arguments ----------------", flush=True)

    ############################################################################################################################
    # start of calculations and derived values

    def configure_distributed_args(self):
        """
        Configures distributed training arguments from local variables set by deepspeed launcher.
        """
        if self.deepspeed_mpi:
            from deepspeed.comm import mpi_discovery

            mpi_discovery()

        if self.deepspeed_slurm:
            os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
            os.environ["RANK"] = os.environ["SLURM_PROCID"]
            # This is incorrect, should be SLURM_GPUS_ON_NODE * SLURM_NNODES
            os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

        if (
            self.use_srun_launcher
        ):  # srun_launcher uses torchrun to managed distributed launch, so SLURM_NTASKS would give incorrect world sizes
            global_num_gpus = getattr(self, "global_num_gpus", None)
            if global_num_gpus is None:
                assert "SLURM_GPUS_ON_NODE" in os.environ
                assert "SLURM_NNODES" in os.environ
                global_num_gpus = int(os.environ["SLURM_GPUS_ON_NODE"]) * int(os.environ["SLURM_NNODES"])
                self.update_value("global_num_gpus", global_num_gpus)
            os.environ["WORLD_SIZE"] = str(global_num_gpus)

        self.update_value("local_rank", int(os.getenv("LOCAL_RANK", "0")))
        self.update_value("rank", int(os.getenv("RANK", "0")))
        self.update_value("world_size", int(os.getenv("WORLD_SIZE", "1")))

        if self.rank == 0:
            print(
                self.__class__.__name__
                + ".configure_distributed_args() using world size: {}, pipe-parallel size: {}, context-parallel size: {}, and model-parallel size: {} ".format(
                    self.world_size,
                    self.pipe_parallel_size,
                    self.context_parallel_size,
                    self.model_parallel_size,
                ),
                flush=True,
            )

    @staticmethod
    def calculate_batch_parameters(dp_world_size, train_batch=None, micro_batch=None, grad_acc=None):
        # all values are provided nothing needs to be set
        if train_batch is not None and micro_batch is not None and grad_acc is not None:
            return train_batch, micro_batch, grad_acc

        # gradient_accumulation_steps needs to be set
        elif train_batch is not None and micro_batch is not None:
            grad_acc = train_batch // micro_batch
            grad_acc //= dp_world_size

        # micro_batch_per_gpu needs to be set
        elif train_batch is not None and grad_acc is not None:
            micro_batch = train_batch // dp_world_size
            micro_batch //= grad_acc

        # train_batch_size needs to be set
        elif micro_batch is not None and grad_acc is not None:
            train_batch = micro_batch * grad_acc
            train_batch *= dp_world_size

        # gradient_accumulation_steps and micro_batch_per_gpus is set
        elif train_batch is not None:
            grad_acc = 1
            micro_batch = train_batch // dp_world_size

        # train_batch_size and gradient_accumulation_step is set
        elif micro_batch is not None:
            train_batch = micro_batch * dp_world_size
            grad_acc = 1

        # either none of the three parameters are provided or just gradient_accumulation_step is provided
        else:
            assert False, "Either train_batch_size or train_micro_batch_size_per_gpu needs to be provided"
        return int(train_batch), int(micro_batch), int(grad_acc)

    @staticmethod
    def check_batch_parameters(dp_world_size, train_batch, micro_batch, grad_acc):
        assert train_batch > 0, f"Train batch size: {train_batch} has to be greater than 0"

        assert micro_batch > 0, f"Micro batch size per gpu: {micro_batch} has to be greater than 0"

        assert grad_acc > 0, f"Gradient accumulation steps: {grad_acc} has to be greater than 0"

        assert train_batch == micro_batch * grad_acc * dp_world_size, (
            f"Check batch related parameters. train_batch_size is not equal"
            " to micro_batch_per_gpu * gradient_acc_step * world_size \n"
            f"{train_batch} != {micro_batch} * {grad_acc} * {dp_world_size}"
        )

    def calculate_derived(self):
        """
        Derives additional configuration values necessary for training from the current config
        """
        # number of gpus
        # Get number of GPUs param or hostfile to determine train_batch_size
        global_num_gpus = getattr(self, "global_num_gpus", None)
        if global_num_gpus is None:

            if self.use_srun_launcher:
                expected_env = [
                    "WORLD_SIZE",
                    "SLURM_GPUS_ON_NODE",
                    "SLURM_JOB_NUM_NODES",
                    "SLURM_NTASKS",
                    "SLURM_NTASKS_PER_NODE",
                    "GLOBAL_NUM_GPUS",
                ]
                for env in expected_env:
                    assert (
                        env in os.environ
                    ), f"{env} not found in env, should have already been set by launcher script or SLURM"
                # @jeromeku NOTE: MUST use SLURM_JOB_NUM_NODES and not SLURM_NNODES
                # in slurm launcher script, we are calling this function within an `srun` step that only executes
                # on the MASTER_NODE; SLURM_NNODES evaluates to 1 in this case, whereas we are interested in the total
                # number of nodes for the entire SLURM job (SLURM_JOB_NUM_NODES).
                # world_size = int(os.environ["WORLD_SIZE"])
                ntasks = int(os.environ["SLURM_NTASKS"])
                ntasks_per_node = int(os.environ["SLURM_NTASKS_PER_NODE"])
                nnodes = int(os.environ["SLURM_JOB_NUM_NODES"])
                gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
                global_num_gpus = int(os.environ["GLOBAL_NUM_GPUS"])
                print(
                    f"SLURM_NTASKS: {ntasks}, SLURM_NTASKS_PER_NODE: {ntasks_per_node}, SLURM_JOB_NUM_NODES: {nnodes}, SLURM_GPUS_ON_NODE: {gpus_per_node}"
                )

                if self.srun_launcher_type == "srun":
                    # assert (
                    #     world_size == ntasks
                    # ), f"Using launcher {self.srun_launcher_type} requires WORLD_SIZE == SLURM_NTASKS, but WORLD_SIZE: {world_size} != SLURM_NTASKS: {ntasks}"
                    assert (
                        ntasks_per_node == gpus_per_node
                    ), f"Using launcher {self.srun_launcher_type} requires SLURM_NTASKS_PER_NODE == SLURM_GPUS_ON_NODE, but SLURM_NTASKS_PER_NODE: {ntasks_per_node} != SLURM_GPUS_ON_NODE: {gpus_per_node}"

                elif self.srun_launcher_type == "torch" or self.srun_launcher_type == "deepspeed":
                    assert (
                        ntasks_per_node == 1
                    ), f"Using launcher {self.srun_launcher_type} requires SLURM_NTASKS_PER_NODE == 1, but SLURM_NTASKS_PER_NODE: {ntasks_per_node} != 1"
                else:
                    raise ValueError(f"Unknown srun launcher type: {self.srun_launcher_type}")

                # This must be calculated as such, since this will be executed by only 1 srun task (not all tasks, so WORLD_SIZE != SLURM_NTASKS)
                world_size = nnodes * gpus_per_node
                assert (
                    world_size == global_num_gpus
                ), f"Using launcher {self.srun_launcher_type} requires WORLD_SIZE = NNODES * GPUS_ON_NODE == GLOBAL_NUM_GPUS, but {nnodes} * {gpus_per_node} != GLOBAL_NUM_GPUS: {global_num_gpus}"

                print(f"Using launcher {self.srun_launcher_type}", flush=True)
                print(f"Setting global_num_gpus to {global_num_gpus}", flush=True)

            elif self.hostfile is not None or os.path.exists(runner.DLTS_HOSTFILE):
                hostfile_path = self.hostfile or runner.DLTS_HOSTFILE
                resources = obtain_resource_pool(hostfile_path, self.include or "", self.exclude or "")
                if self.num_nodes is not None and self.num_nodes > 0:
                    resources = {k: resources[k] for k in list(resources.keys())[: self.num_nodes]}
                global_num_gpus = sum(map(len, resources.values()))
                if self.num_gpus is not None and self.num_gpus > 0:
                    global_num_gpus = self.num_gpus * len(resources)

            else:
                global_num_gpus = torch.cuda.device_count()
            self.update_value("global_num_gpus", global_num_gpus)

        logging.info(
            self.__class__.__name__
            + ".calculate_derived() "
            + f"Total number of GPUs determined to be: {global_num_gpus}"
        )

        # get world size in the model/pipe parallel case, the actual `world size` deepspeed uses is the size of the
        # data-parallel group, or (num_gpus / mp_size) / pp_size
        pp_size = self.pipe_parallel_size
        pp_size = pp_size if pp_size >= 1 else 1
        mp_size = self.model_parallel_size
        mp_size = mp_size if mp_size >= 1 else 1
        cp_size = self.context_parallel_size
        cp_size = cp_size if cp_size >= 1 else 1
        self.update_value("model_parallel_size", mp_size)
        self.update_value("context_parallel_size", cp_size)

        # pp_size, mp_size, and cp_size are only used here to compute dp world size and nowhere else.
        dp_world_size = (global_num_gpus / pp_size) / (mp_size * cp_size)
        if not (dp_world_size % 1 == 0):
            error_message = (
                self.__class__.__name__
                + ".calculate_derived() "
                + f"(global_num_gpus / pp_size) / mp_size [({global_num_gpus} / {pp_size}) / {mp_size}] must be a whole number"
            )
            logging.error(error_message)
            raise AssertionError(error_message)

            # Automatically derive train_batch_size = train_micro_batch_size_per_gpu*global_num_gpus*gradient_accumulation_steps
        (
            train_batch_size,
            train_micro_batch_size_per_gpu,
            gradient_accumulation_steps,
        ) = self.calculate_batch_parameters(
            dp_world_size=dp_world_size,
            train_batch=self.train_batch_size,
            micro_batch=self.train_micro_batch_size_per_gpu,
            grad_acc=self.gradient_accumulation_steps,
        )
        self.check_batch_parameters(
            dp_world_size=dp_world_size,
            train_batch=train_batch_size,
            micro_batch=train_micro_batch_size_per_gpu,
            grad_acc=gradient_accumulation_steps,
        )
        self.update_values(
            {
                # batch size params
                "train_batch_size": train_batch_size,
                "train_micro_batch_size_per_gpu": train_micro_batch_size_per_gpu,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "batch_size": train_micro_batch_size_per_gpu,
                # duplicate items
                "gas": self.gradient_accumulation_steps,
                "clip_grad": self.gradient_clipping,
            }
        )

        # derive steps where checkpoint should be saved
        if self.checkpoint_factor or self.extra_save_iters:
            if self.extra_save_iters:
                save_iters = set(self.extra_save_iters)
            else:
                save_iters = set()

            step = self.checkpoint_factor  # don't save step 0 or 1
            while step < self.train_iters:
                save_iters.add(step)
                if self.checkpoint_scale == "log":
                    step *= self.checkpoint_factor
                elif self.checkpoint_scale == "linear":
                    step += self.checkpoint_factor

            save_iters = list(save_iters)
            save_iters.sort()

            self.update_values(
                {
                    "save_iters": save_iters,
                }
            )

        # derive precision
        fp16_conflict = "DeepSpeed fp16 field was set but precision conflicts"
        if self.fp16 and self.fp16.get("enabled", False):
            if self.precision is None:
                self.update_value("precision", "fp16")
            else:
                assert self.precision == "fp16", fp16_conflict

        if self.precision == "fp16":
            if isinstance(self.fp16, dict) and len(self.fp16) > 0:
                fp16_args = copy.deepcopy(self.fp16)
                fp16_args["enabled"] = True
            else:
                fp16_args = {"type": "fp16", "enabled": True}
            self.update_value("fp16", fp16_args)
        elif self.precision == "bfloat16":
            bf_config = {"bf16": {"enabled": True}}
            if self.deepspeed_extra_args is None:
                self.update_value("deepspeed_extra_args", bf_config)
            else:
                extra_args = copy.deepcopy(self.deepspeed_extra_args)
                extra_args.update(bf_config)
                self.update_value("deepspeed_extra_args", extra_args)
        else:
            self.update_value("precision", "fp32")

        # zero optimization
        if self.zero_optimization is None:
            self.zero_optimization = copy.deepcopy(
                ZERO_DEFAULTS
            )  # a dict is overwritten and not updated key by key
        try:
            stage = self.zero_optimization["stage"]
            if stage in (0, 1, 2, 3):
                self.update_values(
                    {
                        "zero_stage": self.zero_optimization.get("stage", ZERO_DEFAULTS["stage"]),
                        "zero_reduce_scatter": self.zero_optimization.get(
                            "reduce_scatter", ZERO_DEFAULTS["reduce_scatter"]
                        ),
                        "zero_contiguous_gradients": self.zero_optimization.get(
                            "contiguous_gradients",
                            ZERO_DEFAULTS["contiguous_gradients"],
                        ),
                        "zero_reduce_bucket_size": self.zero_optimization.get(
                            "reduce_bucket_size", ZERO_DEFAULTS["reduce_bucket_size"]
                        ),
                        "zero_allgather_bucket_size": self.zero_optimization.get(
                            "allgather_bucket_size",
                            ZERO_DEFAULTS["allgather_bucket_size"],
                        ),
                    }
                )
            else:
                assert (
                    self.autotuning is not None
                ), f"Zero Stage must be an integer unless you are doing autotuning, not {stage}"
        except KeyError as ke:
            print(f"Zero Optimization config: {self.zero_optimization}")
            raise ke

        # optimizer and scheduler
        opt_params = self.optimizer or {
            "type": OPT_DEFAULT,
            "params": OPT_PARAMS_DEFAULTS,
        }
        self.update_values(
            {
                "optimizer_type": opt_params.get("type", OPT_DEFAULT),
                "lr": opt_params["params"].get("lr", OPT_PARAMS_DEFAULTS["lr"]),
            }
        )

        if self.optimizer_type.lower() == "onebitadam":
            # onebitadam needs to instantiated by deepspeed, and so we need to pass deepspeed scheduler args
            # for all other optimizers, the scheduling is handled by megatron
            self.scheduler = {
                "type": "WarmupDecayLR",  # for now this is the only ds scheduler offering decay
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": self.lr,
                    "warmup_num_steps": int(self.train_iters * self.warmup),
                    "total_num_steps": self.lr_decay_iters or self.train_iters,
                },
            }

        # Fp16 loss scaling.
        self.update_value("dynamic_loss_scale", self.loss_scale is None)

        # Update 'is pipe parallel' flag
        # if we set pipe_parallel_size to 0 or 1, BackbonePipe.to_sequential() is called, and we run training with
        # the sequential model without the PipelineModule wrapper to avoid the overhead it incurs
        self.update_value("is_pipe_parallel", self.pipe_parallel_size > 1)
        
        # update 'is context parallel' flag
        self.update_value("is_context_parallel", self.context_parallel_size > 1)

        # Attention config
        if self.operator_config is None:
            self.update_value("operator_config", [[["global"], self.num_layers]])
        self.update_value(
            "operator_config",
            expand_operator_types(self.operator_config, self.num_layers),
        )
        assert (
            len(self.operator_config) == self.num_layers
        ), f"Length of attention config ({len(self.operator_config)}) list must equal num_layers ({self.num_layers})"
        
        assert all([item in ["flash_te", "ring", "hyena", "hyena_mr", "hyena_se"] for item in self.operator_config]) or (
            not self.is_context_parallel
        ), "Context parallel requires operators with flash_te or ring communication."
        
        for item in self.operator_config:
            assert item in operator_type_CHOICES, f"Attention type {item} not recognized"
        
        if self.is_context_parallel:
            if self.use_cp_flash_te:
                assert all([item in ["flash_te", "hyena", "hyena_mr", "hyena_se"] for item in self.operator_config]), "CP requires use of flash_te, hyena, hyena_mr, or hyena_se operators"
            elif self.use_cp_ring:
                assert all([item in ["ring", "hyena", "hyena_mr", "hyena_se"] for item in self.operator_config]), "CP requires use of ring, hyena, hyena_mr, or hyena_se operators"
            else:
                assert self.use_cp_hyena, "CP requires use of hyena or CP attn operators (flash_te or ring), must set use_cp_hyena or one of use_cp_flash_te or use_cp_ring"
        
        if "gmlp" in self.operator_config or "amlp" in self.operator_config:
            assert not self.partition_activations, "GMLP Blocks are not compatible with partition activations"

        if self.pretraining_strategy == "AR":
            self.causal = True
        elif self.pretraining_strategy in ["MLM", "OADM", "SPAN", "SPAN_R"]:
            self.causal = False

        # Sparsity config
        if self.sparsity_config is None:
            # Can't have a default value as an empty dict so need to set it here
            self.update_value("sparsity_config", {})

        # Adding equal dataset weights if none are provided
        if self.train_data_paths and (self.train_data_weights is None):
            self.train_data_weights = [1.0] * len(self.train_data_paths)
        if self.valid_data_paths and (self.valid_data_weights is None):
            self.valid_data_weights = [1.0] * len(self.valid_data_paths)
        if self.test_data_paths and (self.test_data_weights is None):
            self.test_data_weights = [1.0] * len(self.test_data_paths)

        # if a sample input file is provided, default text_gen_type type to input-file
        if self.text_gen_type is None:
            if self.sample_input_file:
                self.update_value("text_gen_type", "input-file")
            else:
                self.update_value("text_gen_type", "unconditional")

    ############################################################################################################################
    # start of validation functions

    @classmethod
    def validate_keys(cls):
        """
        test that there are no duplicate arguments
        """
        source_classes = list(cls.__bases__)
        defined_properties = dict()

        for source_class in source_classes:
            source_vars = list(source_class.__dataclass_fields__)
            for item in source_vars:
                if item in defined_properties.keys():
                    logging.error(
                        f"({cls.__name__}) duplicate of item: {item}, in class {source_class.__name__} and {defined_properties[item]}"
                    )
                    return False
                else:
                    defined_properties[item] = source_class.__name__
        return True

    def validate_values(self):
        # the current codebase assumes running with deepspeed only
        if not self.deepspeed:
            return False

        # learning rate
        if self.lr is None:
            error_message = self.__class__.__name__ + ".validate_values() lr is None"
            logging.error(error_message)
            raise ValueError(error_message)
            return False

        # required arguments
        required_args = [
            "num_layers",
            "hidden_size",
            "num_attention_heads",
        ]
        for req_arg in required_args:
            if getattr(self, req_arg) is None:
                error_message = self.__class__.__name__ + ".validate_values() " + req_arg + " is None."
                logging.error(error_message)
                raise ValueError(error_message)
                return False

        # Checks.
        if self.hidden_size % self.num_attention_heads != 0:
            error_message = (
                self.__class__.__name__
                + ".validate_values() hidden_size must be divisible by num_attention_heads"
            )
            logging.error(error_message)
            raise ValueError(error_message)
            return False

        # if self.seq_length is not None:
        #    if not (self.max_position_embeddings >= self.seq_length):
        #        error_message = (
        #            self.__class__.__name__
        #            + ".validate_values() max_position_embeddings must be bigger or equal seq_length"
        #        )
        #        logging.error(error_message)
        #        raise ValueError(error_message)
        #        return False

        if not (self.min_lr <= self.lr):
            error_message = self.__class__.__name__ + ".validate_values() min_lr must be smaller or equal lr"
            logging.error(error_message)
            raise ValueError(error_message)
            return False

        if self.save is not None and self.checkpoint_factor is None and self.extra_save_iters is None:
            error_message = (
                self.__class__.__name__
                + ".validate_values() checkpoint_factor or extra_save_iters must be defined if save is defined"
            )
            logging.error(error_message)
            raise ValueError(error_message)
            return False

        # Parameters sharing does not work with torch DDP.
        if (self.num_unique_layers is not None) and (self.num_layers is not None):
            if not (self.num_unique_layers <= self.num_layers):
                error_message = (
                    self.__class__.__name__
                    + ".validate_values() num-unique-layers must be smaller or equal num_layers"
                )
                logging.error(error_message)
                raise ValueError(error_message)
                return False

            if not (self.num_layers % self.num_unique_layers == 0):
                error_message = (
                    self.__class__.__name__
                    + ".validate_values() num-layers should be divisible by num-unique-layers"
                )
                logging.error(error_message)
                raise ValueError(error_message)
                return False

        # Sequence parallel does not work with partition activations
        # If you're using sequence parallel, you should not need to partition activations
        # for activation checkpointing
        if self.sequence_parallel and self.partition_activations:
            error_message = (
                self.__class__.__name__
                + ".validate_values() only one of sequence-parallel and partition-activations should be set."
            )
            logging.error(error_message)
            raise ValueError(error_message)
            return False

        if self.fp16_lm_cross_entropy and self.precision != "fp16":
            error_message = (
                self.__class__.__name__
                + ".validate_values() lm cross entropy in fp16 only support in fp16 mode."
            )
            logging.error(error_message)
            raise ValueError(error_message)
            return False

        # assert that if one of train/test/valid_data_path are provided, data_path should not be
        has_separate_path = [
            data_path is not None
            for data_path in [
                self.train_data_paths,
                self.valid_data_paths,
                self.test_data_paths,
            ]
        ]
        if all(has_separate_path):
            assert self.data_path is None, (
                "Please provide *either* `data_path` or `train/valid/test_data_path` " "in args "
            )

        # assert that if one of train/test/valid_data_path are provided, all should be
        assert_error_mess = "One or more of train/valid/test data_path are not provided:\n\t"
        assert_error_mess += "\n\t".join(
            [
                f"{name} data paths: {data_path},"
                for name, data_path in [
                    ["train", self.train_data_paths],
                    ["valid", self.valid_data_paths],
                    ["test", self.test_data_paths],
                ]
            ]
        )
        assert any(has_separate_path) == all(has_separate_path), assert_error_mess

        # assert that if train / valid / test data path(s) and weights are provided, that the paths and the weights should be equal length
        if self.train_data_paths is not None:
            assert len(self.train_data_paths) == len(self.train_data_weights)
        if self.valid_data_paths is not None:
            assert len(self.valid_data_paths) == len(self.valid_data_weights)
        if self.test_data_paths is not None:
            assert len(self.test_data_paths) == len(self.test_data_weights)

        return True

    def validate_types(self):
        """
        At runtime, checks types are actually the type specified.
        """
        for field_name, field_def in self.__dataclass_fields__.items():
            actual_value = getattr(self, field_name)
            if actual_value is None:
                continue  # we allow for some values not to be configured

            if self.autotuning is not None and actual_value == "auto":
                continue

            actual_type = type(actual_value)
            #print(f"{field_name=}: {field_def=} {field_def.type=} {actual_value=} {actual_type=}")
            if actual_type != field_def.type:
                if (
                    actual_type == int and field_def.type == float
                ):  # floats should be able to be configured as ints
                    continue

                
                # for typing.Literal (i.e a list of choices) - checks that actual value is in accepted values
                elif field_def.type.__origin__ == Literal:
                    accepted_values = field_def.type.__args__
                    if actual_value in accepted_values:
                        continue
                    elif type(actual_value) == str:
                        # case insensitive checking
                        lowercase_accepted_values = [i.lower() for i in accepted_values if isinstance(i, str)]
                        if actual_value.lower() in lowercase_accepted_values:
                            continue
                    logging.error(
                        self.__class__.__name__
                        + ".validate_types() "
                        + f"{field_name}: '{actual_value}' Not in accepted values: '{accepted_values}'"
                    )
                    return False
                elif field_def.type.__origin__ == Union:
                    accepted_types = field_def.type.__args__
                    if actual_type in accepted_types:
                        continue
                    else:
                        logging.error(
                            self.__class__.__name__
                            + ".validate_types() "
                            + f"{field_name}: '{actual_type}' not in {accepted_types}"
                        )
                        return False

                logging.error(
                    self.__class__.__name__
                    + ".validate_types() "
                    + f"{field_name}: '{actual_type}' instead of '{field_def.type}'"
                )
                return False

        # validate deepspeed dicts
        for field_name in ["optimizer", "scheduler"]:
            value = getattr(self, field_name)
            if isinstance(value, dict):  # dict is checked above, only fields are checked here
                if "type" in value:
                    if not isinstance(value["type"], str):
                        logging.error(
                            self.__class__.__name__
                            + ".validate_types() "
                            + f"{field_name}: key 'type' must be a string"
                        )
                        return False
                else:
                    logging.error(
                        self.__class__.__name__
                        + ".validate_types() "
                        + f"{field_name}: must contain key 'type'"
                    )
                    return False
                if "params" in value:
                    if not isinstance(value["params"], dict):
                        logging.error(
                            self.__class__.__name__
                            + ".validate_types() "
                            + f"{field_name}: key 'params' must be a dict"
                        )
                        return False
                else:
                    logging.error(
                        self.__class__.__name__
                        + ".validate_types() "
                        + f"{field_name}: must contain key 'params'"
                    )
                    return False

        for field_name in ["fp16", "amp", "flops_profiler"]:
            value = getattr(self, field_name)
            if isinstance(value, dict):
                if "enabled" not in value:
                    error_message = (
                        self.__class__.__name__
                        + ".validate_types() "
                        + f"{field_name}: must contain key 'enabled'"
                    )
                    logging.error(error_message)
                    return False

        return True
