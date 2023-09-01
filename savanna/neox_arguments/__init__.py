"""
NeoX Arguments manages all configuration arguments.

**general**

* The implementation makes use of the python dataclass.
* The main class 'GlobalConfig' (in ./arguments) exposes all configuration attributes that are relevant to GPT NeoX
* No attributes are nested (apart from attributes with type dict)
* Output functions (enable_logging, save_yml, print) are implemented
* Instantiation always runs GlobalConfig.__post_init__(), which calculates derived values and performs a validation (values, types, keys).
* it is possible to set undefined attributes (e.g. line of code 'GlobalConfig().my_undefined_config = 42' works fine); such set attributes are not validated
* It is possible to update attributes (e.g. line of code 'GlobalConfig().do_train = True' works fine); a validation can be performed by calling the validation functions on the class instance
* In order to avoid setting undefined attributes you can use the function GlobalConfig().update_value(); this function raises an error if the to be set attribute is not defined

**instantiation**
NeoX args can be instantiated with the following options

* GlobalConfig.from_ymls(["path_to_yaml1", "path_to_yaml2", ...]): load yaml configuration files and instantiate with the values provided; checks for duplications and unknown arguments are performed
* GlobalConfig.from_dict({"num_layers": 12, ...}): load attribute values from dict; checks unknown arguments are performed

* GlobalConfig.consume_deepy_args(): entry point for deepy.py configuring and consuming command line arguments (i.e. user_script, conf_dir, conf_file, wandb_group, wandb_team); global_config.get_deepspeed_main_args() produces a list of command line arguments to feed to deepspeed.launcher.runner.main
* GlobalConfig.consume_global_config(): In the call stack deepy.py -> deepspeed -> pretrain_gpt2.py; arguments are passed to pretrain_gpt2.py by global_config.get_deepspeed_main_args(). So produced arguments can be read with consume_global_config() to instantiate a GlobalConfig instance.


**code structure**

* NeoX args (in ./arguments) inherits from the following subclasses: GlobalConfigDeepspeedRunner, GlobalConfigDeepspeedConfig, GlobalConfigModel, GlobalConfigTokenizer, GlobalConfigTraining, GlobalConfigParallelism, GlobalConfigLogging, GlobalConfigOther, GlobalConfigTextgen
* The Subclasses group args according to their purpose
* The attributes of GlobalConfigDeepspeedRunner are directly mapped to the expected command line args of deepspeed.launcher.runner.main; no attributes unknown to deepspeed should be included; no arguments relevant for deepspeed should be omitted
* The attributes of GlobalConfigDeepspeedConfig are directly mapped to the expected keys of the deepspeed config; no arguments relevant for deepspeed should be omitted
* calculated attributes (decorator '@property') are available as attribute, but would not be included in dataclass fields (e.g. GlobalConfig().__dataclass_fields__.items())
* refer to docstrings in code for more information
"""


from .arguments import GlobalConfig
