# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from datetime import datetime

import torch

import savanna.lazy_imports


def print_rank_0(*message, debug=False):
    """If distributed is initialized print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            if debug:
                print("DEBUG:",*message, flush=True)
            else:
                print(*message, flush=True)
    else:
        if debug:
            print("DEBUG:",*message, flush=True)
        else:
            print(*message, flush=True)


def print_datetime(string: str, rank_0: bool = True):
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if rank_0:
        print_rank_0('[' + string + '] datetime: {} '.format(time_str))
    else:
        print('[' + string + '] datetime: {} '.format(time_str))

from .arguments import GlobalConfig
from .initialize import initialize_megatron
