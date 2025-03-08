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

"""Model parallel utility interface."""

from .cross_entropy import vocab_parallel_cross_entropy
from .data import (
    broadcast_data,
    zigzag_gather_from_cp_ranks,
    zigzag_split_across_cp_ranks
)
from .initialize import (
    get_sequence_parallel_group,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sequence_parallel_src_rank,
    get_sequence_data_parallel_group,
    get_sequence_data_parallel_rank,
    get_sequence_data_parallel_world_size,
)
from .initialize import (
    destroy_model_parallel,
    get_data_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_src_rank,
    get_data_parallel_world_size,
    get_fp32_allreduce,
    get_io_parallel_group,
    get_model_parallel_group,
    get_model_parallel_rank,
    get_model_parallel_src_rank,
    get_model_parallel_world_size,
    get_pipe_parallel_group,
    get_pipe_parallel_rank,
    get_pipe_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_topology,
    initialize_model_parallel,
    is_unitialized,
    model_parallel_is_initialized,
    set_model_parallel_rank,
    set_model_parallel_world_size,
)
from .layers import (
    ColumnParallelLinear,
    ParallelRelativePositionBias,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from .mappings import (
    copy_to_model_parallel_region,
    gather_from_model_parallel_region,
    gather_from_sequence_parallel_region,
    reduce_from_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from .random import checkpoint, get_cuda_rng_tracker, model_parallel_cuda_manual_seed
from .utils import divide, split_tensor_along_last_dim
