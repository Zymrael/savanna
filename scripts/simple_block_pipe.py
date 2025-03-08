import argparse
import datetime
import os

import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format

#from vortex.model.layers import TELinear

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.set_printoptions(precision=5)

#NOTE: jeromeku: MUST set reduce_amax=False for pipelining to work, else will hang as TE will try to sync amax across ranks
FP8_RECIPE = DelayedScaling(fp8_format=Format.E4M3, reduce_amax=False)


class Block(torch.nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        use_norm,
        use_te_linears,
        dtype=torch.bfloat16,
        use_fp8=False,
        bias=False,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.use_norm = use_norm
        self.use_te_linears = use_te_linears
        self.use_fp8 = use_fp8
        
        if use_fp8:
            assert use_te_linears, "TE linears are required for FP8"

        self.dtype = dtype


        self.linear = (
            te.Linear(
                in_features=input_size,
                out_features=output_size,
                init_method=torch.nn.init.xavier_uniform_,
                bias=bias,
                params_dtype=dtype,
            )
            if use_te_linears
            else torch.nn.Linear(self.input_size, self.output_size, bias=bias, dtype=dtype)
        )
        if use_norm:
            self.norm = torch.nn.LayerNorm(input_size, dtype=dtype)

    def forward(self, x):
        if self.use_norm:
            x = self.norm(x)
        # if self.use_fp8:
        #     with te.fp8_autocast(enabled=True, fp8_recipe=FP8_RECIPE):
        #         x = self.linear(x)
        # else:
      
        x = self.linear(x)
      
        if isinstance(x, tuple):
            x = x[0]
        return x


class PipelineModel(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        num_layers,
        output_factor=1,
        use_norm=False,
        use_te_linears=False,
        dtype=torch.bfloat16,
        use_fp8=False,
        world_size=None,
        rank=None,
        verbose=False,
    ):
        super().__init__()

        world_size = world_size if world_size is not None else torch.distributed.get_world_size()
        rank = rank if rank is not None else torch.distributed.get_rank()
        self.rank = rank
        self.world_size = world_size

        assert num_layers % world_size == 0, "num_layers must be divisible by world_size"

        layers_per_rank = num_layers // world_size
        self.start_layer = rank * layers_per_rank
        self.end_layer = (rank + 1) * layers_per_rank
        self.hidden_size = hidden_size
        self.verbose = verbose

        if verbose:
            print(
                f"DEBUG::PipelineModel::__init__:{rank=} layers_per_rank {layers_per_rank} start_layer {self.start_layer} end_layer {self.end_layer}"
            )

        self.layers = torch.nn.ModuleList()
        output_size = output_factor * hidden_size
        for i in range(num_layers):
            input_size = hidden_size if i == 0 else output_size
            if self.start_layer <= i < self.end_layer:
                self.layers.append(
                    Block(
                        input_size=input_size,
                        output_size=output_size,
                        use_norm=use_norm,
                        use_te_linears=use_te_linears,
                        dtype=dtype,
                        use_fp8=use_fp8,
                    )
                )
            else:
                self.layers.append(torch.nn.Identity())

    def forward(self, input_ids, intermediate_tensors=None):
        if self.rank == 0:
            hidden_states = input_ids
        else:
            hidden_states = intermediate_tensors

        if self.verbose:
            print(f"DEBUG::PipelineModel::forward:{self.rank=} hidden_states {hidden_states.view(-1)[:10]}")

        for i in range(self.start_layer, self.end_layer):
            hidden_states = self.layers[i](hidden_states)
            if self.verbose:
                print(f"DEBUG::PipelineModel::forward:{self.rank=} layer {i} output {hidden_states.view(-1)[:10]}")
        return hidden_states


def is_first_rank():
    return torch.distributed.get_rank() == 0


def is_last_rank():
    return torch.distributed.get_rank() == torch.distributed.get_world_size() - 1


def save_checkpoint(model, checkpoint_path):
    if not os.path.exists(os.path.dirname(checkpoint_path)):
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)


def load_checkpoint(model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))


def create_reference_model(
    num_layers,
    hidden_size,
    output_factor,
    use_norm,
    use_te_linears,
    dtype,
    use_fp8,
    rank=0,
    checkpoint_path="pipe_ref.pt",
    verbose=False,
):
    # Create single rank reference checkpoint
    if not rank == 0:
        return None

    world_size = 1
    model = PipelineModel(
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_factor=output_factor,
        use_norm=use_norm,
        use_te_linears=use_te_linears,
        dtype=dtype,
        use_fp8=use_fp8,
        world_size=world_size,
        rank=rank,
        verbose=verbose,
    ).cuda()
    model.eval()

    # save_checkpoint(model, checkpoint_path)
    # torch.save(output, f"checkpoints/pipe_ref_output.pt")
    return model


def calibrate(model, inputs, use_fp8=False, num_steps=10):
    """Calibration function."""
    model.eval()
    with torch.no_grad():
        for _ in range(num_steps):
            with te.fp8_autocast(enabled=use_fp8, calibrating=True):
                output = model(inputs)
    return model

def main(args):

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = rank % torch.cuda.device_count()
    if args.verbose:
        print(f"{rank=} {local_rank=} {world_size=}", flush=True)

    torch.cuda.set_device(local_rank)

    if rank == 0:
        inputs = torch.randn(args.batch_size, args.seq_len, args.hidden_size, device="cuda", dtype=args.dtype)
    else:
        inputs = None

    input_size = args.hidden_size
    output_size = args.output_factor * args.hidden_size

    # Create reference single rank model
    ref_checkpoint_path = f"checkpoints/pipe_ref.pt"
    if rank == 0:
        print(f"Creating reference model with FP8={args.use_fp8} and TE={args.use_te_linears}")
    ref_model = create_reference_model(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        output_factor=args.output_factor,
        use_norm=args.use_norm,
        use_te_linears=args.use_te_linears,
        use_fp8=args.use_fp8,
        dtype=args.dtype,
        rank=rank,
        checkpoint_path=ref_checkpoint_path,
        verbose=args.verbose,
    )
    torch.distributed.barrier()

    if args.verbose and rank == 0:
        print(f"Running reference model", flush=True)
    if rank == 0:
        ref_model = calibrate(ref_model, inputs, use_fp8=False)
        
        with torch.no_grad():
            with te.fp8_autocast(enabled=args.use_fp8, fp8_recipe=FP8_RECIPE):
                ref_output = ref_model(inputs)
        
        if args.verbose:
            print(f"Ref output {ref_output.view(-1)[:10]}")
        
        save_checkpoint(ref_model, f"checkpoints/pipe_ref.pt")
    else:
        ref_output = torch.empty(args.batch_size, args.seq_len, output_size, device="cuda", dtype=args.dtype)


    torch.distributed.barrier()

    torch.distributed.broadcast(ref_output, src=0)

    # Create pipeline model
    model = PipelineModel(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_factor=args.output_factor,
        use_norm=args.use_norm,
        use_te_linears=args.use_te_linears,
        dtype=args.dtype,
        verbose=args.verbose,
        use_fp8=args.use_fp8,
    ).cuda().eval()

    # Load checkpoint
    if args.verbose and rank == 0:
        print(f"Loading checkpoint {ref_checkpoint_path}", flush=True)
    model.load_state_dict(torch.load(ref_checkpoint_path, map_location="cuda"), strict=False)
    save_checkpoint(model, f"checkpoints/pipe_rank_{rank}.pt")

    # Run pipeline model
    intermediate_tensors = None
    if not is_first_rank():
        if args.verbose:
            print(f"rank{rank}: Receiving from rank {rank - 1}", flush=True)
        
        intermediate_tensors = torch.empty(
            args.batch_size, args.seq_len, output_size, device="cuda", dtype=args.dtype
        )
        torch.distributed.recv(intermediate_tensors, src=rank - 1)
    else:
        intermediate_tensors = inputs

    with torch.no_grad():
        with te.fp8_autocast(enabled=args.use_fp8, fp8_recipe=FP8_RECIPE):
            output = model(inputs, intermediate_tensors)

    if not is_last_rank():
        if args.verbose:
            print(f"rank{rank}: Sending to rank {rank + 1}", flush=True)
        torch.distributed.send(output, dst=rank + 1)

    torch.save(output, f"checkpoints/pipe_rank_{rank}_output.pt")

    if is_last_rank():
        print(f"Ref vs Pipeline abs max diff: {(ref_output.cpu() - output.cpu()).abs().max()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--output_factor", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=10)
    parser.add_argument("--use_norm", action="store_true")
    parser.add_argument("--use_te_linears", action="store_true")
    parser.add_argument("--use_fp8", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    args.dtype = getattr(torch, args.dtype)

    torch.distributed.init_process_group(backend="nccl")
    main(args)

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()