import torch

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.set_printoptions(precision=5)

class PipelineModel(torch.nn.Module):
    def __init__(self, d=10, num_layers=2):
        super().__init__()
        world_size = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()
        layers_per_rank = num_layers // world_size
        self.start_layer = rank * layers_per_rank
        self.end_layer = (rank + 1) * layers_per_rank
        print(f"DEBUG::PIPELINE_SEND_RECEIVE::{rank=} layers_per_rank {layers_per_rank} start_layer {self.start_layer} end_layer {self.end_layer}")
        self.layers =  torch.nn.ModuleList()
        for i in range(num_layers):
            if self.start_layer <= i < self.end_layer:
                self.layers.append(torch.nn.Linear(d, d, bias=False))
                print(f"DEBUG::PIPELINE_SEND_RECEIVE::{rank=} layer {i} {self.layers[-1].weight.view(-1)[:5]}", flush=True)
            else:
                torch.nn.Linear(d,d, bias=False)
                self.layers.append(torch.nn.Identity())

    def forward(self, input_ids, intermediate_tensors=None):
        if self.rank == 0:
            hidden_states = input_ids
        else:
            hidden_states = intermediate_tensors
        # if self.rank == 1:
        #     print(f"DEBUG::PIPELINE_SEND_RECEIVE::{self.rank=} hidden_states {hidden_states.view(-1)[:10]}")
        for i in range(self.start_layer, self.end_layer):
            hidden_states = self.layers[i](hidden_states)
        return hidden_states

if __name__ == "__main__":
    torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    # print(f"DEBUG::PIPELINE_SEND_RECEIVE::{rank=} {world_size=}")

    torch.cuda.set_device(rank)
    d = 10
    num_layers = 4

    model = PipelineModel(d=d, num_layers=num_layers).cuda()
#    print(f"DEBUG::PIPELINE_SEND_RECEIVE::{rank=} model {model}", flush=True)
 
    intermediate_tensors = None
    if rank == 1:
        intermediate_tensors = torch.empty(d, d, device="cuda")
        torch.distributed.recv(intermediate_tensors, src=0)
    
    if rank == 0:
        input_ids = torch.randn(d, d, device="cuda")
    else:
        input_ids = None
    # if rank == 0:
    #     print(f"DEBUG::PIPELINE_SEND_RECEIVE::{rank=} input_ids {input_ids.view(-1)[:10]}")
    output = model(input_ids, intermediate_tensors)

    if rank == 0 and world_size > 1:
        torch.distributed.send(output, dst=1)
    
    print(f"DEBUG::PIPELINE_SEND_RECEIVE::{rank=} output {output.view(-1)[:10]}", flush=True)

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    