import argparse

import torch
import torch.nn as nn
from vortex.model.layers import TELinear

torch.use_deterministic_algorithms(True)

class TestBlock(nn.Module):
    def __init__(self, hidden_size, use_te_linear=True, dtype=torch.bfloat16, is_first_block=True, block_idx=0, use_norm=True):
        super().__init__()
        self.use_te_linear = use_te_linear
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.block_idx = block_idx
        
        input_size = hidden_size if is_first_block else hidden_size * 3
        output_size = hidden_size * 3
        
        # Initialize on cuda:0
        if use_te_linear:
            self.linear = TELinear(
                input_size, 
                output_size, 
                init_method=torch.nn.init.xavier_uniform_,
                bias=False,
            ).to(device="cuda:0", dtype=dtype)
        else:
            self.linear = nn.Linear(input_size, output_size, bias=False, dtype=dtype, device="cuda:0")
            
        self.use_norm = use_norm
        if use_norm:
            self.norm = nn.LayerNorm(input_size).to(device="cuda:0", dtype=dtype)
    
    def forward(self, x):
        device = self.linear.weight.device
        x = x.to(device=device, dtype=self.dtype)
        if self.use_norm:
            x = self.norm(x)
        x = self.linear(x)
        if isinstance(x, tuple):
            x = x[0]
        return x

class SplitGPUModel(nn.Module):
    def __init__(self, hidden_size=32, use_te_linear=True, dtype=torch.bfloat16, num_blocks=10, use_norm=True):
        super().__init__()
        self.num_blocks = num_blocks
        self.split_idx = num_blocks // 2
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.use_te_linear = use_te_linear
        self.use_norm = use_norm

        print("\nInitializing all blocks on cuda:0...")
        self.blocks = nn.ModuleList([
            TestBlock(
                hidden_size=hidden_size,
                use_te_linear=use_te_linear,
                dtype=dtype,
                is_first_block=(i==0),
                block_idx=i,
                use_norm=use_norm
            )
            for i in range(num_blocks)
        ])
    
    def move_to_devices(self):
        print("\nMoving blocks 5-9 to cuda:1 as a group...")
        split_idx = self.split_idx

        self.blocks[split_idx:].to("cuda:1")  # Move as a group
        
        # Verify devices
        for i, block in enumerate(self.blocks):
            print(f"Block {i} now on {block.linear.weight.device}")
    
    def forward(self, x, return_intermediates=False):
        intermediates = []
        for block in self.blocks:
            x = block(x)
            if return_intermediates:
                intermediates.append(x)
        
        if return_intermediates:
            return x, intermediates
        return x

def _create_checkpoint(num_blocks, hidden_size, dtype):
    checkpoint = {}
    torch.manual_seed(42)
    
    # Create all weights on cuda:0 first
    for i in range(num_blocks):
        is_first = i == 0
        in_size = hidden_size if is_first else hidden_size * 3
        out_size = hidden_size * 3
        checkpoint[f'blocks.{i}.linear.weight'] = torch.randn(
            out_size, in_size, dtype=dtype, device="cuda:0"
        )
        checkpoint[f'blocks.{i}.norm.weight'] = torch.ones(in_size, dtype=dtype, device="cuda:0")
        checkpoint[f'blocks.{i}.norm.bias'] = torch.zeros(in_size, dtype=dtype, device="cuda:0")
    
    return checkpoint

def create_checkpoint(model: nn.Module, output_path: str = "checkpoint.pth"):
    torch.save(model.state_dict(), output_path)
    return output_path

def print_tensor_sample(name, tensor, num_samples=5):
    flat_tensor = tensor.flatten()
    samples = flat_tensor[:num_samples].tolist()
    print(f"{name} first {num_samples} values: {samples}")

def compare_params(te_model, nn_model):
    for name, te_param in te_model.named_parameters():
        nn_param = nn_model.get_parameter(name)

        if not torch.equal(te_param.to("cpu"), nn_param.to("cpu")):
            print(f"{name}: te: {te_param.dtype} {te_param.view(-1)[:5]} nn: {nn_param.dtype} {nn_param.view(-1)[:5]}")
            return False
    return True

def check_keys(te_model, nn_model):
    te_keys = set(te_model.state_dict().keys())
    nn_keys = set(nn_model.state_dict().keys())
    extra_keys = te_keys - nn_keys
    print(f"TE keys: {extra_keys}")
    print(f"NN keys: {nn_keys - te_keys}")

def run_test(num_steps=3, num_blocks=10, hidden_size=32, batch_size=2, seq_len=16, dtype=torch.bfloat16, use_norm=True):
    print("Running test with group device movement...")
    print("\nCreating models...")
    te_model = SplitGPUModel(use_te_linear=True, dtype=dtype, use_norm=use_norm, num_blocks=num_blocks)
    nn_model = SplitGPUModel(use_te_linear=False, dtype=dtype, use_norm=use_norm, num_blocks=num_blocks)
    
    checkpoint_path = create_checkpoint(nn_model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    te_model.load_state_dict(checkpoint, strict=False)
    nn_model.load_state_dict(checkpoint)

    if not compare_params(te_model, nn_model):
        print("Parameters do not match!")
        return
    
    print("\nMoving models to split devices...")
    te_model.move_to_devices()
    nn_model.move_to_devices()

    if not compare_params(te_model, nn_model):
        print("Parameters do not match after moving to split devices!")
        return
    
    print("\nRunning forward passes...")
    with torch.no_grad():
        for step in range(num_steps):
            x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype).to("cuda:0")
            
            te_out, te_intermediates = te_model(x, return_intermediates=True)
            nn_out, nn_intermediates = nn_model(x, return_intermediates=True)
            
            for block_idx, (te_block_out, nn_block_out) in enumerate(zip(te_intermediates, nn_intermediates)):
                diff = (te_block_out - nn_block_out).abs().max().item()
                if diff > 1e-3:
                    print(f"\nStep {step} - Block {block_idx} (on {'cuda:0' if block_idx < te_model.split_idx else 'cuda:1'}):")
                    print(f"Max difference: {diff:.8f}")
                    print(f"TE mean: {te_block_out.abs().mean().item():.8f}")
                    print(f"NN mean: {nn_block_out.abs().mean().item():.8f}")
                    
                    print("\nSample values:")
                    print_tensor_sample("TE output", te_block_out)
                    print_tensor_sample("NN output", nn_block_out)
                    
                    if block_idx > 0:
                        print("\nInput to this block:")
                        print_tensor_sample("TE input", te_intermediates[block_idx-1])
                        print_tensor_sample("NN input", nn_intermediates[block_idx-1])
                    
                    break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_blocks", type=int, default=10)
    parser.add_argument("--use_norm", action="store_true")
    parser.add_argument("--num_steps", type=int, default=3)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    args = parser.parse_args()
    args.dtype = getattr(torch, args.dtype)
    run_test(num_steps=args.num_steps, num_blocks=args.num_blocks, use_norm=args.use_norm, hidden_size=args.hidden_size, batch_size=args.batch_size, seq_len=args.seq_len, dtype=args.dtype)