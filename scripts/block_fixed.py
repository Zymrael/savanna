import torch
import torch.nn as nn
from vortex.model.layers import TELinear


class TestBlock(nn.Module):
    def __init__(self, hidden_size, use_te_linear=True, dtype=torch.bfloat16, is_first_block=True, block_idx=0):
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
            
        self.norm = nn.LayerNorm(input_size).to(device="cuda:0", dtype=dtype)
    
    def forward(self, x):
        device = self.linear.weight.device
        x = x.to(device=device, dtype=self.dtype)
        x = self.norm(x)
        x = self.linear(x)
        if isinstance(x, tuple):
            x = x[0]
        return x

class SplitGPUModel(nn.Module):
    def __init__(self, hidden_size=32, use_te_linear=True, dtype=torch.bfloat16, num_blocks=10):
        super().__init__()
        
        print("\nInitializing all blocks on cuda:0...")
        self.blocks = nn.ModuleList([
            TestBlock(
                hidden_size=hidden_size,
                use_te_linear=use_te_linear,
                dtype=dtype,
                is_first_block=(i==0),
                block_idx=i
            )
            for i in range(num_blocks)
        ])
    
    def move_to_devices(self):
        print("\nMoving blocks 5-9 to cuda:1 as a group...")
        self.blocks[5:].to("cuda:1")  # Move as a group
        
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

def create_checkpoint(hidden_size, dtype):
    checkpoint = {}
    torch.manual_seed(42)
    
    # Create all weights on cuda:0 first
    for i in range(10):
        is_first = i == 0
        in_size = hidden_size if is_first else hidden_size * 3
        
        checkpoint[f'blocks.{i}.linear.weight'] = torch.randn(
            hidden_size * 3, in_size, dtype=dtype, device="cuda:0"
        )
        checkpoint[f'blocks.{i}.norm.weight'] = torch.ones(in_size, dtype=dtype, device="cuda:0")
        checkpoint[f'blocks.{i}.norm.bias'] = torch.zeros(in_size, dtype=dtype, device="cuda:0")
    
    return checkpoint

def print_tensor_sample(name, tensor, num_samples=5):
    flat_tensor = tensor.flatten()
    samples = flat_tensor[:num_samples].tolist()
    print(f"{name} first {num_samples} values: {samples}")

def run_test(num_steps=3):
    print("Running test with group device movement...")
    hidden_size = 32
    batch_size = 2
    seq_len = 16
    dtype = torch.bfloat16
    
    print("\nCreating models...")
    te_model = SplitGPUModel(use_te_linear=True, dtype=dtype)
    nn_model = SplitGPUModel(use_te_linear=False, dtype=dtype)
    
    print("\nLoading checkpoint...")
    checkpoint = create_checkpoint(hidden_size, dtype)
    te_model.load_state_dict(checkpoint,strict=False)
    nn_model.load_state_dict(checkpoint)
    
    print("\nMoving models to split devices...")
    te_model.move_to_devices()
    nn_model.move_to_devices()
    
    print("\nRunning forward passes...")
    with torch.no_grad():
        for step in range(num_steps):
            x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype).to("cuda:0")
            
            te_out, te_intermediates = te_model(x, return_intermediates=True)
            nn_out, nn_intermediates = nn_model(x, return_intermediates=True)
            
            for block_idx, (te_block_out, nn_block_out) in enumerate(zip(te_intermediates, nn_intermediates)):
                diff = (te_block_out - nn_block_out).abs().max().item()
                if diff > 1e-3:
                    print(f"\nStep {step} - Block {block_idx} (on {'cuda:0' if block_idx < 5 else 'cuda:1'}):")
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
    run_test()