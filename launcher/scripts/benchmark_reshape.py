import torch
from triton.testing import do_bench

B = 4
L = 8192
D = 256
G = 3 * 16

device = torch.device("cuda")

x = torch.randn(B, L, G, D, device=device)

def naive_interleave(x):
    x1 =  x[:, :, 0::3, :]
    x2 = x[:, :, 1::3, :]
    x3 = x[:, :, 2::3, :]
    return x1, x2, x3

def reshape_interleave(x): 
    x = x.reshape(B, L, G // 3, 3, D)
    return x[:, :, :, 0], x[:, :, :, 1], x[:, :, :, 2]

y_naive = naive_interleave(x)
y_reshape = reshape_interleave(x)
assert y_naive[0].shape == torch.Size([B, L, G // 3, D])
# print(y_naive[0])
# print(y_reshape[0])

print('---')

x1, x2, v = y_naive
for label, y_ in zip(['x1','x2','v'], y_naive):
    print(label, y_.shape)
    
print('---')


assert torch.allclose(y_naive[0], y_reshape[0])
assert torch.allclose(y_naive[1], y_reshape[1])
assert torch.allclose(y_naive[2], y_reshape[2])


t_naive = do_bench(lambda: naive_interleave(x))
t_interleave = do_bench(lambda: reshape_interleave(x))

print(f"Naive: {t_naive} ms")
print(f"Interleave: {t_interleave} ms")

x1 = torch.empty_like(x1)
x2 = torch.empty_like(x2)
v = torch.empty_like(v)

