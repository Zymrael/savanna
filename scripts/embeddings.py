import pickle

import torch

bs = 1
KB = 2 ** 10
seqlen = 1024 * KB
d = 4096
MB = 2 ** 20
dtype = torch.bfloat16

vocab_size = 512
torch.cuda.memory._record_memory_history(max_entries=100000)
mem_alloc = torch.cuda.memory_allocated() / MB
print(f"DEBUG::MEM_ALLOC:START {mem_alloc:.2f} MB")
embeddings = torch.nn.Embedding(vocab_size, d, dtype=dtype, device="cuda")
input_ids = torch.randint(0, vocab_size, (bs, seqlen), dtype=torch.long, device="cuda")
mem_alloc = torch.cuda.memory_allocated() / MB
print(f"DEBUG::MEM_ALLOC:EMBEDDINGS_INIT {mem_alloc:.2f} MB")
words_embeddings = embeddings(input_ids)
mem_alloc = torch.cuda.memory_allocated() / MB
print(f"DEBUG::MEM_ALLOC:WORDS_EMBEDDINGS_OUT {mem_alloc:.2f} MB")
print(f"DEBUG::MEM_ALLOC:WORDS_EMBEDDINGS_OUT_DIFF {(words_embeddings.numel() * 2) / MB:.2f} MB")
output = "embeddings_mem.pkl"
with open(output, "wb") as f:
    pickle.dump(torch.cuda.memory._snapshot(), f)

print(f"DEBUG::WORDS_EMBEDDINGS: {words_embeddings.shape} {words_embeddings.dtype}")
