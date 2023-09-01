import torch
from einops import rearrange
from dataclasses import dataclass, field
from typing import Optional
from torch import Tensor


@dataclass
class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    max_sequence_len: int
    max_batch_size: int
    sequence_len_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    fused_ft_kernel: bool = False
    lengths_per_sample: Optional[Tensor] = None


# https://github.com/NVIDIA/Megatron-LM/blob/0bb597b42c53355a567aba2a1357cc34b9d99ddd/megatron/text_generation/sampling.py
# https://github.com/huggingface/transformers/blob/a44985b41cfa2de48a5e1de7f1f93b7483da25d1/src/transformers/generation/logits_process.py#L170
def modify_logits_for_top_p_filtering(logits, top_p):
    """Set the logits for none top-p values to -inf."""
    if top_p <= 0.0:
        return
    # First sort and calculate cumulative sum of probabilities.
    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, float("-inf"))


# https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/utils/generation.py
def sample(logits, top_k=1, top_p=0.0, temperature=1.0):
    """Sample from top-k logits.
    Arguments:
        logits: Tensor of shape (batch_size, vocab_size)
    """
    if top_k == 1:  # Short-circuit for greedy decoding
        return logits.argmax(dim=-1)
    else:
        if top_p > 0.0:
            assert top_p <= 1.0, "top-p should be in (0, 1]."
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))  # Safety check
            logits_top, indices = torch.topk(logits, top_k, dim=-1)
            logits_top /= temperature
            modify_logits_for_top_p_filtering(logits_top, top_p)
            return indices[
                torch.arange(indices.shape[0], device=indices.device),
                torch.multinomial(
                    torch.softmax(logits_top, dim=-1), num_samples=1
                ).squeeze(dim=-1),
            ]
        else:
            logits_top = logits / temperature
            modify_logits_for_top_p_filtering(logits_top, top_p)
            return torch.multinomial(
                torch.softmax(logits_top, dim=-1), num_samples=1
            ).squeeze(dim=-1)


def logits_from_lm(
    model,
    input_ids,
    position_ids=None,
    attention_mask=None,
    pass_inference_params=False,
    **kwargs
):
    if pass_inference_params:
        out = model(input_ids, **kwargs)
    else:
        x = (input_ids, attention_mask, position_ids)
        out = model(x)

    if type(out) == torch.Tensor:
        logits = out
    else:
        logits = out[0].logits if type(out) == tuple else out.logits
    return logits


def generate(
    model,
    input_ids,
    max_seq_len,
    max_new_tokens,
    top_k=1,
    top_p=0.0,
    temperature=1.0,
    kv_caching=False,
    pass_inference_params=False,
    output_scores=False,
    eos_token_id=-1,
    attention_mask=None,
    position_ids=None,
    **kwargs
):
    """
    Generation for the purpose of benchmarking, must generate max_new_tokens number of tokens
    Arguments:
        input_ids: (batch, seq_len)
        max_seq_leng: int
    Returns:
        output_ids: (batch, seq_len + max_new_tokens)
    """
    early_exit = False

    if type(input_ids) == list:
        input_ids = torch.cat(input_ids, dim=0)[None]

    batch_size, seqlen_og = input_ids.shape

    if kv_caching:
        inference_params = InferenceParams(
            max_sequence_len=max_seq_len, max_batch_size=batch_size
        )
    else:
        inference_params = None

    # ensure that the input ids are less than the max sequence length
    input_ids_cond = (
        input_ids if input_ids.size(1) <= max_seq_len else input_ids[:, -max_seq_len:]
    )

    with torch.inference_mode():
        logits = logits_from_lm(
            model,
            input_ids_cond,
            inference_params=inference_params,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pass_inference_params=pass_inference_params,
        )[:, -1]

        next_token = sample(logits, top_k=top_k, top_p=top_p, temperature=temperature)
        next_token = rearrange(next_token, "b -> b 1")
        input_ids = torch.cat((input_ids, next_token), dim=1)
        sequence_len_offset = seqlen_og

        if output_scores:
            score = torch.zeros(
                batch_size, max_new_tokens, logits.shape[-1], device=input_ids.device
            )
            score[:, 0] = logits

        for n in range(max_new_tokens - 1):
            if kv_caching:
                inference_params.sequence_len_offset = min(
                    max_seq_len - 1, sequence_len_offset - 1
                )
                input_ids_cond = next_token
                position_ids = torch.full(
                    (batch_size, 1),
                    sequence_len_offset,
                    dtype=torch.long,
                    device=input_ids.device,
                )
            else:
                input_ids_cond = torch.cat((input_ids_cond, next_token), dim=1)
                position_ids = None

            input_ids_cond = (
                input_ids_cond
                if input_ids_cond.size(1) <= max_seq_len
                else input_ids_cond[:, -max_seq_len:]
            )

            logits = logits_from_lm(
                model,
                input_ids_cond,
                position_ids=position_ids,
                inference_params=inference_params,
                attention_mask=attention_mask,
                pass_inference_params=pass_inference_params,
            )[:, -1]

            if output_scores:
                score[:, n + 1] = logits

            next_token = sample(logits, top_k=top_k, temperature=temperature)
            next_token = rearrange(next_token, "b -> b 1")

            # HACK: only works for batch size 1
            if next_token.shape[0] == 1:
                if next_token.item() == eos_token_id:
                    early_exit = True
                    break
            sequence_len_offset += 1

            input_ids = torch.cat((input_ids, next_token), dim=1)

    if not early_exit:
        assert input_ids.shape[1] == seqlen_og + max_new_tokens

    return input_ids if not output_scores else (input_ids, score)


if __name__ == "__main__":
    import sys
    from savanna.text_generation_utils import generate_samples_from_prompt

    sys.path.append("/var/cr01_data/mpoli/code/safari-neox")
    from savanna.loading import load_checkpoint
    import yaml

    print("Loading checkpoint")

    with open("./configs/hyena/reference/evals/test_load.yml", "r") as file:
        test_cfg = yaml.load(file, Loader=yaml.FullLoader)

    print(test_cfg)

    model, tokenizer = load_checkpoint(test_cfg, to_sequential=True)
    model = model.cuda()
    print(model)
    print(tokenizer)
    print("Done loading")

    # safari generation util
    for trial in range(1):
        print("Generating...")
        input_ids = tokenizer.tokenize(
            "The man in the blue shirt sits on the chair next to the sink. The other man begins washing his hair. he then walks over to the sink and smiles while shaking his wet hair."
        )
        print(input_ids)

        devices = [p.device for p in model.sequential.parameters()]

        assert all([d == devices[0] for d in devices])
        device = devices[0]

        dtype = next(model.sequential.parameters()).dtype
        for p in model.sequential.parameters():
            p.data = p.data.to(dtype)

        input_ids = torch.tensor(input_ids, dtype=torch.long).to(device).unsqueeze(0)
        position_ids = torch.arange(input_ids.shape[-1]).to(device).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids, dtype=dtype)
        x = (input_ids, attention_mask, position_ids)

        output_ids = generate(
            model,
            input_ids,
            max_seq_len=256,
            max_new_tokens=12,
            kv_caching=False,
            output_scores=True,
        )
        if type(output_ids) == tuple:
            output_ids, scores = output_ids
            print(scores.shape, output_ids.shape)
            print(scores)
            print("Scores for output")
            import pdb

            pdb.set_trace()
            print(
                scores[:, :, output_ids[:, -scores.shape[1] :]],
                scores[:, :, output_ids[:, -scores.shape[1] :]].shape,
            )

        print(output_ids)
        print(tokenizer.detokenize(output_ids[0].tolist()))
        print("Done generating")

    # neox generation util
    from savanna.neox_arguments import GlobalConfig
    from savanna.checkpointing import load_checkpoint
    from savanna.model.gpt2_model import GPT2ModelPipe
    from savanna.tokenizer import build_tokenizer

    global_config = GlobalConfig(**test_cfg)
    # [MP]: rough edge: global_vocab_size is set inside build tokenizer, but is required
    # during model init
    tokenizer = build_tokenizer(global_config)
    # model = GPT2ModelPipe(global_config, num_tokentypes=0, parallel_output=True)
    # _ = load_checkpoint(global_config, model, optimizer=None, lr_scheduler=None)
    global_config.is_pipe_parallel = False

    # generated_texts = generate_samples_from_prompt(
    #     global_config=global_config,
    #     model=model,
    #     tokenizer=tokenizer,
    #     text="Honey Badger is a promising new language modeling architecture based on Natural Language Reasoning.",
    #     recompute=True,
    # )
    # print(generated_texts)
