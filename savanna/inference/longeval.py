import argparse
import os
import json
from tqdm import tqdm

import yaml
import re
import torch
import transformers
from savanna.inference.load import load_checkpoint
from savanna.inference.generation import generate


def get_output_dir(args):
    path = args.model_name_or_path

    if path[-1] == "/":
        path = path[:-1]
    name = path.split("/")[-1]

    output_dir = f"evaluation/{args.task}/predictions/{name}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"output to {output_dir}")
    return output_dir


def longeval_load_model(args):
    if "hhyena-1.5b-8k" in args.model_name_or_path:
        with open(args.config_file, "r") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        print(config)
        model, tokenizer = load_checkpoint(config, to_sequential=True)
        model = model.cuda()
        dtype = next(model.sequential.parameters()).dtype
        for p in model.sequential.parameters():
            p.data = p.data.to(dtype)

    elif "mosaicml/mpt-7b-storywriter" in args.model_name_or_path:
        # Adapt from: https://huggingface.co/mosaicml/mpt-7b-storywriter
        # filter_string()
        config = transformers.AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        config.attn_config["attn_impl"] = "triton"

        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            config=config,
            torch_dtype=torch.bfloat16,  # Load model weights in bfloat16
            trust_remote_code=True,
        )

        tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    elif "mosaicml/mpt-30b-chat" in args.model_name_or_path:
        config = transformers.AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            max_seq_len=16384,
            device_map="auto",
            max_memory={i: f"{args.max_gpu_memory}GiB" for i in range(args.num_gpus)},
            torch_dtype=torch.float16,
        )
        model.attn_impl = "triton"

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            use_fast=True,
            model_max_length=16384,
        )
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
    elif "THUDM/chatglm2-6b" in args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_name_or_path, trust_remote_code=True
        )
        model = (
            transformers.AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)
            .half()
            .cuda()
        )
        model = model.eval()
    elif "gpt-" in args.model_name_or_path:
        tokenizer = None
        model = None
    elif "claude" in args.model_name_or_path:
        tokenizer = None
        model = None
    else:
        raise NotImplementedError
    return model, tokenizer


def load_testcases(test_file):
    with open(test_file, "r") as json_file:
        json_list = list(json_file)

    test_cases = []
    for test_case in json_list:
        test_case = json.loads(test_case)
        test_cases.append(test_case)

    return test_cases


def test_topics_one_sample(model, tokenizer, test_case, output_file, idx, args):
    prompt = test_case["prompt"]
    topics = test_case["topics"]

    if "hhyena-1.5b-8k" in args.model_name_or_path:
        device = next(model.parameters()).device
        input_ids = tokenizer.tokenize(prompt)
        input_ids = torch.tensor(input_ids, dtype=torch.long).to(device).unsqueeze(0)
        prompt_length = len(input_ids[0])
        new_tokens = 42
        output = generate(
            model,
            input_ids,
            max_seq_len=8192,
            max_new_tokens=new_tokens,
            kv_caching=False,
        )[:, -new_tokens:]
        output = tokenizer.detokenize(output[0].tolist())

    elif "mosaicml/mpt-7b-storywriter" in args.model_name_or_path:
        from transformers import pipeline

        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cuda:0")
        # Use next word prediction to get storywriter answer
        prompt += "\n ASSISTANT: The first topic is"
        prompt_length = len(tokenizer(prompt).input_ids)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = pipe(prompt, max_new_tokens=15, do_sample=True, use_cache=True)[0]["generated_text"][
                len(prompt) :
            ]
    elif "THUDM/chatglm2-6b" in args.model_name_or_path:
        prompt_length = len(tokenizer(prompt).input_ids)
        output, _ = model.chat(tokenizer, prompt, history=[], max_length=16384)
        output = [output]
    elif "gpt-" in args.model_name_or_path:
        prompt_length, output = retrieve_from_openai(prompt, args.model_name_or_path)
    elif "claude" in args.model_name_or_path:
        prompt_length, output = retrieve_from_anthropic(prompt, args.model_name_or_path)
    else:
        if "longchat" in args.model_name_or_path:
            conv = get_conversation_template("vicuna")
        else:
            conv = get_conversation_template(args.model_name_or_path)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input = tokenizer(prompt, return_tensors="pt")
        prompt_length = input.input_ids.size()[-1]

        # Disable use_cache if using longchat models with flash attention
        use_cache = not ("longchat" in args.model_name_or_path and args.longchat_flash_attn)

        output = model.generate(input.input_ids.to(model.device), max_new_tokens=50, use_cache=use_cache)[0]
        output = output[prompt_length:]
        output = tokenizer.batch_decode([output], skip_special_tokens=True)

    summary = f"Label: {topics[0]}, Predict: {output}, prompt length: {prompt_length}".replace("\n", " ")
    print(summary)
    if idx == 0:
        with open(output_file, "w") as f:
            f.write(summary)
            f.write("\n")
    else:
        with open(output_file, "a+") as f:
            f.write(summary)
            f.write("\n")

    return None, prompt_length, summary


def test_lines_one_sample(model, tokenizer, test_case, output_file, idx, args):
    prompt = test_case["prompt"]
    correct_line = test_case["correct_line"]
    expected_number = test_case["expected_number"]

    if "hhyena-1.5b-8k" in args.model_name_or_path:
        device = next(model.parameters()).device
        input_ids = tokenizer.tokenize(prompt)
        input_ids = torch.tensor(input_ids, dtype=torch.long).to(device).unsqueeze(0)
        prompt_length = len(input_ids[0])
        output = generate(
            model,
            input_ids,
            max_seq_len=8192,
            max_new_tokens=32,
            kv_caching=False,
        )
        output = tokenizer.detokenize(output[0].tolist())

    elif "mosaicml/mpt-7b-storywriter" in args.model_name_or_path:
        from transformers import pipeline

        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cuda:0")
        # Use next word prediction to get storywriter answer
        prompt += f'Line <{test_case["random_idx"][0]}>: <REGISTER_CONTENT> is'
        prompt_length = len(tokenizer(prompt).input_ids)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = pipe(prompt, max_new_tokens=15, do_sample=True, use_cache=True)[0]["generated_text"][
                len(prompt) :
            ]
    elif "THUDM/chatglm2-6b" in args.model_name_or_path:
        prompt_length = len(tokenizer(prompt).input_ids)
        output, _ = model.chat(tokenizer, prompt, history=[], max_length=16384)
    elif "gpt-" in args.model_name_or_path:
        prompt_length, output = retrieve_from_openai(prompt, args.model_name_or_path)
    elif "claude" in args.model_name_or_path:
        prompt_length, output = retrieve_from_anthropic(prompt, args.model_name_or_path)
    else:
        if "longchat" in args.model_name_or_path:
            conv = get_conversation_template("vicuna")
        else:
            conv = get_conversation_template(args.model_name_or_path)
        print(f"Using conversation template: {conv.name}")

        if "mosaicml/mpt-30b-chat" in args.model_name_or_path:
            prompt += f'Answer in the format <{test_case["random_idx"][0]}> <REGISTER_CONTENT>.'

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input = tokenizer(prompt, return_tensors="pt")
        prompt_length = input.input_ids.shape[-1]

        # Disable use_cache if using longchat models with flash attention
        use_cache = not ("longchat" in args.model_name_or_path and args.longchat_flash_attn)

        output = model.generate(input.input_ids.to(model.device), max_new_tokens=100, use_cache=use_cache)[0]
        output = output[prompt_length:]
        output = tokenizer.batch_decode([output], skip_special_tokens=True)[0]

    # Matching the last digit of the model output
    response_number = re.findall("\d+", output)
    if response_number is not None and len(response_number) > 0:
        response_number = int(response_number[-1])
    else:
        print("Got unparsable result")
        response_number = -1

    summary = f"Label: {expected_number}, Predict: {output}, Parsed: {response_number}, prompt length: {prompt_length}".replace(
        "\n", " "
    )
    print(summary)
    if idx == 0:
        with open(output_file, "w") as f:
            f.write(summary)
            f.write("\n")
    else:
        with open(output_file, "a+") as f:
            f.write(summary)
            f.write("\n")

    return expected_number == response_number, prompt_length, summary


def longeval_test(model, tokenizer, output_dir, args):
    if args.task == "topics":
        for num_topics in [5, 10, 15, 20, 25]:
            print(f"************ Start testing {num_topics} topics per prompt ***********")
            avg_length = 0

            test_file = os.path.join(args.test_dir, f"topics/testcases/{num_topics}_topics.jsonl")
            output_file = os.path.join(output_dir, f"{num_topics}_response.txt")

            test_cases = load_testcases(test_file)
            print(len(test_cases))
            for idx, test_case in tqdm(enumerate(test_cases)):
                _, prompt_length, summary = test_topics_one_sample(
                    model=model,
                    tokenizer=tokenizer,
                    test_case=test_case,
                    output_file=output_file,
                    idx=idx,
                    args=args,
                )
                avg_length += prompt_length / len(test_cases)

            print(
                f"************ Finish testing {num_topics} topics per prompt with average prompt length {avg_length} ************"
            )
            if args.eval_shortest_only:
                break

    elif args.task == "lines":
        for num_lines in [200, 300, 400, 500, 600, 680]:
            print(f"************ Start testing {num_lines} lines per LRT prompt ************")
            test_file = os.path.join(args.test_dir, f"lines/testcases/{num_lines}_lines.jsonl")

            output_file = os.path.join(output_dir, f"{num_lines}_response.txt")
            num_correct = 0
            avg_length = 0

            test_cases = load_testcases(test_file)
            for idx, test_case in tqdm(enumerate(test_cases)):
                correct, prompt_length, summary = test_lines_one_sample(
                    model=model,
                    tokenizer=tokenizer,
                    test_case=test_case,
                    output_file=output_file,
                    idx=idx,
                    args=args,
                )
                avg_length += prompt_length / len(test_cases)
                num_correct += correct
            accuracy = num_correct / len(test_cases)

            with open(output_file, "a+") as f:
                f.write(f"Accuracy: {accuracy}")

            print(
                f"************ Finish testing {num_lines} lines per prompt with average prompt length {avg_length}, accuracy: {accuracy} ************"
            )
            if args.eval_shortest_only:
                break
    else:
        print(f"Unsupported task: {args.task}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True, help="model path")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Which evaluation task to use. currently support [topics, lines]",
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="number of gpus to use")
    parser.add_argument(
        "--max_gpu_memory",
        type=int,
        default=40,
        help="max per gpu memory in GiB. A100 is 40 or 80.",
    )
    parser.add_argument(
        "--longchat_flash_attn",
        action="store_true",
        help="Only apply to longchat models. Whether to enable flash attention to save memory, but slower.",
    )
    parser.add_argument(
        "--longchat_ratio",
        type=int,
        default=8,
        help="Only apply to longchat models. Use ratio=8 for 16K context length model. Only ratio=8 is supported now.",
    )
    parser.add_argument(
        "--eval_shortest_only",
        action="store_true",
        default=0,
        help="Only eval the shortest case for illustration purpose",
    )
    parser.add_argument("--test_dir", type=str, default="evaluation", help="Directory of the testcases")
    parser.add_argument("--config_file", type=str, default="configs", help="")
    args = parser.parse_args()

    output_dir = get_output_dir(args)

    model, tokenizer = longeval_load_model(args)
    longeval_test(model, tokenizer, output_dir, args)
