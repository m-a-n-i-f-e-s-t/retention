# generate outputs using power retention
# python generate_power.py --input "def bubble_sort(arr):"
import torch
import argparse
import threading
import time
import math
import json
from tqdm import tqdm
import models.powercoder  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TextIteratorStreamer, AutoConfig

GREEN = "\033[32m"
BLUE  = "\033[34m"
RED   = "\033[31m"
RESET = "\033[0m"

def parse_args():
    parser = argparse.ArgumentParser(description="Generate with power attention")
    # model config
    parser.add_argument('--model', type=str, default='manifestai/powercoder-3b')
    parser.add_argument('--tokenizer', type=str, default='bigcode/starcoder2-3b')
    parser.add_argument('--chunk-size', type=int, default=None)
    parser.add_argument('--switch-over-seq-len', type=int, default=None)
    parser.add_argument('--disable-sliding-window', action='store_true', default=False)
    # generation config
    parser.add_argument('--max-new-tokens', type=int, default=1024)
    parser.add_argument('--max-hours', type=float, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top-p', type=float, default=0.5)
    parser.add_argument('--repetition-penalty', type=float, default=1.1)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--no-stream', action='store_true', default=False)
    parser.add_argument('--no-eos', action='store_true', default=False)
    parser.add_argument('--stats-only', action='store_true', default=False)
    parser.add_argument('--label', type=str, default=None)
    parser.add_argument('--no-sample', action='store_true', default=False)
    # input text
    parser.add_argument('--input', '-i', type=str, help='Input text to generate from, if not provided, will enter interactive mode')
    parser.add_argument('--input-file', '-f', type=str, help='Input file to generate from')
    return parser.parse_args()

def _generate(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, args: argparse.Namespace, inputs: torch.Tensor):
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True) if not args.no_stream else None
    config = GenerationConfig(
        do_sample=not args.no_sample,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id if not args.no_eos else None,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    if streamer is not None:
        def generate_func():
            with torch.no_grad():
                model.generate(
                    inputs.input_ids,
                    generation_config=config,
                    attention_mask=inputs.attention_mask,
                    streamer=streamer,
                    eos_token_id=tokenizer.eos_token_id if not args.no_eos else None,
                    chunk_size=args.chunk_size,
                    switch_over_seq_len=args.switch_over_seq_len,
                )
        thread = threading.Thread(target=generate_func)
        thread.start()
        step_time = time.time()
        start_time = step_time
        stats = {
            'time_to_first_token': None,
            'tokens_per_second': [],
            'tokens_generated': 0,
        }
        try:
            for text in tqdm(streamer, disable=not args.stats_only, desc="Generating", total=args.max_new_tokens if args.no_eos else None):
                if stats['time_to_first_token'] is None:
                    stats['time_to_first_token'] = time.time() - step_time
                stats['tokens_per_second'].append(1 / (time.time() - step_time))
                step_time = time.time()
                stats['tokens_generated'] += 1
                if not args.stats_only:
                    print(text, end="", flush=True)
                if args.max_hours is not None and (time.time() - start_time) / 3600 > args.max_hours:
                    break
        except Exception as e:
            if "OOM" in str(e) or "CUDA out of memory" in str(e):
                print(f"{RED}Warning: CUDA out of memory{RESET}")
            elif isinstance(e, KeyboardInterrupt):
                print(f"{RED}Warning: Keyboard interrupt{RESET}")
            else:
                raise e
        stats['tokens_per_second'] = stats['tokens_per_second'][1:-1] # remove the first and last token
        if len(stats['tokens_per_second']) == 0:
            print(f"{RED}Warning: No tokens generated{RESET}")
            return
        stats['tokens_per_second_avg'] = sum(stats['tokens_per_second']) / len(stats['tokens_per_second'])
        stats['tokens_per_second_std'] = math.sqrt(sum((x - stats['tokens_per_second_avg'])**2 for x in stats['tokens_per_second']) / len(stats['tokens_per_second']))
        stats['total_time'] = time.time() - start_time
        print(f"{BLUE}Stats:{RESET}")
        print(f"Time to first token: {GREEN}{stats['time_to_first_token']:.2f}s{RESET}\nTokens per second: {GREEN}{stats['tokens_per_second_avg']:.2f} ± {stats['tokens_per_second_std']:.2f}{RESET}\nTokens generated: {GREEN}{stats['tokens_generated']}{RESET}\nPrompt tokens: {GREEN}{inputs.input_ids.shape[1]}{RESET}\nTotal tokens: {GREEN}{inputs.input_ids.shape[1] + stats['tokens_generated']}{RESET}\nTotal time: {GREEN}{stats['total_time']:.2f}s{RESET}")

        if args.stats_only: # write stats to file
            # Build a JSON-safe record
            args_dict = {}
            for k, v in vars(args).items():
                if isinstance(v, (set, tuple)):
                    args_dict[k] = list(v)
                else:
                    try:
                        json.dumps(v)
                        args_dict[k] = v
                    except TypeError:
                        args_dict[k] = str(v)

            record = {
                'time_to_first_token': stats['time_to_first_token'],
                'tokens_per_second': stats['tokens_per_second'],
                'tokens_per_second_avg': stats['tokens_per_second_avg'],
                'tokens_per_second_std': stats['tokens_per_second_std'],
                'tokens_generated': stats['tokens_generated'],
                'prompt_tokens': int(inputs.input_ids.shape[1]),
                'total_tokens': int(inputs.input_ids.shape[1] + stats['tokens_generated']),
                'total_time': stats['total_time'],
                'args': args_dict,
                'model': {
                    'name_or_path': getattr(getattr(model, 'config', None), '_name_or_path', None),
                    'model_type': getattr(getattr(model, 'config', None), 'model_type', None),
                },
                'tokenizer': {
                    'name_or_path': getattr(tokenizer, 'name_or_path', None),
                    'model_max_length': getattr(tokenizer, 'model_max_length', None),
                },
                'label': args.label,
            }
            try:
                with open("generate_stats.json", "r") as f:
                    content = f.read().strip()
                    instances = json.loads(content) if content else []
            except (FileNotFoundError, json.JSONDecodeError):
                instances = []
            instances = instances if isinstance(instances, list) else []
            instances.append(record)
            with open("generate_stats.json", "w") as f:
                json.dump(instances, f, indent=2)
    else:
        outputs = model.generate(
            inputs.input_ids,
            generation_config=config,
            attention_mask=inputs.attention_mask,
            chunk_size=args.chunk_size,
            switch_over_seq_len=args.switch_over_seq_len,
        )
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))

def generate_interactive(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, args: argparse.Namespace):
    while True:
        try:
            input_text = input("Enter input text: ")
        except (EOFError, KeyboardInterrupt):
            break
        if input_text == "exit":
            break
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        _generate(model, tokenizer, args, inputs)

def generate_file(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, args: argparse.Namespace, input_file: str):
    with open(input_file, "r") as f:
        input_texts = ''.join([line for line in f])
        inputs = tokenizer(input_texts, return_tensors="pt").to("cuda")
        _generate(model, tokenizer, args, inputs)

def generate_input(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, args: argparse.Namespace, input: str):
    inputs = tokenizer(input, return_tensors="pt").to("cuda")
    _generate(model, tokenizer, args, inputs)

def main():
    args = parse_args()
    assert not (args.input and args.input_file), "Cannot provide both --input and --input_file"
    print(f"{BLUE}args: {GREEN}{args.__dict__}{RESET}")
    if 'power' in args.model:
        model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
    else:
        config = AutoConfig.from_pretrained(args.model)
        if args.disable_sliding_window:
            config.sliding_window = None
        model = AutoModelForCausalLM.from_pretrained(args.model, config=config, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model.eval()
    model.to("cuda")

    if not args.input and not args.input_file:
        generate_interactive(model, tokenizer, args)
    elif args.input:
        generate_input(model, tokenizer, args, args.input)
    elif args.input_file:
        generate_file(model, tokenizer, args, args.input_file)

if __name__ == "__main__":
    main()

    
   

