# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import logging
import torch
import sys
import os
import json
import re
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from transformers.generation import GenerationConfig

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from peft import PeftModel

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to baseline model",
        required=True,
    )
    parser.add_argument(
        "--lora_name_or_path",
        type=str,
        default="",
        help="Path to pretrained model"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to evaluation dataset",
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save output results",
        required=True,
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_beam_groups",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=4,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--penalty_alpha",
        type=float,
        default=0.6,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help='Specify num of return sequences',
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help='Specify num of return sequences',
    )
    parser.add_argument("--language",
                        type=str,
                        default="English",
                        choices=["English", "Chinese", "Japanese"])

    args = parser.parse_args()

    return args

def ChatGLMPrompt(instruction, inputs):
    return f"[Round 1]\n\n问：{instruction}\n{inputs}\n\n答："

def prompt_eval(args, model, tokenizer, data):
    for prompt in data:

        print("========== Prompt =========")
        print(f"\n{prompt['instruction']}\n{prompt['input']}\n")

        print("========== Ground Truth =========")
        print(f"\n{prompt['output']}\n")

        print(f'========== StockGPT =========')
        input_text = ChatGLMPrompt(prompt['instruction'], prompt['input'])
        # chat model
        output, history = model.chat(tokenizer, input_text, history=[])
        # base model
        # inputs = tokenizer(input_text, return_tensors='pt')
        # inputs = inputs.to(model.device)
        # pred = model.generate(**inputs)
        # output = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
        # if len(output) > len(input_text):
        #     output = output[len(input_text):]
        print(f'\n{output}\n')

        print("====================prompt end=============================\n\n")

        with open(args.output_path, 'a+') as f:
            result = prompt
            result.update({"stockgpt": output})
            f.write(json.dumps(result) + '\n')


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model.generation_config = GenerationConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if len(args.lora_name_or_path) > 0:
        model = PeftModel.from_pretrained(model, args.lora_name_or_path)
    device = torch.device('cuda:0')

    model.to(device)
    
    with open(args.data_path, 'r') as f:
        prompts = json.load(f)
    
    print(f'Loading dataset')
    data = []
    for example in tqdm(prompts):
        line = example['input']
        name_ptn = r'这是以(\S+?)（([0-9]+?)）.*'
        date_ptn = r'.*在([0-9]{4}-[0-9]{2}-[0-9]{2} 00:00:00)日期发布的研究报告'
        name_tmp = re.findall(name_ptn, line)
        stock_name, stock_code = name_tmp[0]
        date = re.findall(date_ptn, line)[0]
        data.append({
            "instruction": example['instruction'],
            'input': example['input'],
            'output': example['output'],
            'stock_name': stock_name,
            'stock_code': stock_code,
            'date': date
        })

    prompt_eval(args, model, tokenizer, data)


if __name__ == "__main__":
    main()
