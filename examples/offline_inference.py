from vllm import LLM, SamplingParams

import os

import json
from typing import List

# Reference: https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json
ALPACA_DATASET_JSON_FILE = "alpaca_data.json"

NON_EMPTY_INPUT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

EMPTY_INPUT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""

def load_alpaca_dataset(json_file: str = ALPACA_DATASET_JSON_FILE) -> (List[str], List[str]):
    prompts = []
    outputs = []
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    with open(os.path.join(current_dir, json_file), "r") as file:
        data = json.load(file)

        if isinstance(data, list):
            for item in data:
                instruction, input, output = item.get("instruction"), item.get("input"), item.get("output")
                # print(f"Instruction: {instruction}")
                # print(f"Input: {input}")
                # print(f"Output: {output}")
                if len(input) > 0:
                    prompt = NON_EMPTY_INPUT_TEMPLATE.format(instruction=instruction, input=input)
                else:
                    prompt = EMPTY_INPUT_TEMPLATE.format(instruction=instruction)
                # print(f"Prompt: {prompt}")
                prompts.append(prompt)
                outputs.append(output)
                # print("---")
    return prompts, outputs

prompts, _ = load_alpaca_dataset()
# prompts = prompts[:1000]

# Sample prompts.
# prompts = [
#     "Hello, my name is",
#     "The president of the United States is",
#     "The capital of France is",
#     "The future of AI is",
# ]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)

# Create an LLM.
# llm = LLM(model="facebook/opt-125m")
model_path = "/workspace/model"
llm = LLM(model=model_path, tokenizer=model_path, disable_log_stats=True)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
