import moe_model_files.model_configurations as model_configurations
import moe_model_files.model_utils as model_utils
import data.data_utils as data_utils
import argument_parser as argument_parser

import json
import torch
import sys
import re
from datasets import load_dataset


def extract_writing_prompt(item_list):
    # 1. Access the last dictionary in the list (the GPT response)
    question = item_list[-1]["value"]

    # 2. Check if the header exists and split the string
    header = "Prompt:"
    if header in question:
        # Split at the header and take everything after it, then clean up whitespace
        return question.split(header)[1].strip(' \n"')

    return question


if __name__ == "__main__":
    arguments = argument_parser.parse_arguments()
    root_folder = arguments.root
    model_id = arguments.model_id
    print_logging = arguments.print_logging

    if print_logging:
        print(f"\nPython version: {sys.version}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA build version: {torch.version.cuda}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"First GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Test tensor on GPU: {torch.rand(5).cuda().device}")

    models = [
        "Qwen/Qwen3-30B-A3B-Instruct-2507",  # 0
        "microsoft/Phi-3.5-MoE-instruct",  # 1
        "mistralai/Mixtral-8x7B-Instruct-v0.1",  # 2
        "openai/gpt-oss-20b",  # 3
        "Qwen/Qwen1.5-MoE-A2.7B-Chat",  # 4
        "tencent/Hunyuan-A13B-Instruct",  # 5
        "deepseek-ai/deepseek-moe-16b-chat",  # 6
        "IntervitensInc/pangu-pro-moe-model",  # 7
    ]

    model_config = model_configurations.models[models[model_id]]

    print(f"\nSelected model: {model_config.model_name}")

    ds = load_dataset("openerotica/erotica-analysis")
    questions = ds["train"]["conversations"][12288:]
    questions = [extract_writing_prompt(item) for item in questions]
    questions = list(dict.fromkeys(questions))

    model, tokenizer = model_utils.load_model(models[model_id])

    prompts = data_utils.construct_prompt(tokenizer, questions, model_config.model_name)

    data_utils.create_directory(f"{root_folder}/data/adult_refusal")
    output_filepath = f"{root_folder}/data/adult_refusal/{model_config.model_name}_adult_refusal_prompts.jsonl"

    print("\nStarting generation and saving incrementally...")

    safe_batch_size = 64

    question_index = 0

    for batch_responses in model_utils.generate_output_with_yield(model, model_config.model_name, tokenizer, prompts, batch_size=safe_batch_size):

        batch_refusals = []

        for response in batch_responses:
            # Match the response back to its original question
            original_question = questions[question_index]

            if response.strip().startswith("I'm sorry"):
                batch_refusals.append(original_question)

            question_index += 1

        # Save to file immediately if we found any refusals in this batch
        if batch_refusals:
            with open(output_filepath, "a", encoding="utf-8") as f:
                for refusal in batch_refusals:
                    f.write(json.dumps({"prompt": refusal}) + "\n")

    print("\n------------------ Job Finished ------------------")

    # 1. Read everything into memory and process it
    # filename = output_filepath
    # unique_lines = []
    # seen_data = set()

    # with open(filename, 'r', encoding='utf-8') as f:
    #     for line in f:
    #         line = line.strip()
    #         if not line:
    #             continue
                
    #         try:
    #             # 1. Parse the JSON to handle it as data, not just text
    #             data = json.loads(line)
                
    #             # 2. Convert back to a string with sorted keys.
    #             # This ensures {"a":1, "b":2} and {"b":2, "a":1} are seen as the same.
    #             canonical_json = json.dumps(data, sort_keys=True)
                
    #             # 3. Only keep it if we haven't seen this exact data before
    #             if canonical_json not in seen_data:
    #                 unique_lines.append(canonical_json)
    #                 seen_data.add(canonical_json)
                    
    #         except json.JSONDecodeError:
    #             print(f"Skipping malformed line: {line[:50]}...")

    # # 4. Overwrite the original file with the unique set
    # with open(filename, 'w', encoding='utf-8') as f:
    #     for line in unique_lines:
    #         f.write(line + '\n')

    # print(f"Deduplication complete. Total unique records saved: {len(unique_lines)}")

    #########################################

    # processed_lines = []
    # seen_prompts = set()

    # with open(filename, 'r', encoding='utf-8') as f:
    #     for line in f:
    #         try:
    #             data = json.loads(line)
                
    #             if 'prompt' in data:
    #                 # \n+ matches one or more newline characters
    #                 # This turns \n, \n\n, or \n\n\n into a single " "
    #                 data['prompt'] = re.sub(r'\n+', ' ', data['prompt'])
                    
    #                 # OPTIONAL: If you also want to remove double spaces 
    #                 # that were already there, use \s+ instead:
    #                 # data['prompt'] = re.sub(r'\s+', ' ', data['prompt']).strip()

    #             json_string = json.dumps(data, sort_keys=True)
                
    #             if json_string not in seen_prompts:
    #                 processed_lines.append(json_string)
    #                 seen_prompts.add(json_string)
    #         except json.JSONDecodeError:
    #             continue

    # with open(filename, 'w', encoding='utf-8') as f:
    #     for line in processed_lines:
    #         f.write(line + '\n')

    # print(f"Successfully cleaned newlines and deduplicated {filename}!")
