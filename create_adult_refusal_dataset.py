import moe_model_files.model_configurations as model_configurations
import moe_model_files.model_utils as model_utils
import data.data_utils as data_utils
import argument_parser as argument_parser

import json
import torch
import sys
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
    questions = ds["train"]["conversations"]
    questions = [extract_writing_prompt(item) for item in questions]
    questions = list(dict.fromkeys(questions))

    model, tokenizer = model_utils.load_model(models[model_id])

    prompts = data_utils.construct_prompt(tokenizer, questions, model_config.model_name)

    data_utils.create_directory(f"{root_folder}/data/adult_refusal")
    output_filepath = f"{root_folder}/data/adult_refusal/{model_config.model_name}_adult_refusal_prompts.jsonl"

    # output_filepath = f"{root_folder}/data/adult_refusal/adult_refusal_prompts_total.jsonl"

    # result = []
    # with open(output_filepath, "a", encoding="utf-8") as f:
    #     for q in questions:
    #         f.write(json.dumps({"prompt": q}) + "\n")

    # dd

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
