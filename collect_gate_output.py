import moe_model_files.model_configurations as model_configurations
import moe_model_files.model_utils as model_utils
import data.data_utils as data_utils
import argument_parser as argument_parser

import torch
import numpy as np
import sys
import inspect

from collections import defaultdict
from tqdm import tqdm


def find_numpy_indices(input_tokens_question, input_tokens_prompt, print_logging):
    len_input_tokens_question = input_tokens_question.shape[0]
    len_input_tokens_prompt = input_tokens_prompt.shape[0]
    potential_starts = np.where(input_tokens_prompt == input_tokens_question[0])[0]

    if print_logging:
        print(f"potential_starts: {potential_starts}")

    for start in potential_starts:
        if start + len_input_tokens_question <= len_input_tokens_prompt:
            if np.array_equal(
                input_tokens_prompt[start : start + len_input_tokens_question],
                input_tokens_question,
            ):
                end = start + len_input_tokens_question - 1
                if print_logging:
                    print(f"start, end: {start, end}")
                return start, end

                
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

    model, tokenizer = model_utils.load_model(models[model_id])

    questions, labels = data_utils.load_twinset(root_folder, malicious_only=False)  # label 1 = refusal behavior

    hook_handles, top_k_expert_indices = model_utils.register_activation_hooks(model_config.model_name, model, model_config.top_k, model_config.gate_name)

    batch_size = 32

    prompts = data_utils.construct_prompt(tokenizer, questions, model_config.model_name)

    total_batches = (len(prompts) + batch_size - 1) // batch_size

    for batch_prompts in tqdm(data_utils.batchify(prompts, batch_size), total=total_batches):
        # Tokenize the batch
        input_tokens = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

        if "token_type_ids" in input_tokens:
            forward_args = inspect.signature(model.forward).parameters
            if "token_type_ids" not in forward_args:
                input_tokens.pop("token_type_ids")

        model.eval()
        with torch.no_grad():
            _ = model(**input_tokens)

    # Remove hooks to clean up
    for handle in hook_handles:
        handle.remove()

    if print_logging:
        print("\n-----------------------------------------------------")
        print("-----------------------------------------------------")
        for layer_name in top_k_expert_indices:
            print(f"Number of batches recorded in layer '{layer_name}': '{len(top_k_expert_indices[layer_name])}'")
            print(f"top_k_expert_indices[{layer_name}][0]: '{top_k_expert_indices[layer_name][0]}'")

            for batch_index, batch_expert_indices in enumerate(top_k_expert_indices[layer_name]):
                print(f"Layer: '{layer_name}', batch_index: '{batch_index}', shape: '{batch_expert_indices.shape}'")
        print("-----------------------------------------------------")
        print("-----------------------------------------------------")

    for batch_index, batch_prompts in enumerate(tqdm(data_utils.batchify(prompts, batch_size), total=total_batches)):
        input_ids = tokenizer(batch_prompts, return_tensors="np", padding=True, truncation=True)["input_ids"]
        dynamic_batch_size, dynamic_seq_len = input_ids.shape[0], input_ids.shape[1]

        for layer_name in top_k_expert_indices:
            top_k_expert_indices[layer_name][batch_index] = top_k_expert_indices[layer_name][batch_index].view(
                dynamic_batch_size,
                dynamic_seq_len,
                top_k_expert_indices[layer_name][batch_index].shape[1],
            )
        continue

    if print_logging:
        print("\n-----------------------------------------------------")
        print("-----------------------------------------------------")
        for layer_name in top_k_expert_indices:
            print(f"Number of batches recorded in layer '{layer_name}': '{len(top_k_expert_indices[layer_name])}'")
            print(f"top_k_expert_indices[{layer_name}][0]: '{top_k_expert_indices[layer_name][0]}'")

            for batch_index, batch_expert_indices in enumerate(top_k_expert_indices[layer_name]):
                print(f"Layer: '{layer_name}', batch_index: '{batch_index}', shape: '{batch_expert_indices.shape}'")
        print("-----------------------------------------------------")
        print("-----------------------------------------------------")

    filtered_expert_choices_per_token = defaultdict(list)

    for batch_index, batch_prompts in enumerate(tqdm(data_utils.batchify(prompts, batch_size), total=total_batches)):
        input_tokens = tokenizer(batch_prompts, return_tensors="np", padding=True, truncation=True)["input_ids"]

        for prompt_index, input_tokens_prompt in enumerate(input_tokens):
            if model_config.model_name == "deepseek-moe-16b-chat":
                question = " " + questions[(batch_index * batch_size) + prompt_index]
                input_tokens_question = tokenizer(
                    question,
                    return_tensors="np",
                    add_special_tokens=False,
                    padding=False,
                    truncation=False,
                )[
                    "input_ids"
                ][0]
            elif model_config.model_name == "Mixtral-8x7B-Instruct-v0.1":
                question = "\n" + questions[(batch_index * batch_size) + prompt_index]
                input_tokens_question = tokenizer(
                    question,
                    return_tensors="np",
                    add_special_tokens=False,
                    padding=False,
                    truncation=False,
                )["input_ids"][
                    0
                ][2:]
            else:
                question = questions[(batch_index * batch_size) + prompt_index]
                input_tokens_question = tokenizer(
                    question,
                    return_tensors="np",
                    add_special_tokens=False,
                    padding=False,
                    truncation=False,
                )[
                    "input_ids"
                ][0]

            if print_logging:
                print("\n-----------------------------------------------------")
                print("-----------------------------------------------------")
                print(f"batch_prompts[prompt_index]: {batch_prompts[prompt_index]}")
                print(f"question: {question}")
                print(f"input_tokens_prompt.shape: {input_tokens_prompt.shape}")
                print(f"input_tokens_question.shape: {input_tokens_question.shape}")
                print(f"input_tokens_prompt: {input_tokens_prompt}")
                print(f"input_tokens_question: {input_tokens_question}")
                print("-----------------------------------------------------")
                print("-----------------------------------------------------")

            start, end = find_numpy_indices(input_tokens_question, input_tokens_prompt, print_logging)

            for layer_name in top_k_expert_indices:
                filtered_expert_choices_per_token[layer_name].append(top_k_expert_indices[layer_name][batch_index][prompt_index][start : end + 1])

    if print_logging:
        for layer_name in filtered_expert_choices_per_token:
            print("\n-----------------------------------------------------")
            print("-----------------------------------------------------")

            print(f"Number of prompts recorded in layer '{layer_name}': '{len(filtered_expert_choices_per_token[layer_name])}'")

            for prompt_index, prompt_expert_indices in enumerate(filtered_expert_choices_per_token[layer_name]):
                print(f"Layer: '{layer_name}', prompt_index: '{prompt_index}', shape: '{prompt_expert_indices.shape}'")

    data_utils.create_directory(f"{root_folder}/results/gate_output/{model_config.model_name}")

    data_utils.save_data(
        filtered_expert_choices_per_token,
        f"{root_folder}/results/gate_output/{model_config.model_name}/{model_config.model_name}_expert_indices.pkl",
    )

    print("\n------------------ Job Finished ------------------")
