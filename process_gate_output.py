import moe_model_files.model_configurations as model_configurations
import data.data_utils as data_utils
import argument_parser as argument_parser

import sys
import torch


def get_token_traces(activations_dict, print_logging):
    """
    Reshapes layer-wise MoE data into token-wise traces.

    Args:
        activations_dict: Dictionary where keys are layer names
                          and values are lists of tensors (one tensor per prompt).

    Returns:
        traces: dict where traces[prompt_index][token_index] = list of 32 layers' experts
                e.g., traces[0][5] might look like [[1, 4], [3, 7], ... ]
    """
    result_traces = {}

    # 1. Sort the layer keys numerically
    # This ensures we build the path in the correct order of depth.
    layer_keys = sorted(
        activations_dict.keys(),
        # Extracts '0' from 'model.layers.0...'
        key=lambda x: int(x.split(".")[2]),
    )

    if print_logging:
        print(f"layer_keys: {layer_keys}")

    # 2. Determine number of prompts by looking at Layer 0
    # We assume the list of tensors is the same length across all layers
    first_layer_data = activations_dict[layer_keys[0]]

    # if print_logging:
    #     print(f"first_layer_data: {first_layer_data}")

    num_prompts = len(first_layer_data)

    if print_logging:
        print(f"num_prompts: {num_prompts}")

    for prompt_idx in range(num_prompts):
        result_traces[prompt_idx] = {}

        # 3. Determine number of tokens for THIS specific prompt
        # We look at the tensor shape in the first layer for this prompt
        # e.g., torch.Size([18, 2]) -> num_tokens = 18
        tensor_shape = first_layer_data[prompt_idx].shape
        num_tokens = tensor_shape[0]

        if print_logging and prompt_idx < 2:
            print(f"num_tokens: {num_tokens}")

        # Initialize the list for every token in this prompt
        for token_idx in range(num_tokens):
            result_traces[prompt_idx][token_idx] = []

        # 4. Iterate through every layer to build the trace
        for layer_key in layer_keys:
            # Get the tensor for the current prompt in the current layer
            # Shape is [num_tokens, 2]
            layer_tensor = activations_dict[layer_key][prompt_idx]

            # Extract the experts for each token
            for token_idx in range(num_tokens):
                # .tolist() converts the tensor row [ExpA, ExpB] into a python list
                experts = layer_tensor[token_idx].tolist()
                result_traces[prompt_idx][token_idx].append(experts)

    return result_traces


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
        # LLMs
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

    print("\nLoading precomputed activation analysis...")

    expert_indices = data_utils.load_data(f"{root_folder}/results/gate_output/{model_config.model_name}/{model_config.model_name}_expert_indices.pkl")

    if print_logging:
        for layer_index, layer_name in enumerate(expert_indices):
            print(f"Number of prompts recorded in layer '{layer_name}': {len(expert_indices[layer_name])}")
            print(f"expert_indices[{layer_name}][0]: {expert_indices[layer_name][0]}")
            print(f"expert_indices[{layer_name}][-1]: {expert_indices[layer_name][-1]}")

            for prompt_index, prompt_expert_indices in enumerate(expert_indices[layer_name]):
                print(f"Layer: '{layer_name}', prompt_index: '{prompt_index}', shape: '{prompt_expert_indices.shape}'")

                if prompt_index == 2:
                    break

            if layer_index == 1:
                break

    traces = get_token_traces(expert_indices, print_logging)
    _, labels = data_utils.load_adult_set(root_folder, model_config.model_name, malicious_only=False)  # label 1 = refusal behavior

    if print_logging:
        print(f"traces[0]: {traces[0]}")
        print(f"labels[0]: {labels[0]}")
        print(f"traces[-1]: {traces[len(traces)-1]}")
        print(f"labels[-1]: {labels[len(labels)-1]}")

    data_utils.create_directory(f"{root_folder}/results/lstm_input/{model_config.model_name}")

    data_utils.save_data(
        traces,
        f"{root_folder}/results/lstm_input/{model_config.model_name}/{model_config.model_name}_traces.pkl",
    )
    data_utils.save_data(
        labels,
        f"{root_folder}/results/lstm_input/{model_config.model_name}/{model_config.model_name}_labels.pkl",
    )

    print("\n------------------ Job Finished ------------------")
