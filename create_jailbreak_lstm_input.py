import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import json
import inspect
from tqdm import tqdm

# Project imports
import moe_model_files.model_configurations as model_configurations
import moe_model_files.model_utils as model_utils
import data.data_utils as data_utils
import argument_parser as argument_parser


def find_token_range(question_ids, prompt_ids, print_logging):
    """Finds the start and end indices of the question within the full prompt."""
    len_q = len(question_ids)
    len_p = len(prompt_ids)
    for i in range(len_p - len_q + 1):
        if np.array_equal(prompt_ids[i : i + len_q], question_ids):
            if print_logging:
                print(f"prompt_ids: {prompt_ids}")
                print(f"question_ids: {question_ids}")
                print(f"start: {i}")
                print(f"end: {i + len_q - 1}")
            return i, i + len_q - 1
    return None, None


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
    ]

    model_config = model_configurations.models[models[model_id]]
    print(f"\nInitializing: {model_config.model_name}")

    model, tokenizer = model_utils.load_model(models[model_id])

    # Load Balanced Data
    conversations, labels = data_utils.load_jailbreak_dataset(root_folder=root_folder, model_name=model_config.model_name, malicious_only=False)

    prompts = []
    final_user_texts = []

    print("Formatting prompts with chat templates...")
    base_model_name = model_config.model_name.split("/")[-1]

    for history in conversations:
        # The target for our token range is the last message in the history
        final_user_texts.append(history[-1]["content"])

        # Format full context appropriately for the model
        chat = [m for m in history if m["role"] != "system"] if base_model_name == "deepseek-moe-16b-chat" else history

        if base_model_name == "Hunyuan-A13B-Instruct":
            prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        else:
            prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        prompts.append(prompt)

    # Pre-tokenize the final user turn for matching
    print("Pre-tokenizing target prompts for alignment...")

    tokenized_questions = []
    for q_text in tqdm(final_user_texts, desc="Tokenizing Final Prompts"):
        if base_model_name == "deepseek-moe-16b-chat":
            q_text = " " + q_text
        elif base_model_name == "Mixtral-8x7B-Instruct-v0.1":
            q_text = "\n" + q_text

        q_ids = tokenizer(q_text, add_special_tokens=False)["input_ids"]

        if base_model_name == "Mixtral-8x7B-Instruct-v0.1":
            q_ids = q_ids[2:]

        tokenized_questions.append(q_ids)

    # Setup PyTorch Hooks
    current_batch_activations = {}

    def get_activation_hook(layer_idx):
        def hook(module, input, output):
            logits = output[0] if isinstance(output, (tuple, list)) else output
            probs = F.softmax(logits.detach(), dim=-1).to(torch.float16)
            current_batch_activations[layer_idx] = probs

        return hook

    handles = []
    layer_names = [n for n, m in model.named_modules() if n.lower().endswith(model_config.gate_name.lower())]
    for i, name in enumerate(layer_names):
        module = dict(model.named_modules())[name]
        handles.append(module.register_forward_hook(get_activation_hook(i)))

    final_traces = []
    final_labels = []
    failed_matches = 0

    batch_size = 16  # Should be a bit lower to accommodate longer multi-turn contexts
    total_batches = (len(prompts) + batch_size - 1) // batch_size

    # Batched Forward Pass & Logit Extraction
    print(f"\nStarting Trace Collection (Softmax Probabilities)...")

    for b_idx, batch_prompts in enumerate(tqdm(data_utils.batchify(prompts, batch_size), total=total_batches)):
        current_batch_activations.clear()

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

        if "token_type_ids" in inputs:
            forward_args = inspect.signature(model.forward).parameters
            if "token_type_ids" not in forward_args:
                inputs.pop("token_type_ids")

        b_size, s_len = inputs.input_ids.shape

        with torch.no_grad():
            model(**inputs)

        for l_idx in range(len(layer_names)):
            if current_batch_activations[l_idx].dim() == 2:
                current_batch_activations[l_idx] = current_batch_activations[l_idx].view(b_size, s_len, -1)

        input_ids_np = inputs.input_ids.cpu().numpy()

        for p_idx in range(b_size):
            global_p_idx = (b_idx * batch_size) + p_idx
            if global_p_idx >= len(tokenized_questions):
                break

            q_ids = tokenized_questions[global_p_idx]
            start, end = find_token_range(q_ids, input_ids_np[p_idx], print_logging if b_idx == 0 and p_idx == 0 else False)

            if start is None:
                failed_matches += 1
                continue

            prompt_trace = []

            for l_idx in range(len(layer_names)):
                token_probs = current_batch_activations[l_idx][p_idx, start : end + 1, :].cpu().numpy()
                prompt_trace.append(token_probs)

            stacked_trace = np.stack(prompt_trace, axis=1)

            final_traces.append(stacked_trace)
            final_labels.append(labels[global_p_idx])

    for h in handles:
        h.remove()

    if failed_matches > 0:
        print(f"\nWarning: Could not find exact token match for {failed_matches} prompts. Check tokenizer spacing logic.")

    # Save Results
    save_path = f"{root_folder}/results/lstm_input/{base_model_name}"
    os.makedirs(save_path, exist_ok=True)

    print("\nSaving traces to disk...")
    data_utils.save_data(final_traces, f"{save_path}/{base_model_name}_multi_turn_traces.pkl")
    data_utils.save_data(final_labels, f"{save_path}/{base_model_name}_multi_turn_labels.pkl")

    if print_logging and len(final_traces) > 0:
        print(f"\nNumber of traces: {len(final_traces)}")
        print(f"Number of labels: {len(final_labels)}")
        print(f"Shape of a trace: {final_traces[0].shape}")

    print(f"\nSaved {len(final_traces)} traces to {save_path}")
    print("\n------------------ Job Finished ------------------")
