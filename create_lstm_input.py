import torch
import torch.nn.functional as F
import numpy as np
import sys
import inspect
from tqdm import tqdm
from collections import defaultdict

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

    raise ValueError(f"No valid ")


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

    # Load Model Configuration
    models = [
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "microsoft/Phi-3.5-MoE-instruct",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "openai/gpt-oss-20b",
        "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        "tencent/Hunyuan-A13B-Instruct",
        "deepseek-ai/deepseek-moe-16b-chat",
        "IntervitensInc/pangu-pro-moe-model",
    ]
    model_config = model_configurations.models[models[model_id]]
    print(f"\nInitializing: {model_config.model_name}")

    # Load Model & Tokenizer
    model, tokenizer = model_utils.load_model(models[model_id]) # Function laod_model already puts model on device and in .eval() mode

    # Load Dataset
    questions, labels = data_utils.load_adult_set(root_folder, model_config.model_name, malicious_only=False)
    prompts = data_utils.construct_prompt(tokenizer, questions, model_config.model_name)

    # Dictionary to temporarily store batch activations
    # We use a hook to populate this
    current_batch_activations = {}

    def get_activation_hook(layer_idx):
        def hook(module, inp, out):
            # Handle Auxiliary outputs (DeepSeek style)
            logits = out[0] if isinstance(out, (tuple, list)) else out

            # Convert to Softmax Probabilities (High Fidelity)
            # We detach and move to CPU/float16 immediately to save VRAM
            probs = F.softmax(logits.detach(), dim=-1).to("cpu", dtype=torch.float16)

            current_batch_activations[layer_idx] = probs

        return hook

    # Register Hooks
    handles = []
    layer_names = [n for n, m in model.named_modules() if n.lower().endswith(model_config.gate_name.lower())]
    for i, name in enumerate(layer_names):
        module = dict(model.named_modules())[name]
        handles.append(module.register_forward_hook(get_activation_hook(i)))

    # Final Data Storage
    final_traces = []
    final_labels = []

    batch_size = 32  # Adjust based on your GPU VRAM
    total_batches = (len(prompts) + batch_size - 1) // batch_size

    print(f"\nStarting Trace Collection (Softmax Probabilities)...")

    for b_idx, batch_prompts in enumerate(tqdm(data_utils.batchify(prompts, batch_size), total=total_batches)):
        # Reset batch storage
        current_batch_activations.clear()

        # Tokenize
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        if "token_type_ids" in inputs:
            forward_args = inspect.signature(model.forward).parameters
            if "token_type_ids" not in forward_args:
                inputs.pop("token_type_ids")

        b_size, s_len = inputs.input_ids.shape

        # Forward Pass
        with torch.no_grad():
            model(**inputs)

        # Process each prompt in the batch
        input_ids_np = inputs.input_ids.cpu().numpy()

        for p_idx in range(b_size):
            # Find the question within the prompt
            q_text = questions[(b_idx * batch_size) + p_idx]
            q_ids = tokenizer(q_text, add_special_tokens=False)["input_ids"]

            start, end = find_token_range(q_ids, input_ids_np[p_idx], print_logging)

            if start is None:
                continue  # Skip if alignment fails

            # Construct the Trace: (Tokens, Layers, Experts)
            # We want to stack all layers for these specific tokens
            prompt_trace = []

            for l_idx in range(len(layer_names)):
                layer_probs = current_batch_activations[l_idx]

                # Unflatten if necessary
                if layer_probs.dim() == 2:
                    layer_probs = layer_probs.view(b_size, s_len, -1)

                # Slice the specific tokens for this prompt [Tokens, Experts]
                token_probs = layer_probs[p_idx, start : end + 1, :]
                prompt_trace.append(token_probs)

            # Stack into [Tokens, Layers, Experts]
            # Convert to numpy to keep final storage light
            stacked_trace = np.stack(prompt_trace, axis=1)

            final_traces.append(stacked_trace)
            final_labels.append(labels[(b_idx * batch_size) + p_idx])

    # Cleanup
    for h in handles:
        h.remove()

    # Save
    save_path = f"{root_folder}/results/lstm_input/{model_config.model_name}"
    data_utils.create_directory(save_path)

    data_utils.save_data(final_traces, f"{save_path}/{model_config.model_name}_traces.pkl")
    data_utils.save_data(final_labels, f"{save_path}/{model_config.model_name}_labels.pkl")

    if print_logging:
        print(f"final_traces[0]: {final_traces[0]}")
        print(f"final_labels[0]: {final_labels[0]}")
        print(f"final_labels[-1]: {final_labels[len(final_labels)-1]}")
        print(f"final_labels[-1]: {final_labels[len(final_labels)-1]}")

    print(f"\nJob Finished. Saved {len(final_traces)} traces to {save_path}")
