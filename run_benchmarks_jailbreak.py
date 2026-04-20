import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from datasets import load_dataset
import gc
import lm_eval
from lm_eval.models.huggingface import HFLM

# Project Modules
import argument_parser as argument_parser
import moe_model_files.model_utils as model_utils
import moe_model_files.model_configurations as model_configurations
import data.data_utils as data_utils
import lstm.lstm_model as lstm_model


def run_benchmarks(model, tokenizer, tasks, limit, batch_size):
    """
    Runs standard capability benchmarks on the model in its current state (steered or unsteered).
    limit: Set to an integer (e.g., 200) for fast testing, or None for the full benchmark.
    """
    print(f"\n[+] Running Benchmarks: {tasks} (Limit: {limit if limit else 'ALL'})")
    
    # Wrap the already-loaded model. 
    # The wrapper will respect any active PyTorch hooks.
    eval_model = HFLM(
        pretrained=model, 
        tokenizer=tokenizer, 
        batch_size=batch_size,
        max_gen_toks=256
    )
    
    # Run the evaluation
    results = lm_eval.simple_evaluate(
        model=eval_model,
        tasks=tasks,
        num_fewshot=5, # Using 0-shot for speed, change to 5 for standard MMLU or 8 for GSM8K
        limit=limit,
        batch_size=batch_size
    )
    
    # Extract and print results robustly (handles different lm-eval versions)
    metrics = {}
    for task_name, task_results in results['results'].items():
        # Look for standard accuracy or exact match keys
        acc = task_results.get('acc,none', task_results.get('exact_match,strict-match', task_results.get('acc')))
        metrics[task_name] = acc
        print(f" -> {task_name.upper()} Score: {acc:.4f}")
        
    return metrics

# ==========================================
# 1. Helper Classes & Alignment Tools
# ==========================================
class SafetyEvaluator:
    def __init__(self, judge_model, judge_tokenizer):
        self.model = judge_model
        self.tokenizer = judge_tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def evaluate(self, conversation_histories, generated_text, batch_size=16):
        judge_prompts = data_utils.construct_judge_prompt_histories(histories=conversation_histories, responses=generated_text)
        safety_flags = []
        total_batches = (len(judge_prompts) + batch_size - 1) // batch_size

        for batch_prompts in tqdm(data_utils.batchify(judge_prompts, batch_size), total=total_batches, desc="Llama-Guard Batches"):
            batch_outputs = model_utils.batched_moderate(self.model, self.tokenizer, batch_prompts)

            for output in batch_outputs:
                safety_flags.append("unsafe" in output.lower())

        return safety_flags


def find_token_range_by_offsets(prompt_text, question_text, offsets):
    """Finds token start and end using character offset mappings."""
    char_start = prompt_text.rfind(question_text.strip())
    if char_start == -1:
        return None, None

    char_end = char_start + len(question_text.strip())
    start_idx, end_idx = None, None

    for i, (tok_start, tok_end) in enumerate(offsets):
        if tok_start == tok_end:
            continue
        if start_idx is None and tok_end > char_start:
            start_idx = i
        if tok_start < char_end:
            end_idx = i

    return start_idx, end_idx


def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# ==========================================
# 2. Phase 1: Universal Circuit Discovery
# ==========================================
def discover_universal_steering_circuit(lstm_model, dataloader, num_layers, num_total_experts, target_class, l1_lambda, lr, epochs, threshold):
    device = next(lstm_model.parameters()).device
    S = nn.Parameter(torch.zeros(num_layers, num_total_experts, device=device))
    nn.init.kaiming_uniform_(S)

    optimizer = torch.optim.Adam([S], lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    lstm_model.train()

    for param in lstm_model.parameters():
        param.requires_grad = False

    print(f"\nStarting Circuit Discovery (Lambda: {l1_lambda})...", flush=True)

    for epoch in tqdm(range(epochs), desc="Optimizing Circuit", unit="epoch"):
        total_loss, bce_loss_sum = 0, 0

        for batch_logits, batch_lengths in dataloader:
            batch_logits = batch_logits.to(device)
            batch_lengths = batch_lengths.cpu()
            target_tensor = torch.full((batch_logits.size(0), 1), target_class, dtype=torch.float32, device=device)

            optimizer.zero_grad()

            with torch.no_grad():
                sigma_l = batch_logits.std(dim=(0, 1, 3), keepdim=True) + 1e-5

            steered_logits = batch_logits + (S.unsqueeze(0).unsqueeze(0) * sigma_l)
            lstm_preds = lstm_model(steered_logits, batch_lengths)

            bce_loss = criterion(lstm_preds, target_tensor)
            l1_loss = l1_lambda * torch.norm(S, p=1)

            loss = bce_loss + l1_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            bce_loss_sum += bce_loss.item()

        avg_loss = total_loss / len(dataloader)
        avg_bce = bce_loss_sum / len(dataloader)
        active_experts = (torch.abs(S) > threshold).sum().item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            tqdm.write(f"Epoch {epoch+1:02d}/{epochs} | Avg Loss: {avg_loss:.4f} (BCE: {avg_bce:.4f}) | Nodes: {active_experts}")

    return S.detach()


# ==========================================
# 3. Phase 2: Inference Steering Hooks
# ==========================================
class ModelAwareSteeringHook:
    def __init__(self, model_name, layer_name, steering_vector, alpha, top_k=6):
        self.model_name = model_name
        self.layer_name = layer_name
        self.steering_vector = steering_vector
        self.alpha = alpha
        self.top_k = top_k  # DeepSeek 16b typically routes to 6 experts

    def __call__(self, module, inputs, output):
        # Extract raw logits. In DeepSeek, the gate output is typically a tuple:
        # e.g., (topk_indices, topk_weights, raw_logits)
        if isinstance(output, (tuple, list)) and len(output) >= 3:
            logits = output[2]
        else:
            logits = output

        if self.steering_vector.device != logits.device:
            self.steering_vector = self.steering_vector.to(logits.device)

        # 1. Apply steering to the raw logits
        sigma_l = logits.std() + 1e-5
        steered_logits = logits + (self.alpha * sigma_l * self.steering_vector)

        # 2. If it's a MoE gate, recompute the top-K routing!
        if isinstance(output, (tuple, list)) and len(output) >= 3:
            # Calculate new probabilities from the steered logits
            scores = steered_logits.softmax(dim=-1)

            # Recompute the winning experts and their weights
            topk_weights, topk_indices = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

            # 3. Robustly reconstruct the tuple.
            # We check the dtype of original output[0] to match the expected order.
            # If the original model returns (indices, weights, logits), output[0] is an integer.
            is_output0_indices = output[0].dtype in (torch.int8, torch.int16, torch.int32, torch.int64)

            if is_output0_indices:
                new_output = [topk_indices, topk_weights, steered_logits]
            else:
                new_output = [topk_weights, topk_indices, steered_logits]

            # Preserve any extra items in the tuple if they exist
            if len(output) > 3:
                new_output.extend(list(output[3:]))

            return tuple(new_output) if isinstance(output, tuple) else new_output

        return steered_logits


def apply_steering_hooks(model_name, model, gate_name, sparse_S, alpha):
    hook_handles = []
    layer_idx = 0

    for layer_name, module in model.named_modules():
        if layer_name.lower().endswith(gate_name.lower()):
            layer_steering_vector = sparse_S[layer_idx]
            if torch.any(layer_steering_vector != 0):
                hook_fn = ModelAwareSteeringHook(model_name, layer_name, layer_steering_vector, alpha)
                hook_handles.append(module.register_forward_hook(hook_fn))
            layer_idx += 1

    return hook_handles


# ==========================================
# 4. Main Execution Pipeline
# ==========================================
if __name__ == "__main__":
    arguments = argument_parser.parse_arguments()
    root_folder = arguments.root
    model_id = arguments.model_id

    models = [
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "microsoft/Phi-3.5-MoE-instruct",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "openai/gpt-oss-20b",
        "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        "tencent/Hunyuan-A13B-Instruct",
        "deepseek-ai/deepseek-moe-16b-chat",
    ]

    model_config = model_configurations.models[models[model_id]]
    print(f"\nInitializing: {model_config.model_name}")

    # --- Load Primary Model ---
    model, tokenizer = model_utils.load_model(models[model_id])
    device = next(model.parameters()).device

    print("\n--- BASELINE EVALUATION ---")
    # =["mmlu", "gsm8k"]
    baseline_metrics = run_benchmarks(model, tokenizer, tasks=["gsm8k"], limit=None, batch_size=8)

    del model, tokenizer
    flush()


    # # --- Load Dataset ---
    # conversation_histories, labels = data_utils.load_jailbreak_dataset(root_folder=root_folder, model_name=model_config.model_name, malicious_only=True)
    # prompts, final_user_questions, eval_histories = [], [], []

    # for history in conversation_histories:
    #     final_user_questions.append(history[-1]["content"])
    #     chat = [m for m in history if m["role"] != "system"] if model_config.model_name == "deepseek-moe-16b-chat" else history
    #     eval_histories.append([m for m in history if m["role"] != "system"])

    #     prompt = tokenizer.apply_chat_template(
    #         chat, tokenize=False, add_generation_prompt=True, enable_thinking=False if model_config.model_name == "Hunyuan-A13B-Instruct" else None
    #     )
    #     prompts.append(prompt)

    # EVAL_SIZE = len(prompts)
    # eval_prompts = prompts[:EVAL_SIZE]
    # eval_user_questions = final_user_questions[:EVAL_SIZE]
    # eval_chat_histories = eval_histories[:EVAL_SIZE]

    # # --- Step 1: Collect Baseline Logits ---
    # all_logits, all_lengths = [], []

    # current_batch_activations = {}

    # def get_activation_hook(layer_idx):
    #     def hook(module, input, output):
    #         logits = output[2] if isinstance(output, (tuple, list)) else output
    #         current_batch_activations[layer_idx] = logits.detach().to(torch.float16).cpu()

    #     return hook

    # collection_handles = []
    # layer_idx = 0

    # for layer_name, module in model.named_modules():
    #     if layer_name.lower().endswith(model_config.gate_name.lower()):
    #         collection_handles.append(module.register_forward_hook(get_activation_hook(layer_idx)))
    #         layer_idx += 1

    # BATCH_SIZE = 8

    # for b_idx, batch_prompts in enumerate(
    #     tqdm(data_utils.batchify(prompts, BATCH_SIZE), total=(len(prompts) + BATCH_SIZE - 1) // BATCH_SIZE, desc="Collect Baseline Logits")
    # ):
    #     current_batch_activations.clear()
    #     inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True)
    #     offset_mappings = inputs.pop("offset_mapping").cpu().numpy()
    #     inputs = inputs.to(device)

    #     with torch.inference_mode():
    #         model(**inputs)

    #     b_size, s_len = inputs.input_ids.shape

    #     for l_idx in range(layer_idx):
    #         if current_batch_activations[l_idx].dim() == 2:
    #             current_batch_activations[l_idx] = current_batch_activations[l_idx].view(b_size, s_len, -1)

    #     for p_idx in range(b_size):
    #         global_idx = (b_idx * BATCH_SIZE) + p_idx

    #         if global_idx >= len(prompts):
    #             break

    #         start, end = find_token_range_by_offsets(batch_prompts[p_idx], final_user_questions[global_idx], offset_mappings[p_idx])

    #         if start is None:
    #             raise Exception(f"Could not find token range for prompt: {batch_prompts[p_idx]} and question: {final_user_questions[global_idx]}")

    #         prompt_trace = [current_batch_activations[l_idx][p_idx, start : end + 1, :] for l_idx in range(layer_idx)]
    #         stacked_trace = torch.stack(prompt_trace, dim=1)
    #         all_logits.append(stacked_trace.cpu())
    #         all_lengths.append(torch.tensor(stacked_trace.shape[0], dtype=torch.int64))

    #     flush()

    # for h in collection_handles:
    #     h.remove()

    # trace_dataset = TensorDataset(torch.nn.utils.rnn.pad_sequence(all_logits, batch_first=True, padding_value=0.0), torch.stack(all_lengths))
    # trace_dataloader = DataLoader(trace_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # if model_config.model_name == "Qwen3-30B-A3B-Instruct-2507":
    #     configs_to_test = [
    #         {"lambda": 1e-5, "alphas": [1.75]},
    #     ]
    # elif model_config.model_name == "Phi-3.5-MoE-instruct":
    #     configs_to_test = [
    #         {"lambda": 1e-5, "alphas": [1.0]},
    #     ]
    # elif model_config.model_name == "Mixtral-8x7B-Instruct-v0.1":
    #     configs_to_test = [
    #         {"lambda": 1e-5, "alphas": [1.0]},
    #     ]
    # # elif model_config.model_name == "gpt-oss-20b":
    # #     configs_to_test = [
    # #         {"lambda": 1e-5, "alphas": [1.75]},
    # #     ]
    # elif model_config.model_name == "Qwen1.5-MoE-A2.7B-Chat":
    #     configs_to_test = [
    #         {"lambda": 1e-4, "alphas": [1.75]},
    #     ]
    # # elif model_config.model_name == "Hunyuan-A13B-Instruct":
    # #     configs_to_test = [
    # #         {"lambda": 1e-5, "alphas": [1.75]},
    # #     ]
    # elif model_config.model_name == "deepseek-moe-16b-chat":
    #     configs_to_test = [
    #         {"lambda": 1e-4, "alphas": [0.75]},
    #     ]


    # for config in configs_to_test:
    #     _lambda = config["lambda"]

    #     # Load LSTM only for discovery
    #     checkpoint = torch.load(
    #         os.path.join(root_folder, "lstm", "trained_lstm_models", model_config.model_name, f"{model_config.model_name}_jailbreak_lstm.pkl"),
    #         map_location=device,
    #     )
    #     NUM_TOTAL_EXPERTS, NUM_LAYERS = checkpoint["num_total_experts"], checkpoint["num_layers"]
    #     lstm = lstm_model.MoETraceClassifierLinear(NUM_TOTAL_EXPERTS, NUM_LAYERS).to(device)
    #     lstm.load_state_dict(checkpoint["model_state_dict"])

    #     S_optimized = discover_universal_steering_circuit(lstm, trace_dataloader, NUM_LAYERS, NUM_TOTAL_EXPERTS, 0.0, _lambda, 0.05, 100, 0.1)

    #     S_abs = torch.abs(S_optimized).detach().cpu().numpy().flatten()
    #     print("\n" + "=" * 40)
    #     print(f"S TENSOR DISTRIBUTION ANALYSIS (Lambda: {_lambda})")
    #     print("=" * 40)
    #     print(f"Total Experts: {len(S_abs)}")
    #     print(f"Max absolute value: {S_abs.max():.4f}")
    #     print(f"Mean absolute value: {S_abs.mean():.6f}")
    #     print("\nPercentiles:")
    #     for p in [50, 75, 90, 95, 99, 99.9]:
    #         print(f"  {p}th percentile: {np.percentile(S_abs, p):.6f}")
    #     print("\nValue Bins (Absolute Magnitude):")
    #     bins = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 10.0]
    #     hist, bin_edges = np.histogram(S_abs, bins=bins)
    #     for i in range(len(hist)):
    #         print(f"  [{bin_edges[i]:.2f} to {bin_edges[i+1]:.2f}): {hist[i]:>5} experts")
    #     print("=" * 40 + "\n")

    #     S_sparse = torch.where(torch.abs(S_optimized) > 0.1, S_optimized, torch.zeros_like(S_optimized))

    #     # Delete LSTM immediately after discovery to free VRAM
    #     del lstm, checkpoint
    #     flush()

    #     for steering_alpha in config["alphas"]:
    #         experiment_id = f"lambda_{_lambda}_alpha_{steering_alpha}"
    #         print(f"\nRUNNING GENERATION: {experiment_id}")

    #         active_steering_hooks = apply_steering_hooks(model_config.model_name, model, model_config.gate_name, S_sparse, steering_alpha)
        
    #         print(f"\n--- STEERED EVALUATION ({experiment_id}) ---")
    #         steered_metrics = run_benchmarks(model, tokenizer, tasks=["mmlu", "gsm8k"])

    #         for hook in active_steering_hooks:
    #             hook.remove()

    print("\n------------------ All Experiments Finished ------------------")
