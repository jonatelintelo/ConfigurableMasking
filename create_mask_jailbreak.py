import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import sys
import numpy as np
from tqdm import tqdm
import os
from datasets import load_dataset

# Modules defined in your project
import argument_parser as argument_parser
import moe_model_files.model_utils as model_utils
import moe_model_files.model_configurations as model_configurations
import data.data_utils as data_utils
import lstm.lstm_model as lstm_model


class SafetyEvaluator:
    def __init__(self, j_model, j_tokenizer):
        self.model = j_model
        self.tokenizer = j_tokenizer

    def evaluate(self, user_questions, generated_text):
        # Wrap the strings in lists so `zip()` processes them as single items
        judge_prompts = data_utils.construct_judge_prompt(questions=user_questions, responses=generated_text)

        safety_flags = []

        # Extract the single formatted chat from the returned list
        for judge_prompt in tqdm(judge_prompts):

            # Call the moderate function with the single chat prompt
            judge_output = model_utils.moderate(self.model, self.tokenizer, judge_prompt)
            safety_flags.append("unsafe" in judge_output.lower())

        return safety_flags


def find_token_range(question_ids, prompt_ids):
    """Finds the start and end indices of the question within the full prompt."""
    len_q = len(question_ids)
    len_p = len(prompt_ids)
    for i in range(len_p - len_q + 1):
        if np.array_equal(prompt_ids[i : i + len_q], question_ids):
            return i, i + len_q - 1
    return None, None


# ==========================================
# 2. Phase 1: Universal Circuit Discovery
# ==========================================
def discover_universal_steering_circuit(
    lstm_model,
    dataloader,
    num_layers,
    num_total_experts,
    target_class,
    l1_lambda,
    lr,
    epochs,
    threshold,
):
    device = next(lstm_model.parameters()).device

    S = nn.Parameter(torch.zeros(num_layers, num_total_experts, device=device))
    nn.init.kaiming_uniform_(S)
    optimizer = torch.optim.Adam([S], lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    lstm_model.train()
    for param in lstm_model.parameters():
        param.requires_grad = False

    print("\nStarting Universal Circuit Discovery...", flush=True)

    for epoch in tqdm(range(epochs), desc="Optimizing Circuit", unit="epoch"):
        total_loss, bce_loss_sum = 0, 0

        for batch_logits, batch_lengths in dataloader:
            batch_logits = batch_logits.to(device)
            batch_lengths = batch_lengths.cpu()
            target_tensor = torch.full((batch_logits.size(0), 1), target_class, dtype=torch.float32, device=device)

            optimizer.zero_grad()

            # --- ADAPTIVE LOGIT SCALING ---
            with torch.no_grad():
                # Calculate standard deviation per layer (across Batch, Tokens, and Experts)
                # Adding 1e-5 to prevent division by zero in the backward pass
                sigma_l = batch_logits.std(dim=(0, 1, 3), keepdim=True) + 1e-5

            # Adaptive Logit Scaling: Scale S by layer variance
            steered_logits = batch_logits + (S.unsqueeze(0).unsqueeze(0) * sigma_l)

            # Predict & Calculate Loss
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
    def __init__(self, model_name, layer_name, steering_vector, alpha):
        self.model_name = model_name
        self.layer_name = layer_name
        self.steering_vector = steering_vector
        self.alpha = alpha

    def __call__(self, module, inputs, output):
        if self.model_name == "deepseek-moe-16b-chat":
            logits = output[0]
        else:
            logits = output

        # Calculate dynamic logit standard deviation for this layer's current forward pass
        # This keeps inference steering perfectly scaled to the discovery phase
        sigma_l = logits.std() + 1e-5

        # Apply adaptive scaled intervention
        steered = logits + (self.alpha * sigma_l * self.steering_vector.to(logits.device))

        # Re-pack output based on model architecture
        if self.model_name == "deepseek-moe-16b-chat":
            if isinstance(output, tuple):
                return (steered,) + output[1:]
            elif isinstance(output, list):
                return [steered] + output[1:]
            return steered
        else:
            return steered


def apply_steering_hooks(model_name, model, gate_name, sparse_S, alpha):
    hook_handles = []
    layer_idx = 0

    for layer_name, module in model.named_modules():
        if layer_name.lower().endswith(gate_name):
            layer_steering_vector = sparse_S[layer_idx]
            if torch.any(layer_steering_vector != 0):
                hook_fn = ModelAwareSteeringHook(model_name, layer_name, layer_steering_vector, alpha)
                handle = module.register_forward_hook(hook_fn)
                hook_handles.append(handle)
                # print(f"Injected hook on: '{layer_name}' (Layer {layer_idx})")
            layer_idx += 1
    return hook_handles


# ==========================================
# 4. Main Execution Pipeline
# ==========================================
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
        if torch.cuda.is_available():
            print(f"First GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"Test tensor on GPU: {torch.rand(5).cuda().device}")

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

    model, tokenizer = model_utils.load_model(models[model_id])
    device = next(model.parameters()).device

    print("\nLoading trained LSTM model...")
    lstm_model_dir = os.path.join(root_folder, "results", "trained_lstm_models", model_config.model_name, f"{model_config.model_name}_jailbreak_lstm.pkl")
    checkpoint = torch.load(lstm_model_dir, map_location=device)
    NUM_TOTAL_EXPERTS, NUM_LAYERS = checkpoint["num_total_experts"], checkpoint["num_layers"]

    lstm = lstm_model.MoETraceClassifierLinear(NUM_TOTAL_EXPERTS, NUM_LAYERS)
    lstm.load_state_dict(checkpoint["model_state_dict"])
    lstm = lstm.to(device)

    conversation_histories, labels = data_utils.load_jailbreak_dataset(root_folder=root_folder, model_name=model_config.model_name, malicious_only=True)

    prompts = []
    final_user_questions = []

    print("Formatting prompts with chat templates...")
    for history in conversation_histories:
        # The target for our token range is the last message in the history
        final_user_questions.append(history[-1]["content"])

        # Format full context appropriately for the model
        chat = [m for m in history if m["role"] != "system"] if model_config.model_name == "deepseek-moe-16b-chat" else history

        if model_config.model_name == "Hunyuan-A13B-Instruct":
            prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        else:
            prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        prompts.append(prompt)

    # Pre-tokenize the final user turn for matching
    print("Pre-tokenizing target prompts for alignment...")

    tokenized_questions = []
    for q_text in tqdm(final_user_questions, desc="Tokenizing Final User Questions"):
        if model_config.model_name == "deepseek-moe-16b-chat":
            q_text = " " + q_text
        elif model_config.model_name == "Mixtral-8x7B-Instruct-v0.1":
            q_text = "\n" + q_text

        q_ids = tokenizer(q_text, add_special_tokens=False)["input_ids"]

        if model_config.model_name == "Mixtral-8x7B-Instruct-v0.1":
            q_ids = q_ids[2:]

        tokenized_questions.append(q_ids)

    all_logits, all_lengths = [], []
    current_batch_activations = {}

    def get_collection_hook(layer_index):
        def hook(module, inp, out):
            logits = out[0] if isinstance(out, (tuple, list)) else out
            current_batch_activations[layer_index] = logits.detach()  # Keep raw logits

        return hook

    collection_handles = []
    for i, name in enumerate([n for n, m in model.named_modules() if n.lower().endswith(model_config.gate_name.lower())]):
        module = dict(model.named_modules())[name]
        collection_handles.append(module.register_forward_hook(get_collection_hook(i)))

    print(f"Collecting baseline routing logits for {len(prompts)} prompts...")

    BATCH_SIZE = 16

    for b_idx, batch_prompts in enumerate(tqdm(data_utils.batchify(prompts, BATCH_SIZE), total=(len(prompts) + BATCH_SIZE - 1) // BATCH_SIZE)):
        current_batch_activations.clear()
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        b_size, s_len = inputs.input_ids.shape

        with torch.no_grad():
            model(**inputs)

        for l_idx in range(NUM_LAYERS):
            if current_batch_activations[l_idx].dim() == 2:
                current_batch_activations[l_idx] = current_batch_activations[l_idx].view(b_size, s_len, -1)

        input_ids_np = inputs.input_ids.cpu().numpy()

        for p_idx in range(b_size):
            global_idx = (b_idx * BATCH_SIZE) + p_idx
            if global_idx >= len(prompts):
                break

            start, end = find_token_range(tokenized_questions[global_idx], input_ids_np[p_idx])
            if start is None:
                continue

            prompt_trace = []
            for l_idx in range(NUM_LAYERS):
                # We collect RAW logits, slice them out, and save
                prompt_trace.append(current_batch_activations[l_idx][p_idx, start : end + 1, :])

            # Stack to [Tokens, Layers, Experts]
            stacked_trace = torch.stack(prompt_trace, dim=1)
            all_logits.append(stacked_trace.cpu())
            all_lengths.append(torch.tensor(stacked_trace.shape[0], dtype=torch.int64))

    for h in collection_handles:
        h.remove()

    # Create padded dataset using PyTorch's pad_sequence
    padded_logits = torch.nn.utils.rnn.pad_sequence(all_logits, batch_first=True, padding_value=0.0)
    dataset_lengths = torch.stack(all_lengths)

    trace_dataset = TensorDataset(padded_logits, dataset_lengths)
    trace_dataloader = DataLoader(trace_dataset, batch_size=BATCH_SIZE, shuffle=True)

    threshold = 0.1
    # lambdas = [0, 1e-5, 1e-4, 1e-3]
    # steering_alphas = [2.0, 1.5, 1.0, 0.6, 0.5, 0.4, 0.2]
    lambdas = [1e-5]
    steering_alphas = [1.5,1.2]


    print("\n--- Loading Baseline (Unsteered) Responses from File ---", flush=True)
    jailbreak_data_path = f"{root_folder}/data/jailbreak/jailbreak_contexts_{model_config.model_name}.jsonl"

    # Load the full list of original responses
    jailbreak_responses = list(load_dataset("json", data_files=jailbreak_data_path, split="train")["jailbreak_response"])

    # Slice the first 10 to match your final_user_questions[:10] loop later
    baseline_responses = jailbreak_responses[:BATCH_SIZE]

    eval_prompts = prompts[:BATCH_SIZE]

    print("\nLoading Judge Model (Safety Evaluator)...")
    judge_model_name = "meta-llama/Llama-Guard-3-8B"
    judge_model, judge_tokenizer = model_utils.load_model(judge_model_name)
    judge_model.eval()

    safety_evaluator = SafetyEvaluator(judge_model, judge_tokenizer)

    hook_handles, top_k_expert_indices = model_utils.register_activation_hooks(model_config.model_name, model, model_config.top_k, model_config.gate_name)

    for _lambda in lambdas:
        S_optimized = discover_universal_steering_circuit(
            lstm_model=lstm,
            dataloader=trace_dataloader,
            num_layers=NUM_LAYERS,
            num_total_experts=NUM_TOTAL_EXPERTS,
            target_class=0.0,
            l1_lambda=_lambda,
            lr=0.05,
            epochs=100,
            threshold=threshold,
        )

        S_abs = torch.abs(S_optimized).detach().cpu().numpy().flatten()

        print("\n" + "=" * 40)
        print("S TENSOR DISTRIBUTION ANALYSIS")
        print("=" * 40)
        print(f"Total Experts: {len(S_abs)}")
        print(f"Max absolute value: {S_abs.max():.4f}")
        print(f"Mean absolute value: {S_abs.mean():.6f}")

        print("\nPercentiles:")
        for p in [50, 75, 90, 95, 99, 99.9]:
            print(f"  {p}th percentile: {np.percentile(S_abs, p):.6f}")

        print("\nValue Bins (Absolute Magnitude):")
        bins = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 10.0]
        hist, bin_edges = np.histogram(S_abs, bins=bins)
        for i in range(len(hist)):
            print(f"  [{bin_edges[i]:.2f} to {bin_edges[i+1]:.2f}): {hist[i]:>5} experts")
        print("=" * 40 + "\n")

        S_sparse = torch.where(torch.abs(S_optimized) > threshold, S_optimized, torch.zeros_like(S_optimized))

        for steering_alpha in steering_alphas:
            # Apply Inference Steering
            print("\n--- Applying Steering Hooks to LLM ---", flush=True)
            active_steering_hooks = apply_steering_hooks(model_config.model_name, model, model_config.gate_name, S_sparse, steering_alpha)

            print("\nGenerating Responses with Steering Active...")
            eval_responses = model_utils.generate_output(model, model_config.model_name, tokenizer, eval_prompts, batch_size=BATCH_SIZE)

            for hook in active_steering_hooks:
                hook.remove()

            print("Evaluating Safety in batch...")
            clean_steered_responses = [r.strip() for r in eval_responses]
            safety_flags = safety_evaluator.evaluate(final_user_questions[BATCH_SIZE], clean_steered_responses)

            print(f"\n--- Model Responses for l1_lambda={_lambda}, threshold={threshold}, and alpha={steering_alpha}---")

            # Zip the safety flags into the loop alongside the other data
            for q, base_r, steered_r, is_unsafe in zip(final_user_questions[:BATCH_SIZE], baseline_responses, eval_responses, safety_flags):
                print(f"\nPrompt: {[q]}")
                print(f"Baseline (Before Masking): {[base_r.strip()]}")
                print(f"Steered  (After Masking) : {[steered_r.strip()]}")
                print(f"Safety Evaluation: {'Unsafe' if is_unsafe else 'Safe'}")

    print("\n------------------ Job Finished ------------------")
