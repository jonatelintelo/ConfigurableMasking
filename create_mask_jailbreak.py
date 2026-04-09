import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from datasets import load_dataset

# Project Modules
import argument_parser as argument_parser
import moe_model_files.model_utils as model_utils
import moe_model_files.model_configurations as model_configurations
import data.data_utils as data_utils
import lstm.lstm_model as lstm_model


# ==========================================
# 1. Helper Classes & Alignment Tools
# ==========================================
class SafetyEvaluator:
    def __init__(self, judge_model, judge_tokenizer):
        self.model = judge_model
        self.tokenizer = judge_tokenizer

    def evaluate(self, user_questions, generated_text):
        judge_prompts = data_utils.construct_judge_prompt(questions=user_questions, responses=generated_text)
        safety_flags = []
        for judge_prompt in tqdm(judge_prompts, desc="Running Llama-Guard"):
            judge_output = model_utils.moderate(self.model, self.tokenizer, judge_prompt)
            safety_flags.append("unsafe" in judge_output.lower())
        return safety_flags


def find_token_range(question_ids, prompt_ids):
    """Finds the start and end indices of the target question within the full prompt."""
    len_q = len(question_ids)
    len_p = len(prompt_ids)
    for i in range(len_p - len_q + 1):
        if np.array_equal(prompt_ids[i : i + len_q], question_ids):
            return i, i + len_q - 1
    return None, None


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

            # Dynamic logit scaling based on batch standard deviation
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
    def __init__(self, model_name, layer_name, steering_vector, alpha):
        self.model_name = model_name
        self.layer_name = layer_name
        self.steering_vector = steering_vector
        self.alpha = alpha

    def __call__(self, module, inputs, output):
        # Unwrap tuple outputs for architectures like Mixtral/DeepSeek
        logits = output[0] if isinstance(output, (tuple, list)) else output
        sigma_l = logits.std() + 1e-5

        steered = logits + (self.alpha * sigma_l * self.steering_vector.to(logits.device))

        # Repack into original output format
        if isinstance(output, tuple):
            return (steered,) + output[1:]
        elif isinstance(output, list):
            return [steered] + output[1:]
        return steered


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
    print_logging = arguments.print_logging

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

    # --- Load Models ---
    model, tokenizer = model_utils.load_model(models[model_id])
    device = next(model.parameters()).device

    print("\nLoading trained LSTM model...")
    lstm_model_dir = os.path.join(root_folder, "results", "trained_lstm_models", model_config.model_name, f"{model_config.model_name}_jailbreak_lstm.pkl")
    checkpoint = torch.load(lstm_model_dir, map_location=device)
    NUM_TOTAL_EXPERTS, NUM_LAYERS = checkpoint["num_total_experts"], checkpoint["num_layers"]

    lstm = lstm_model.MoETraceClassifierLinear(NUM_TOTAL_EXPERTS, NUM_LAYERS)
    lstm.load_state_dict(checkpoint["model_state_dict"])
    lstm = lstm.to(device)

    # --- Load & Format Dataset ---
    conversation_histories, labels = data_utils.load_jailbreak_dataset(root_folder=root_folder, model_name=model_config.model_name, malicious_only=True)

    prompts = []
    final_user_questions = []

    print("Formatting prompts with chat templates...")
    for history in conversation_histories:
        final_user_questions.append(history[-1]["content"])
        chat = [m for m in history if m["role"] != "system"] if model_config.model_name == "deepseek-moe-16b-chat" else history

        if model_config.model_name == "Hunyuan-A13B-Instruct":
            prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        else:
            prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    print("Pre-tokenizing target prompts for alignment...")
    tokenized_questions = []
    for q_text in tqdm(final_user_questions, desc="Tokenizing"):
        if model_config.model_name == "deepseek-moe-16b-chat":
            q_text = " " + q_text
        elif model_config.model_name == "Mixtral-8x7B-Instruct-v0.1":
            q_text = "\n" + q_text

        q_ids = tokenizer(q_text, add_special_tokens=False)["input_ids"]
        if model_config.model_name == "Mixtral-8x7B-Instruct-v0.1":
            q_ids = q_ids[2:]
        tokenized_questions.append(q_ids)

    EVAL_SIZE = len(prompts)
    print(f"\nEvaluating dataset: {EVAL_SIZE} prompts.")

    eval_prompts = prompts[:EVAL_SIZE]
    eval_user_questions = final_user_questions[:EVAL_SIZE]
    eval_tokenized_questions = tokenized_questions[:EVAL_SIZE]

    # --- Collect Baseline Logits (For LSTM Circuit Discovery) ---
    all_logits, all_lengths = [], []
    current_batch_activations = {}

    def get_collection_hook(layer_index):
        def hook(module, inp, out):
            logits = out[0] if isinstance(out, (tuple, list)) else out
            current_batch_activations[layer_index] = logits.detach()

        return hook

    collection_handles = []
    layer_idx = 0
    for layer_name, module in model.named_modules():
        if layer_name.lower().endswith(model_config.gate_name.lower()):
            collection_handles.append(module.register_forward_hook(get_collection_hook(layer_idx)))
            layer_idx += 1

    print(f"Collecting baseline routing logits for {EVAL_SIZE} prompts...")
    BATCH_SIZE = 16

    for b_idx, batch_prompts in enumerate(tqdm(data_utils.batchify(eval_prompts, BATCH_SIZE), total=(len(eval_prompts) + BATCH_SIZE - 1) // BATCH_SIZE)):
        current_batch_activations.clear()
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        b_size, s_len = inputs.input_ids.shape

        with torch.no_grad():
            model(**inputs)

        # Reshape flat logits back to [Batch, Seq_Len, Experts] if necessary
        for l_idx in range(NUM_LAYERS):
            if current_batch_activations[l_idx].dim() == 2:
                current_batch_activations[l_idx] = current_batch_activations[l_idx].view(b_size, s_len, -1)

        input_ids_np = inputs.input_ids.cpu().numpy()

        for p_idx in range(b_size):
            global_idx = (b_idx * BATCH_SIZE) + p_idx
            if global_idx >= len(eval_prompts):
                break

            start, end = find_token_range(eval_tokenized_questions[global_idx], input_ids_np[p_idx])
            if start is None:
                continue

            prompt_trace = []
            for l_idx in range(NUM_LAYERS):
                prompt_trace.append(current_batch_activations[l_idx][p_idx, start : end + 1, :])

            stacked_trace = torch.stack(prompt_trace, dim=1)
            all_logits.append(stacked_trace.cpu())
            all_lengths.append(torch.tensor(stacked_trace.shape[0], dtype=torch.int64))

    for h in collection_handles:
        h.remove()

    padded_logits = torch.nn.utils.rnn.pad_sequence(all_logits, batch_first=True, padding_value=0.0)
    dataset_lengths = torch.stack(all_lengths)
    trace_dataset = TensorDataset(padded_logits, dataset_lengths)
    trace_dataloader = DataLoader(trace_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- Load Evaluator & Baselines ---
    print("\n--- Loading Baseline Responses & Judge Model ---")
    jailbreak_data_path = f"{root_folder}/data/jailbreak/jailbreak_contexts_{model_config.model_name}.jsonl"
    jailbreak_responses = list(load_dataset("json", data_files=jailbreak_data_path, split="train")["jailbreak_response"])
    baseline_responses = jailbreak_responses[:EVAL_SIZE]

    judge_model_name = "meta-llama/Llama-Guard-3-8B"
    judge_model, judge_tokenizer = model_utils.load_model(judge_model_name)
    judge_model.eval()
    safety_evaluator = SafetyEvaluator(judge_model, judge_tokenizer)

    # --- Experiment Tracks ---
    configs_to_test = [
        {"lambda": 1e-05, "alphas": [1.2, 1.5]},  # Track 1: Strict Refusal
        {"lambda": 0.001, "alphas": [0.6, 0.8]},  # Track 2: Nuanced Reframing
        {"lambda": 0.0001, "alphas": [0.8, 1.0]},  # Track 3: The Goldilocks Zone
    ]

    threshold = 0.1
    TOP_K_EXPERTS = model_config.top_k
    current_batch_topk = {}

    def get_topk_hook(layer_idx, k):
        def hook(module, input, output):
            logits = output[0] if isinstance(output, (tuple, list)) else output
            current_batch_topk[layer_idx] = torch.topk(logits, k=k, dim=-1, sorted=False).indices.detach().cpu().to(torch.int16)

        return hook

    for config in configs_to_test:
        _lambda = config["lambda"]

        # Phase 1: Discover Circuit for this Lambda
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
        S_sparse = torch.where(torch.abs(S_optimized) > threshold, S_optimized, torch.zeros_like(S_optimized))

        # Phase 2: Test specific Alphas
        for steering_alpha in config["alphas"]:
            experiment_id = f"lambda_{_lambda}_alpha_{steering_alpha}"
            print(f"\n" + "=" * 60)
            print(f"RUNNING EXPERIMENT: {experiment_id}")
            print("=" * 60)

            active_steering_hooks = apply_steering_hooks(model_config.model_name, model, model_config.gate_name, S_sparse, steering_alpha)

            # Register Top-K Hooks
            active_topk_hooks = []
            layer_idx = 0
            for layer_name, module in model.named_modules():
                if layer_name.lower().endswith(model_config.gate_name.lower()):
                    active_topk_hooks.append(module.register_forward_hook(get_topk_hook(layer_idx, TOP_K_EXPERTS)))
                    layer_idx += 1

            print("Extracting steered Top-K choices for User Question tokens...")
            final_topk_traces = []

            for b_idx, batch_prompts in enumerate(data_utils.batchify(eval_prompts, BATCH_SIZE)):
                current_batch_topk.clear()
                inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
                b_size, s_len = inputs.input_ids.shape

                # Targeted forward pass (triggers Top-K hooks with steering active)
                with torch.no_grad():
                    model(**inputs)

                input_ids_np = inputs.input_ids.cpu().numpy()

                for p_idx in range(b_size):
                    global_idx = (b_idx * BATCH_SIZE) + p_idx
                    if global_idx >= len(eval_prompts):
                        break

                    start, end = find_token_range(eval_tokenized_questions[global_idx], input_ids_np[p_idx])

                    if start is None:
                        # Prevent desync if token matching fails
                        final_topk_traces.append(np.array([]))
                        continue

                    prompt_topk_trace = []
                    for l_idx in range(NUM_LAYERS):
                        if current_batch_topk[l_idx].dim() == 2:
                            current_batch_topk[l_idx] = current_batch_topk[l_idx].view(b_size, s_len, -1)

                        token_choices = current_batch_topk[l_idx][p_idx, start : end + 1, :]
                        prompt_topk_trace.append(token_choices)

                    stacked_trace = torch.stack(prompt_topk_trace, dim=1).numpy()
                    final_topk_traces.append(stacked_trace)

            # Cleanup Top-K hooks before generating text
            for hook in active_topk_hooks:
                hook.remove()

            # Save Top-K traces
            save_dir = os.path.join(root_folder, "results", "topk_routings", model_config.model_name)
            os.makedirs(save_dir, exist_ok=True)
            file_name = f"topk_routing_{experiment_id}.pt"

            torch.save(
                {
                    "traces": final_topk_traces,
                    "hyperparameters": {"lambda": _lambda, "alpha": steering_alpha, "k": TOP_K_EXPERTS},
                },
                os.path.join(save_dir, file_name),
            )

            print(f"Saved Top-K traces for {len(final_topk_traces)} prompts to {file_name}")

            # Generate Responses & Evaluate
            print("Generating Responses with Steering Active...")
            eval_responses = model_utils.generate_output(model, model_config.model_name, tokenizer, eval_prompts, batch_size=BATCH_SIZE)

            for hook in active_steering_hooks:
                hook.remove()

            print("Evaluating Safety...")
            clean_steered_responses = [r.strip() for r in eval_responses]
            safety_flags = safety_evaluator.evaluate(eval_user_questions, clean_steered_responses)

            safe_count = safety_flags.count(False)
            print(f"\n---> Metric for {experiment_id}: {safe_count}/{EVAL_SIZE} evaluated as SAFE.")

    print("\n------------------ All Experiments Finished ------------------")
