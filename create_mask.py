import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset
import sys
import numpy as np
from tqdm import tqdm
import math

# Modules defined in your project
import argument_parser as argument_parser
import moe_model_files.model_utils as model_utils
import moe_model_files.model_configurations as model_configurations
import data.data_utils as data_utils


def find_token_range(question_ids, prompt_ids):
    """Finds the start and end indices of the question within the full prompt."""
    len_q = len(question_ids)
    len_p = len(prompt_ids)
    for i in range(len_p - len_q + 1):
        if np.array_equal(prompt_ids[i : i + len_q], question_ids):
            return i, i + len_q - 1
    return None, None


# ==========================================
# 1. Your Trained LSTM Model
# ==========================================
class MoETraceClassifierLinear(nn.Module):
    def __init__(self, num_total_experts, num_layers, embed_dim=16, hidden_dim=64):
        super().__init__()
        self.num_total_experts = num_total_experts
        self.expert_projection = nn.Linear(num_total_experts, embed_dim, bias=False)
        self.lstm_input_size = num_layers * embed_dim
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x_masked, lengths):
        batch_size, max_seq_len, _, _ = x_masked.shape
        x_emb = self.expert_projection(x_masked)
        x_flat = x_emb.view(batch_size, max_seq_len, -1)

        # CPU lengths required by PyTorch
        lengths_cpu = lengths.cpu()
        packed_input = pack_padded_sequence(x_flat, lengths_cpu, batch_first=True, enforce_sorted=False)
        _, (ht, _) = self.lstm(packed_input)

        # Shape output is [Batch, 1], matches BCEWithLogitsLoss perfectly
        return self.classifier(ht[-1])


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
            batch_lengths = batch_lengths.to(device)
            target_tensor = torch.full((batch_logits.size(0), 1), target_class, dtype=torch.float32, device=device)

            optimizer.zero_grad()

            # 1. Add Steering Vector to Raw Logits
            steered_logits = batch_logits + S.unsqueeze(0).unsqueeze(0)

            # # 2. Smooth, direct Softmax (No more STE needed!)
            # simulated_routing_probs = F.softmax(steered_logits, dim=-1)

            # 2. Probability STE Trick
            TEMP_BACKWARD = 5.0 # Smooths the distribution for healthy gradients
            sharp_probs = F.softmax(steered_logits, dim=-1)
            soft_probs = F.softmax(steered_logits / TEMP_BACKWARD, dim=-1)
            
            # Forward pass is sharp_probs, backward pass flows through soft_probs
            simulated_routing_probs = sharp_probs.detach() - soft_probs.detach() + soft_probs

            # 3. Predict & Calculate Loss
            lstm_preds = lstm_model(simulated_routing_probs, batch_lengths)

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
            steered = logits + (self.alpha * self.steering_vector.to(logits.device))
            if isinstance(output, tuple):
                return (steered,) + output[1:]
            elif isinstance(output, list):
                return [steered] + output[1:]
            return steered
        else:
            return output + (self.alpha * self.steering_vector.to(output.device))


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
    checkpoint = torch.load(f"{root_folder}/results/trained_lstm_models/{model_config.model_name}/{model_config.model_name}_lstm.pkl", map_location=device)
    NUM_TOTAL_EXPERTS, NUM_LAYERS = checkpoint["num_total_experts"], checkpoint["num_layers"]

    lstm = MoETraceClassifierLinear(NUM_TOTAL_EXPERTS, NUM_LAYERS)
    lstm.load_state_dict(checkpoint["model_state_dict"])
    lstm = lstm.to(device)

    questions, labels = data_utils.load_adult_set(root_folder, model_config.model_name, malicious_only=True)
    questions = questions[:1028]  # Sample for discovery

    # Pre-tokenize questions for alignment
    tokenized_qs = []
    for q_text in questions:
        if model_config.model_name == "deepseek-moe-16b-chat":
            q_text = " " + q_text
        elif model_config.model_name == "Mixtral-8x7B-Instruct-v0.1":
            q_text = "\n" + q_text
        q_ids = tokenizer(q_text, add_special_tokens=False)["input_ids"]
        if model_config.model_name == "Mixtral-8x7B-Instruct-v0.1":
            q_ids = q_ids[2:]
        tokenized_qs.append(q_ids)

    prompts = data_utils.construct_prompt(tokenizer, questions, model_config.model_name)
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

    print(f"Collecting baseline routing logits for {len(questions)} prompts...")

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
            if global_idx >= len(questions):
                break

            start, end = find_token_range(tokenized_qs[global_idx], input_ids_np[p_idx])
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
    lambdas = [0,1e-6,1e-5,1e-4,1e-3]

    for _lambda in lambdas:
        S_optimized = discover_universal_steering_circuit(
            lstm_model=lstm,
            dataloader=trace_dataloader,
            num_layers=NUM_LAYERS,
            num_total_experts=NUM_TOTAL_EXPERTS,
            target_class=0.0,
            l1_lambda=_lambda,
            lr=0.05,
            epochs=80,
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

        steering_alphas = [0.6, 0.5, 0.4, 0.3, 0.2]

        for steering_alpha in steering_alphas:
            # Apply Inference Steering
            print("\n--- Applying Steering Hooks to LLM ---", flush=True)
            active_steering_hooks = apply_steering_hooks(model_config.model_name, model, model_config.gate_name, S_sparse, steering_alpha)

            print("\nGenerating Responses with Steering Active...")
            eval_prompts = data_utils.construct_prompt(tokenizer, questions[:10], model_config.model_name)
            eval_responses = model_utils.generate_output(model, model_config.model_name, tokenizer, eval_prompts, batch_size=5)

            for hook in active_steering_hooks:
                hook.remove()

            print(f"\n--- Model Responses for l1_lambda={_lambda}, threshold={threshold}, and alpha={steering_alpha}---")
            for q, r in zip(questions[:10], eval_responses):
                print(f"\nPrompt: {[q]}")
                print(f"Response: {[r.strip()]}")

    print("\n------------------ Job Finished ------------------")
