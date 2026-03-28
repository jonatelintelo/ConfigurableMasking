import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset
import sys

# Modules defined in your project
import argument_parser as argument_parser
import moe_model_files.model_utils as model_utils
import moe_model_files.model_configurations as model_configurations
import data.data_utils as data_utils


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

        # Move lengths to CPU for pack_padded_sequence as required by PyTorch
        lengths_cpu = lengths.cpu()
        packed_input = pack_padded_sequence(x_flat, lengths_cpu, batch_first=True, enforce_sorted=False)
        _, (ht, _) = self.lstm(packed_input)
        return self.classifier(ht[-1])


# ==========================================
# 2. Phase 1: Universal Circuit Discovery
# ==========================================
def discover_universal_steering_circuit(
    lstm_model,
    dataloader,
    num_layers,
    num_total_experts,
    top_k,
    target_class,  # 0.0 = Answer, 1.0 = Refusal
    l1_lambda,  # Higher = sparser circuit
    lr,
    epochs,
):
    """
    Optimizes a universal global steering tensor S using mini-batches
    to find the minimal circuit that flips the LSTM's prediction.
    """
    device = next(lstm_model.parameters()).device
    # S is our universal, batch-independent steering matrix
    S = nn.Parameter(torch.zeros(num_layers, num_total_experts, device=device))
    optimizer = torch.optim.Adam([S], lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    lstm_model.eval()  # Ensure LSTM is frozen
    for param in lstm_model.parameters():
        param.requires_grad = False

    print("\nStarting Universal Circuit Discovery...")
    for epoch in range(epochs):
        total_loss = 0
        bce_loss_sum = 0

        for batch_logits, batch_lengths in dataloader:
            batch_logits = batch_logits.to(device)
            batch_lengths = batch_lengths.to(device)
            target_tensor = torch.full((batch_logits.size(0), 1), target_class, dtype=torch.float32, device=device)

            optimizer.zero_grad()

            # Broadcast universal S to the current batch
            steered_logits = batch_logits + S.unsqueeze(0).unsqueeze(0)

            # Differentiable Routing Proxy
            simulated_routing_trace = F.softmax(steered_logits, dim=-1) * top_k

            # Forward pass through the surrogate (LSTM)
            lstm_logits = lstm_model(simulated_routing_trace, batch_lengths)

            # Calculate Loss: BCE + L1
            bce_loss = criterion(lstm_logits, target_tensor)
            l1_loss = l1_lambda * torch.norm(S, p=1)

            loss = bce_loss + l1_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            bce_loss_sum += bce_loss.item()

        avg_loss = total_loss / len(dataloader)
        avg_bce = bce_loss_sum / len(dataloader)
        active_experts = (torch.abs(S) > 0.1).sum().item()

        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} | Avg Loss: {avg_loss:.4f} (BCE: {avg_bce:.4f}) | Active Circuit Nodes: {active_experts}")

    return S.detach()


# ==========================================
# 3. Phase 2: Inference Steering Hooks
# ==========================================
class ModelAwareSteeringHook:
    def __init__(self, model_name, layer_name, steering_vector, alpha=1.0):
        self.model_name = model_name
        self.layer_name = layer_name
        self.steering_vector = steering_vector
        self.alpha = alpha

    def __call__(self, module, inputs, output):
        # 1. Standard Models
        if self.model_name in [
            "Qwen3-30B-A3B-Instruct-2507",
            "Phi-3.5-MoE-instruct",
            "Mixtral-8x7B-Instruct-v0.1",
            "gpt-oss-20b",
            "Qwen1.5-MoE-A2.7B-Chat",
            "Hunyuan-A13B-Instruct",
        ]:
            steered_logits = output + (self.alpha * self.steering_vector.to(output.device))
            return steered_logits

        # 2. DeepSeek MoE
        elif self.model_name == "deepseek-moe-16b-chat":
            logits = output[0]
            steered_logits = logits + (self.alpha * self.steering_vector.to(logits.device))
            if isinstance(output, tuple):
                return (steered_logits,) + output[1:]
            elif isinstance(output, list):
                return [steered_logits] + output[1:]
            return steered_logits

        # 3. PanGu Pro MoE
        elif self.model_name == "pangu-pro-moe-model":
            steered_logits = output + (self.alpha * self.steering_vector.to(output.device))
            return steered_logits

        else:
            return output + (self.alpha * self.steering_vector.to(output.device))


def apply_steering_hooks(model_name, model, gate_name, sparse_S, alpha=1.0):
    hook_handles = []
    layer_idx = 0
    for layer_name, module in model.named_modules():
        if layer_name.lower().endswith(gate_name):
            layer_steering_vector = sparse_S[layer_idx]
            if torch.any(layer_steering_vector != 0):
                hook_fn = ModelAwareSteeringHook(model_name=model_name, layer_name=layer_name, steering_vector=layer_steering_vector, alpha=alpha)
                handle = module.register_forward_hook(hook_fn)
                hook_handles.append(handle)
                print(f"Injected steering hook on: '{layer_name}'")
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

    # Hyperparameters
    SEQ_LEN = 32
    CHUNK_SIZE = 8  # Number of prompts to process at once during collection to avoid OOM
    BATCH_SIZE = 32  # Batch size for LSTM gradient descent

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if print_logging:
        print(f"\nPython version: {sys.version}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

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
    model = model.to(device)

    # 1. Initialize LSTM
    print("\nLoading trained LSTM model...")
    checkpoint_path = f"{root_folder}/results/trained_lstm_models/{model_config.model_name}/{model_config.model_name}_lstm.pkl"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    NUM_TOTAL_EXPERTS = checkpoint["num_total_experts"]
    NUM_LAYERS = checkpoint["num_layers"]

    lstm = MoETraceClassifierLinear(num_total_experts=NUM_TOTAL_EXPERTS, num_layers=NUM_LAYERS)
    lstm.load_state_dict(checkpoint["model_state_dict"])
    lstm = lstm.to(device)

    # 2. Extract Actual "Refusal" Traces
    print("\nLoading dataset and constructing prompts...")
    # Load ONLY malicious prompts (label 1) to map the refusal circuit
    questions, labels = data_utils.load_adult_set(root_folder, model_config.model_name, malicious_only=True)

    all_logits = []
    all_lengths = []

    print(f"Collecting baseline routing logits for {len(questions)} prompts...")

    # Process in chunks to manage GPU memory
    for i in range(0, len(questions), CHUNK_SIZE):
        chunk_questions = questions[i : i + CHUNK_SIZE]
        prompts = data_utils.construct_prompt(tokenizer, chunk_questions, model_config.model_name)

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=SEQ_LEN).to(device)

        lengths = inputs.attention_mask.sum(dim=1)

        collected_logits = {}

        # Temporary hook strictly for trace collection
        def get_collection_hook(layer_index):
            def hook(module, inp, out):
                if model_config.model_name == "deepseek-moe-16b-chat":
                    logits = out[0]
                else:
                    logits = out
                collected_logits[layer_index] = logits.detach()

            return hook

        collection_handles = []
        layer_idx = 0
        for layer_name, module in model.named_modules():
            if layer_name.lower().endswith(model_config.gate_name):
                handle = module.register_forward_hook(get_collection_hook(layer_idx))
                collection_handles.append(handle)
                layer_idx += 1

        # Run forward pass (no gradients needed for collection)
        with torch.no_grad():
            model(**inputs)

        # Remove hooks after collection
        for handle in collection_handles:
            handle.remove()

        # Stack into (Batch, Seq_Len, Num_Layers, Num_Total_Experts) and move to CPU to save VRAM
        actual_raw_logits = torch.stack([collected_logits[j] for j in range(NUM_LAYERS)], dim=2).cpu()
        all_logits.append(actual_raw_logits)
        all_lengths.append(lengths.cpu())

        print(f"  Processed {min(i + CHUNK_SIZE, len(questions))}/{len(questions)} prompts")

    # Create Universal Dataloader
    dataset_logits = torch.cat(all_logits, dim=0)
    dataset_lengths = torch.cat(all_lengths, dim=0)

    trace_dataset = TensorDataset(dataset_logits, dataset_lengths)
    trace_dataloader = DataLoader(trace_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Discover the Universal Circuit
    S_optimized = discover_universal_steering_circuit(
        lstm_model=lstm,
        dataloader=trace_dataloader,
        num_layers=NUM_LAYERS,
        num_total_experts=NUM_TOTAL_EXPERTS,
        top_k=model_config.top_k,
        target_class=0.0,  # 0.0 = Answer, 1.0 = Refusal
        l1_lambda=0.08,  # Tune this: higher = sparser, fewer experts targeted
        lr=0.05,
        epochs=15,
    )

    # 4. Extract the Minimal Circuit (Sparsify)
    threshold = 0.1
    S_sparse = torch.where(torch.abs(S_optimized) > threshold, S_optimized, torch.zeros_like(S_optimized))

    print("\n--- Discovered Universal Circuit ---")
    active_count = 0
    for layer in range(NUM_LAYERS):
        active = torch.nonzero(S_sparse[layer]).squeeze(-1)
        if len(active) > 0:
            for exp in active:
                val = S_sparse[layer, exp].item()
                action = "BOOST" if val > 0 else "DIMINISH"
                print(f"Layer {layer}, Expert {exp}: {action} (Weight: {val:.3f})")
                active_count += 1
    if active_count == 0:
        print("No active experts found! Try decreasing l1_lambda in the discovery function.")

    # 5. Apply during Inference
    print("\n--- Applying Steering Hooks to LLM ---")
    # Set a default alpha, this is what you will tune later
    STEERING_ALPHA = 1.0
    active_steering_hooks = apply_steering_hooks(
        model_name=model_config.model_name, model=model, gate_name=model_config.gate_name, sparse_S=S_sparse, alpha=STEERING_ALPHA
    )

    print(f"\nModel {model_config.model_name} is now primed with steering hooks (Alpha={STEERING_ALPHA}). Ready for inference generation.")

    print("\nGenerating Responses with Soft Positive Steering Active...")
    eval_prompts = data_utils.construct_prompt(tokenizer, questions[:10], model_config.model_name)
    eval_responses = model_utils.generate_output(model, model_config.model_name, tokenizer, eval_prompts, batch_size=10)

    # To cleanly remove hooks when done with all generation:
    for hook in active_steering_hooks:
        hook.remove()

    print("\n--- Model Responses ---")
    for q, r in zip(questions[:10], eval_responses):
        print(f"\nPrompt: {q}")
        print(f"Response: {[r.strip()]}", flush=True)