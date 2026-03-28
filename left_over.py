def find_temporal_universal_circuit(trained_lstm, dataloader, window_size, num_layers, num_experts, device, lambda_reg=0.05, epochs=50):
    """
    Optimizes a mask over an aligned temporal window (e.g., the last N tokens of a prompt)
    to find the sequence-specific safety circuit.
    """
    trained_lstm.eval()
    for param in trained_lstm.parameters():
        param.requires_grad = False

    # 1. Initialize a mask that explicitly models the temporal window
    # Shape: (1, window_size, num_layers, num_experts)
    windowed_mask = nn.Parameter(torch.ones(1, window_size, num_layers, num_experts, device=device))

    optimizer = optim.Adam([windowed_mask], lr=0.05)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0

        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # batch_x MUST be pre-aligned and truncated/padded to exactly `window_size`
            # Shape: (batch_size, window_size, num_layers, num_experts)

            constrained_mask = torch.clamp(windowed_mask, 0.0, 1.0)

            # Mask applies to specific temporal positions relative to the end of the prompt
            masked_input = batch_x * constrained_mask

            # Flatten spatial dimensions for LSTM
            batch_size = masked_input.size(0)
            lstm_input = masked_input.view(batch_size, window_size, -1)

            logits = trained_lstm(lstm_input)

            cls_loss = criterion(logits, batch_y)
            l1_loss = lambda_reg * torch.sum(torch.abs(constrained_mask))

            loss = cls_loss + l1_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            active_nodes = (windowed_mask > 0.5).sum().item()
            print(f"Epoch {epoch} | Loss: {total_loss/len(dataloader):.4f} | Active Circuit Nodes: {active_nodes}")

    final_mask = torch.clamp(windowed_mask, 0.0, 1.0).detach().squeeze()

    # Returns shape (window_size, num_layers, num_experts)
    temporal_circuit = (final_mask > 0.5).int()

    return temporal_circuit, final_mask


    import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset

# Custom module imports
import moe_model_files.model_configurations as model_configurations
import data.data_utils as data_utils
import moe_model_files.model_utils as model_utils
import argument_parser as argument_parser
import lstm.lstm_data as lstm_data

# ==========================================
# 0. GLOBALS & SEEDING
# ==========================================
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ==========================================
# 1. HOOK DEFINITIONS (INFERENCE ABLATION)
# ==========================================
def register_temporal_steering_hooks(model_name, model, banned_candidates, gate_name, batch_size, intervention_window, template_offset, debug):
    """Wraps the model's MoE gates with temporal ablation hooks."""
    hook_handles = []
    first_layer_attached = False 

    for layer_name, module in model.named_modules():
        if layer_name.lower().endswith(gate_name.lower()):
            if layer_name in banned_candidates and len(banned_candidates[layer_name]) > 0:
                banned_experts = banned_candidates[layer_name]
                
                # Only enable debug printouts for the very first attached layer to prevent console spam
                enable_layer_debug = debug and not first_layer_attached
                
                hook_fn = get_temporal_steering_hook(
                    model_name, layer_name, banned_experts, batch_size, 
                    intervention_window, template_offset, debug=enable_layer_debug
                )
                handle = module.register_forward_hook(hook_fn)
                hook_handles.append(handle)
                
                first_layer_attached = True

    return hook_handles

def get_temporal_steering_hook(model_name, layer_name, banned_experts, expected_batch_size, intervention_window, template_offset, debug):
    """Returns a hook that subtracts a massive logit value from banned experts to prevent routing to them."""
    if debug:
        print(f"Attached ablation hook to '{layer_name}' | Banned Experts: {banned_experts}")

    def steering_hook(module, input, output):
        logits = output[0] if model_name == "deepseek-moe-16b-chat" else output
        is_flattened = logits.dim() == 2
        
        if is_flattened:
            total_tokens, num_experts = logits.shape
            seq_len = total_tokens // expected_batch_size
            logits = logits.view(expected_batch_size, seq_len, num_experts)
        else:
            seq_len = logits.shape[1]

        mask = torch.zeros_like(logits)

        # --- ABLATION LOGIC ---
        if len(banned_experts) > 0:
            banned_idx = torch.tensor(banned_experts, device=logits.device, dtype=torch.long)
            
            # PHASE A: Prefill (Prompt Processing)
            if seq_len > 1:
                # Calculate where the intervention window starts relative to the end of the full sequence
                start_idx = -(intervention_window + template_offset)
                if seq_len < abs(start_idx):
                    start_idx = 0
                
                # Mask from the start of the intervention window all the way through the template suffix
                mask[:, start_idx:, banned_idx] = -10000000000

                if debug and seq_len > abs(start_idx):
                    normal_token_experts = torch.topk((logits + mask)[0, 0, :], k=2).indices.tolist()
                    steered_token_experts = torch.topk((logits + mask)[0, start_idx, :], k=2).indices.tolist()
                    template_token_experts = torch.topk((logits + mask)[0, -1, :], k=2).indices.tolist()
                    
                    print(f"\n[DEBUG] {layer_name} Routing:")
                    print(f"  Token at pos 0            (Normal)   -> Chose: {normal_token_experts}")
                    print(f"  Token at pos {start_idx}  (Steered)  -> Chose: {steered_token_experts} | Banned: {banned_experts}")
                    print(f"  Token at pos -1           (Template) -> Chose: {template_token_experts}")

            # PHASE B: Decoding (Autoregressive Generation)
            else:
                # Keep ablation permanently active for every generated token
                mask[:, :, banned_idx] = -10000000000

        modified_logits = logits + mask

        if is_flattened:
            modified_logits = modified_logits.view(-1, logits.shape[-1])

        if model_name == "deepseek-moe-16b-chat" and isinstance(output, tuple):
            return (modified_logits,) + output[1:]
        return modified_logits

    return steering_hook

# ==========================================
# 2. LSTM MODEL DEFINITIONS (DISCOVERY)
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
        packed_input = pack_padded_sequence(x_flat, lengths, batch_first=True, enforce_sorted=False)
        _, (ht, _) = self.lstm(packed_input)
        return self.classifier(ht[-1])

def find_targeted_refusal_circuit(trained_lstm, dataloader, num_layers, num_experts, device, intervention_window, lambda_reg, epochs):
    """Optimizes a spatial mask applied only to the intervention window to find refusal experts."""
    trained_lstm.train()
    for param in trained_lstm.parameters():
        param.requires_grad = False

    circuit_mask = nn.Parameter(torch.ones(1, 1, num_layers, num_experts, device=device))
    optimizer = optim.Adam([circuit_mask], lr=0.05)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y, batch_lens in dataloader:
            batch_size, max_seq_len, _, _ = batch_x.shape
            batch_x = batch_x.to(device)
            batch_y = batch_y.float().unsqueeze(1).to(device)
            lengths_tensor = torch.tensor(batch_lens, device=device)

            x_multihot = torch.zeros((batch_size, max_seq_len, num_layers, num_experts), dtype=torch.float32, device=device)
            x_multihot.scatter_(dim=3, index=batch_x, src=torch.ones_like(batch_x, dtype=torch.float32))

            optimizer.zero_grad()
            constrained_mask = torch.clamp(circuit_mask, 0.0, 1.0)

            positions = torch.arange(max_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            in_window = (positions >= (lengths_tensor.unsqueeze(1) - intervention_window)) & (positions < lengths_tensor.unsqueeze(1))
            in_window = in_window.unsqueeze(-1).unsqueeze(-1)

            masked_input = torch.where(in_window, x_multihot * constrained_mask, x_multihot)
            logits = trained_lstm(masked_input, batch_lens)

            cls_loss = criterion(logits, batch_y)
            l1_loss = lambda_reg * torch.sum(torch.abs(constrained_mask))

            loss = cls_loss + l1_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            active_nodes = (circuit_mask > 0.5).sum().item()
            print(f"Epoch {epoch} | Loss: {total_loss/len(dataloader):.4f} | Active Circuit Nodes: {active_nodes}")

    final_mask = torch.clamp(circuit_mask, 0.0, 1.0).detach().squeeze()
    circuit = (final_mask > 0.5).int()
    return circuit, final_mask

def calculate_template_offset(tokenizer, raw_question, full_prompt):
    """Calculates exact token count of the chat template suffix."""
    try:
        end_idx = full_prompt.rindex(raw_question) + len(raw_question)
    except ValueError:
        raise ValueError("Raw question not found in formatted prompt.")
    text_up_to_question = full_prompt[:end_idx]
    full_tokens = tokenizer.encode(full_prompt, add_special_tokens=False)
    up_to_question_tokens = tokenizer.encode(text_up_to_question, add_special_tokens=False)
    return len(full_tokens) - len(up_to_question_tokens)

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    arguments = argument_parser.parse_arguments()
    root_folder = arguments.root
    model_id = arguments.model_id
    print_logging = arguments.print_logging

    if print_logging:
        print(f"\nSystem Overview:")
        print(f"Python: {sys.version.split(' ')[0]} | PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")
        print(f"GPUs: {torch.cuda.device_count()} | Primary GPU: {torch.cuda.get_device_name(0)}")

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

    # --- Setup Devices and Base Components ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = model_utils.load_model(models[model_id])
    
    # --- Load LSTM Component ---
    print("\nLoading trained LSTM model and Traces...")
    checkpoint_path = f"{root_folder}/results/trained_lstm_models/{model_config.model_name}/{model_config.model_name}_lstm.pkl"
    checkpoint = torch.load(checkpoint_path)
    
    NUM_TOTAL_EXPERTS = checkpoint["num_total_experts"]
    NUM_LAYERS = checkpoint["num_layers"]
    
    loaded_model = MoETraceClassifierLinear(num_total_experts=NUM_TOTAL_EXPERTS, num_layers=NUM_LAYERS)
    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    loaded_model = loaded_model.to(device)

    # --- Load Data ---
    traces = data_utils.load_data(f"{root_folder}/results/lstm_input/{model_config.model_name}/{model_config.model_name}_traces.pkl")
    labels = data_utils.load_data(f"{root_folder}/results/lstm_input/{model_config.model_name}/{model_config.model_name}_labels.pkl")
    original_dataset = lstm_data.MoETraceDataset(traces, labels)

    # Prepare Evaluation Prompts (Do this ONCE outside the loop)
    print("\nPreparing Evaluation Dataset (StrongREJECT)...")
    ds = load_dataset("walledai/StrongREJECT")
    eval_questions = ds["train"]["prompt"][:10]
    eval_prompts = data_utils.construct_prompt(tokenizer, eval_questions, model_config.model_name)
    exact_offset = calculate_template_offset(tokenizer, eval_questions[0], eval_prompts[0])
    print(f"Calculated Template Offset: {exact_offset} tokens")

    # Get Gate Names
    gate_names = [name for name, _ in model.named_modules() if name.lower().endswith(model_config.gate_name.lower())]
    if len(gate_names) != NUM_LAYERS:
        print(f"Warning: Found {len(gate_names)} '{model_config.gate_name}' modules but LSTM was trained on {NUM_LAYERS} layers.")

    # --- Execution Parameters ---
    BATCH_SIZE = 64
    EPOCHS = 50
    INTERVENTION_WINDOW = 3  # Number of final tokens to optimize mask over

    # --- Circuit Discovery & Ablation Loop ---
    lambda_values = [1e-6,1e-7,1e-8,1e-9,0]
    
    for l_reg in lambda_values:
        print(f"\n{'='*50}")
        print(f"🚀 STARTING RUN FOR LAMBDA_REG = {l_reg}")
        print(f"{'='*50}")

        circuit_dataloader = lstm_data.get_dataLoader(original_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # 1. Discover Refusal Circuit
        circuit, continuous_weights = find_targeted_refusal_circuit(
            trained_lstm=loaded_model,
            dataloader=circuit_dataloader,
            num_layers=NUM_LAYERS,
            num_experts=NUM_TOTAL_EXPERTS,
            device=device,
            intervention_window=INTERVENTION_WINDOW,
            lambda_reg=l_reg,
            epochs=EPOCHS,
        )

        active_nodes = circuit.sum().item()
        print(f"\n✅ Circuit extraction complete. Total active experts: {active_nodes}")

        # Extract and format circuit
        nonzero_indices = torch.nonzero(circuit)
        circuit_list = [(layer, expert) for layer, expert in nonzero_indices.tolist()]
        
        # 2. Map Circuit to Model Gates
        banned_candidates = {name: [] for name in gate_names}
        for layer_idx, expert_idx in circuit_list:
            if layer_idx < len(gate_names):
                banned_candidates[gate_names[layer_idx]].append(expert_idx)

        print("\n--- Banned Mask Summary ---")
        for layer_name, experts in banned_candidates.items():
            if len(experts) > 0:
                print(f"{layer_name}: {len(experts)} experts banned -> {experts}")

        # 3. Register Hooks
        pruning_handles = register_temporal_steering_hooks(
            model_name=model_config.model_name,
            model=model,
            banned_candidates=banned_candidates,
            gate_name=model_config.gate_name,
            batch_size=10,
            intervention_window=INTERVENTION_WINDOW,
            template_offset=exact_offset,
            debug=True,
        )

        # 4. Generate Responses
        print("\nGenerating Responses with Ablation Active...")
        responses = model_utils.generate_output(
            model, model_config.model_name, tokenizer, eval_prompts, batch_size=10
        )

        # 5. Cleanup Hooks
        for handle in pruning_handles:
            handle.remove()

        # 6. Output Results
        print("\n--- Model Responses ---")
        for q, r in zip(eval_questions, responses):
            print(f"\nPrompt: {q}")
            print(f"Response: {[r.strip()]}", flush=True)

    print("\n------------------ Job Finished ------------------")