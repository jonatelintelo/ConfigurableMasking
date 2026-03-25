import moe_model_files.model_configurations as model_configurations
import data.data_utils as data_utils
import argument_parser as argument_parser
import lstm.lstm_model as lstm_model
import lstm.lstm_data as lstm_data

from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import sys

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def find_universal_refusal_circuit(trained_lstm, dataloader, num_layers, num_experts, device, lambda_reg=0.05, epochs=50):
    """
    Optimizes a purely spatial mask (sequence-invariant) to find a single 
    static set of experts that trigger the refusal classification.
    """
    trained_lstm.eval()
    for param in trained_lstm.parameters():
        param.requires_grad = False

    # 1. Initialize a strictly Spatial Mask
    # Shape: (1, 1, num_layers, num_experts) - The second '1' allows it to broadcast over any sequence length
    circuit_mask = nn.Parameter(torch.ones(1, 1, num_layers, num_experts, device=device))

    optimizer = optim.Adam([circuit_mask], lr=0.05)
    
    # Use BCEWithLogitsLoss for binary classification (1 logit output)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        total_loss = 0.0

        for batch_x, batch_y, batch_lens in dataloader:
            batch_size, max_seq_len, _, _ = batch_x.shape
            
            batch_x = batch_x.to(device)
            # Format Y for BCE Loss: (Batch, 1)
            batch_y = batch_y.float().unsqueeze(1).to(device)

            # --- 2. Convert to Multi-Hot on the fly ---
            x_multihot = torch.zeros(
                (batch_size, max_seq_len, num_layers, num_experts), 
                dtype=torch.float32, 
                device=device
            )
            x_multihot.scatter_(dim=3, index=batch_x, src=torch.ones_like(batch_x, dtype=torch.float32))

            optimizer.zero_grad()

            constrained_mask = torch.clamp(circuit_mask, 0.0, 1.0)

            # --- 3. Apply the Spatial Mask ---
            # PyTorch automatically broadcasts the (1, 1, L, E) mask across the max_seq_len dimension
            masked_input = x_multihot * constrained_mask

            # --- 4. Forward Pass ---
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

    # Returns shape (num_layers, num_experts)
    circuit = (final_mask > 0.5).int()

    return circuit, final_mask

def find_targeted_refusal_circuit(trained_lstm, dataloader, num_layers, num_experts, device, intervention_window=5, lambda_reg=0.05, epochs=50):
    """
    Optimizes a single spatial mask (sequence-invariant), but ONLY applies it to 
    the final `intervention_window` tokens of each prompt during the LSTM forward pass.
    """
    trained_lstm.eval()
    for param in trained_lstm.parameters():
        param.requires_grad = False

    # 1. Initialize a strictly Spatial Mask
    # Shape: (1, 1, num_layers, num_experts)
    circuit_mask = nn.Parameter(torch.ones(1, 1, num_layers, num_experts, device=device))

    optimizer = optim.Adam([circuit_mask], lr=0.05)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        total_loss = 0.0

        for batch_x, batch_y, batch_lens in dataloader:
            batch_size, max_seq_len, _, _ = batch_x.shape
            
            batch_x = batch_x.to(device)
            batch_y = batch_y.float().unsqueeze(1).to(device)
            
            # Move lengths to device for tensor math
            # batch_lens is a tuple/list from your collate_fn, convert to tensor
            lengths_tensor = torch.tensor(batch_lens, device=device)

            # --- 2. Convert to Multi-Hot on the fly ---
            x_multihot = torch.zeros(
                (batch_size, max_seq_len, num_layers, num_experts), 
                dtype=torch.float32, 
                device=device
            )
            x_multihot.scatter_(dim=3, index=batch_x, src=torch.ones_like(batch_x, dtype=torch.float32))

            optimizer.zero_grad()
            constrained_mask = torch.clamp(circuit_mask, 0.0, 1.0)

            # --- 3. Identify the Target Window dynamically ---
            # Create a sequence of position indices: [0, 1, 2, ..., max_seq_len-1]
            positions = torch.arange(max_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            
            # A token is in the intervention window IF:
            # It is >= (length - window) AND it is < length (ignoring PAD tokens)
            in_window = (positions >= (lengths_tensor.unsqueeze(1) - intervention_window)) & \
                        (positions < lengths_tensor.unsqueeze(1))
            
            # Reshape boolean mask for broadcasting: (Batch, SeqLen, 1, 1)
            in_window = in_window.unsqueeze(-1).unsqueeze(-1)

            # --- 4. Apply the Mask ONLY to the target window ---
            # If in_window is True, apply (x_multihot * mask). If False, leave x_multihot alone.
            masked_input = torch.where(in_window, x_multihot * constrained_mask, x_multihot)

            # --- 5. Forward Pass ---
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

    # Returns shape (num_layers, num_experts)
    circuit = (final_mask > 0.5).int()

    return circuit, final_mask

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

    print("\nLoading trained LSTM model...")
    checkpoint = torch.load(f"{root_folder}/results/trained_lstm_models/{model_config.model_name}/{model_config.model_name}_lstm.pkl")
    loaded_model = lstm_model.MoETraceClassifierLinear(
        num_total_experts=checkpoint["num_total_experts"], num_layers=checkpoint["num_layers"], top_k=checkpoint["top_k"]
    )
    loaded_model.load_state_dict(checkpoint["model_state_dict"])

    print("\nLoading precomputed traces...")
    traces = data_utils.load_data(f"{root_folder}/results/lstm_input/{model_config.model_name}/{model_config.model_name}_traces.pkl")
    labels = data_utils.load_data(f"{root_folder}/results/lstm_input/{model_config.model_name}/{model_config.model_name}_labels.pkl")
    original_dataset = lstm_data.MoETraceDataset(traces, labels)

    loaded_model.eval()

    # --- Setup Parameters ---
    BATCH_SIZE = 64
    EPOCHS = 50
    LAMBDA_REG = 0.05 # Adjust this to make the circuit more or less sparse (higher = fewer experts)
    
    NUM_TOTAL_EXPERTS = checkpoint["num_total_experts"]
    NUM_LAYERS = checkpoint["num_layers"]

    # Determine device and move model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model = loaded_model.to(device)

    # Initialize the standard dataloader (which uses your pad_collate function)
    circuit_dataloader = lstm_data.get_dataLoader(original_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"\n🚀 Starting Universal Circuit Discovery (Spatial Mask)...")
    print(f"Dataset Size: {len(original_dataset)} prompts | Batch Size: {BATCH_SIZE}")

    # --- Run Circuit Discovery ---
    circuit, continuous_weights = find_universal_refusal_circuit(
        trained_lstm=loaded_model,
        dataloader=circuit_dataloader,
        num_layers=NUM_LAYERS,
        num_experts=NUM_TOTAL_EXPERTS,
        device=device,
        lambda_reg=LAMBDA_REG,
        epochs=EPOCHS,
    )

    print("\n✅ Circuit extraction complete.")
    active_nodes = circuit.sum().item()
    print(f"Total active experts in the spatial circuit: {active_nodes}")

    # --- Print and Extract the Results ---
    print("\n--- Discovered Refusal Circuit ---")
    
    # circuit shape is (num_layers, num_experts)
    nonzero_indices = torch.nonzero(circuit)
    
    circuit_list = []
    for idx in nonzero_indices:
        layer, expert = idx.tolist()
        weight = continuous_weights[layer, expert].item()
        circuit_list.append((layer, expert))
        print(f"Layer: {layer:02d} | Expert: {expert:02d} | Mask Weight: {weight:.4f}")

    # --- Save the Circuit for Inference Interventions ---
    save_path = f"{root_folder}/results/trained_lstm_models/{model_config.model_name}/{model_config.model_name}_spatial_circuit.pt"
    
    # We save both the binary mask tensor and a simple python list of coordinates
    torch.save({
        'circuit_tensor': circuit.cpu(),
        'circuit_coordinates': circuit_list
    }, save_path)
    
    print(f"\n💾 Saved circuit configuration to: {save_path}")

    print("\n------------------ Job Finished ------------------")
