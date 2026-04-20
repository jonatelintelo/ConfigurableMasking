import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import argument_parser as argument_parser


def calculate_expert_frequencies(topk_data, num_experts=None):
    """
    Calculates the activation frequency of each expert per layer.
    """
    # From the collection script: tensor shape is (seq_len, num_layers, top_k)
    num_layers = topk_data[0].shape[1]
    
    # Dynamically find the max expert index if not provided
    if num_experts is None:
        max_expert_idx = max(int(tensor.max().item()) for tensor in topk_data)
        num_experts = max_expert_idx + 1

    # Initialize count matrix: (num_layers, num_experts)
    expert_counts = np.zeros((num_layers, num_experts))

    for tensor in topk_data:
        for layer_idx in range(num_layers):
            # Flatten all expert selections for this layer across all tokens
            layer_experts = tensor[:, layer_idx, :].flatten().cpu().numpy()
            
            # Count occurrences
            unique, counts = np.unique(layer_experts, return_counts=True)
            expert_counts[layer_idx, unique.astype(int)] += counts

    # Normalize counts to get frequencies (each row sums to 1)
    row_sums = expert_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Prevent division by zero
    frequencies = expert_counts / row_sums
    
    return frequencies, num_experts

def main():
    # Adjust these paths to match your folder structure
    arguments = argument_parser.parse_arguments()
    root_folder = arguments.root

    all_models = [
        ("Qwen/Qwen3-30B-A3B-Instruct-2507", "Qwen3"),
        ("microsoft/Phi-3.5-MoE-instruct", "Phi"),
        ("mistralai/Mixtral-8x7B-Instruct-v0.1", "Mixtral"),
        ("openai/gpt-oss-20b", "GPT"),
        ("Qwen/Qwen1.5-MoE-A2.7B-Chat", "Qwen1.5"),
        ("tencent/Hunyuan-A13B-Instruct", "Hunyuan"),
        ("deepseek-ai/deepseek-moe-16b-chat", "DeepSeek"),
    ]

    model_name = "Phi-3.5-MoE-instruct"
    print(f"\nInitializing: {model_name}")
    
    baseline_path = os.path.join(root_folder, "results", "topk", "jailbreak", f"{model_name}_baseline_topk.pt")
    steered_path = os.path.join(root_folder, "results", "topk", "jailbreak", f"{model_name}_steered_topk.pt")

    print("Loading tensors...")
    baseline_data = torch.load(baseline_path)
    steered_data = torch.load(steered_path)

    print("Calculating frequencies...")
    # Calculate baseline first to find the total number of experts
    baseline_freq, num_experts = calculate_expert_frequencies(baseline_data)
    # Pass the same num_experts to steered to ensure identical matrix dimensions
    steered_freq, _ = calculate_expert_frequencies(steered_data, num_experts=num_experts)

    # Calculate the difference (Steered - Baseline)
    # Positive: Selected MORE often during steering
    # Negative: Selected LESS often during steering
    diff_freq = steered_freq - baseline_freq

    print("Generating plot...")
    plt.figure(figsize=(12, 10)) # Slightly adjusted to accommodate the square aspect ratio
    
    # To make the diverging colormap symmetric around 0, find the max absolute difference
    max_abs_diff = np.max(np.abs(diff_freq))
    
    # Plot single heatmap using a diverging colormap
    ax = sns.heatmap(
        diff_freq, 
        cmap="RdBu_r",       
        center=0,              
        vmin=-max_abs_diff,    
        vmax=max_abs_diff,     
        square=True,           # <-- Forces cells to be perfect squares
        cbar_kws={'label': 'Change in Frequency (Steered - Baseline)'}
    )
    
    # Removes the little tick lines on the axes
    ax.tick_params(left=False, bottom=False)
    
    ax.set_title(f"Change in Expert Activation Frequency\n({model_name})", fontsize=14, pad=10)
    ax.set_xlabel("Expert Index", fontsize=12)
    ax.set_ylabel("Layer Index", fontsize=12)
    ax.invert_yaxis() # Put Layer 0 at the bottom

    plt.tight_layout()
    
    # Save and show
    save_dir = os.path.join(root_folder, "results", "figures")
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Change the extension to .pdf
    save_path = os.path.join(save_dir, f"{model_name}_jailbreak_heatmap.pdf")
    
    # 2. Save as format='pdf' and remove dpi (PDFs are vector format)
    plt.savefig(save_path, format='pdf', bbox_inches="tight")
    
    print(f"Plot saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()