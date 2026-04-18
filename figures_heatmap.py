import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

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
    root_folder = "./" 
    model_name = "deepseek-moe-16b-chat"  # Replace with the model you want to plot
    
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

    print("Generating plot...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
    
    # Shared settings for the heatmaps
    cmap = "viridis"  # "magma" or "Blues" also look great for frequency maps
    vmax = max(baseline_freq.max(), steered_freq.max()) # Ensure color scale matches exactly

    # Subplot 1: Baseline
    sns.heatmap(baseline_freq, ax=axes[0], cmap=cmap, vmin=0, vmax=vmax, 
                cbar_kws={'label': 'Activation Frequency'})
    axes[0].set_title(f"Baseline Expert Activation\n({model_name})", fontsize=14, pad=10)
    axes[0].set_xlabel("Expert Index", fontsize=12)
    axes[0].set_ylabel("Layer Index", fontsize=12)
    axes[0].invert_yaxis() # Put Layer 0 at the bottom

    # Subplot 2: Steered
    sns.heatmap(steered_freq, ax=axes[1], cmap=cmap, vmin=0, vmax=vmax,
                cbar_kws={'label': 'Activation Frequency'})
    axes[1].set_title(f"Steered Expert Activation\n({model_name})", fontsize=14, pad=10)
    axes[1].set_xlabel("Expert Index", fontsize=12)
    axes[1].invert_yaxis()

    plt.tight_layout()
    
    # Save and show
    save_dir = os.path.join(root_folder, "results", "figures")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_expert_frequency_comparison.png")
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()