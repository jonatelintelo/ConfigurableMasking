import data.data_utils as data_utils
import moe_model_files.compute_graph_patcher as compute_graph_patcher

import torch
import inspect
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
from tqdm import tqdm
import types


def load_model(full_model_name):
    model_name = full_model_name.split("/")[1]

    if model_name in ["Phi-3.5-MoE-instruct", "Hunyuan-A13B-Instruct"]:
        model = AutoModelForCausalLM.from_pretrained(
            full_model_name,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",
            trust_remote_code=False,
        ).eval()
    elif model_name in ["gpt-oss-20b", "pangu-pro-moe-model"]:
        model = AutoModelForCausalLM.from_pretrained(
            full_model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            full_model_name,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        ).eval()

    if model_name in ["deepseek-moe-16b-chat", "pangu-pro-moe-model"]:
        from transformers.cache_utils import DynamicCache

        # 1. Fix 'seen_tokens' (redirects to get_seq_length)
        if not hasattr(DynamicCache, "seen_tokens"):

            @property
            def seen_tokens(self):
                return self.get_seq_length()

            DynamicCache.seen_tokens = seen_tokens

        # 2. Fix 'get_max_length' (returns None because DynamicCache grows indefinitely)
        if not hasattr(DynamicCache, "get_max_length"):

            def get_max_length(self):
                return None

            DynamicCache.get_max_length = get_max_length

        # 3. Fix 'get_usable_length' (handles both single and double argument calls)
        if not hasattr(DynamicCache, "get_usable_length"):

            def get_usable_length(self, input_seq_len, layer_idx=0):
                # We ignore input_seq_len because DynamicCache doesn't have a fixed window
                # We just return the current length of the cache for that layer
                return self.get_seq_length(layer_idx)

            DynamicCache.get_usable_length = get_usable_length
        print("\nDynamicCache successfully patched for compatibility with deepseek-moe-16b-chat and pangu-pro-moe-model.")
        # --- END PATCH ---

    if model_name == "deepseek-moe-16b-chat":
        for layer in model.model.layers:
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
                layer.mlp.gate.forward = types.MethodType(compute_graph_patcher.deepseek_moe_gate_forward, layer.mlp.gate)
    elif model_name == "Qwen1.5-MoE-A2.7B-Chat":
        for layer in model.model.layers:
            if hasattr(layer, "mlp"):
                layer.mlp.forward = types.MethodType(compute_graph_patcher.qwen1_5_moe_forward, layer.mlp)
    elif model_name == "gpt-oss-20b":
        for layer in model.model.layers:
            if hasattr(layer, "mlp"):
                layer.mlp.forward = types.MethodType(compute_graph_patcher.GptOssMLP_forward, layer.mlp)
                if hasattr(layer.mlp, "router"):
                    layer.mlp.router.forward = types.MethodType(compute_graph_patcher.GptOssTopKRouter_forward, layer.mlp.router)
                # Replace experts
                if hasattr(layer.mlp, "experts"):
                    old_experts = layer.mlp.experts

                    # Create new experts with per-expert gate-up layers
                    new_experts = compute_graph_patcher.GptOssExperts(model.config)
                    new_experts = new_experts.to(old_experts.down_proj.device)
                    new_experts = new_experts.to(old_experts.down_proj.dtype)

                    # Copy weights for down projection
                    new_experts.down_proj.data.copy_(old_experts.down_proj.data)
                    new_experts.down_proj_bias.data.copy_(old_experts.down_proj_bias.data)

                    # Copy weights for each expert's gate-up layer
                    for i in range(new_experts.num_experts):
                        new_experts.gate_up_layers[i].gate_up_proj.data.copy_(old_experts.gate_up_proj.data[i])
                        new_experts.gate_up_layers[i].gate_up_proj_bias.data.copy_(old_experts.gate_up_proj_bias.data[i])

                    # Replace the experts module
                    layer.mlp.experts = new_experts
        print("Patching finished")

    tokenizer = AutoTokenizer.from_pretrained(full_model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"

    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        # We can set pad_token as the eos_token or add a new one
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_output(model, model_name, tokenizer, prompts, batch_size):
    if model_name in [
        "gpt-oss-20b",
        "pangu-pro-moe-model",
        "Kimi-VL-A3B-Thinking",
        "Hunyuan-A13B-Instruct",
        "Kimi-VL-A3B-Thinking-2506",
    ]:
        max_new_tokens = 1024  # We give thinking models more token budget
    else:
        max_new_tokens = 128

    all_outputs = []
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    model.eval()
    with torch.no_grad():
        for batch_prompts in tqdm(data_utils.batchify(prompts, batch_size), total=total_batches):
            input_tokens = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

            if "token_type_ids" in input_tokens:
                forward_args = inspect.signature(model.forward).parameters
                if "token_type_ids" not in forward_args:
                    input_tokens.pop("token_type_ids")

            input_ids = input_tokens["input_ids"]

            output_ids = model.generate(
                **input_tokens,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                max_length=max_new_tokens,
            )

            generated_tokens = [output[ids.shape[-1] :] for ids, output in zip(input_ids, output_ids["sequences"])]

            responses = [tokenizer.decode(generated_tokens[i], skip_special_tokens=True).strip() for i in range(len(generated_tokens))]

            all_outputs.extend(responses)

    return all_outputs


def moderate(model, tokenizer, prompt):
    input_ids = tokenizer.apply_chat_template(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)


def register_activation_hooks(model_name, model, k, gate_name):
    hook_handles = []
    top_k_expert_indices = defaultdict(list)

    for layer_name, module in model.named_modules():
        if layer_name.lower().endswith(gate_name):
            hook_fn = get_activation_hook(model_name, layer_name, top_k_expert_indices, k)
            handle = module.register_forward_hook(hook_fn)
            hook_handles.append(handle)

    return hook_handles, top_k_expert_indices


def get_activation_hook(model_name, layer_name, top_k_expert_indices, k):
    print(f"Activation hook on layer: '{layer_name}'")

    def activation_hook(module, input, output):
        if model_name in [
            "Qwen3-30B-A3B-Instruct-2507",
            "Phi-3.5-MoE-instruct",
            "Mixtral-8x7B-Instruct-v0.1",
            "gpt-oss-20b",
            "Qwen1.5-MoE-A2.7B-Chat",
            "Hunyuan-A13B-Instruct",
        ]:
            top_k_expert_indices[layer_name].append(torch.topk(output, k=k, dim=-1, sorted=False).indices.cpu())  # (nr_tokens, topk_experts)
        elif model_name == "deepseek-moe-16b-chat":
            top_k_expert_indices[layer_name].append(output[0])
        elif model_name == "pangu-pro-moe-model":
            num_groups = 8
            experts_per_group = 8
            routing_weights, selected_experts = torch.max(output.view(output.shape[0], num_groups, -1), dim=-1)
            bias = torch.arange(
                0,
                64,
                experts_per_group,
                device=routing_weights.device,
                dtype=torch.int64,
            ).unsqueeze(0)
            topk_indices = selected_experts + bias
            top_k_expert_indices[layer_name].append(topk_indices)
        else:
            print("---------------------------------------------------------------")
            print(input)
            print("---------------------------------------------------------------")
            print(output)
            print("---------------------------------------------------------------")
            print(output.shape)
            print("---------------------------------------------------------------")
            raise ValueError(f"Activation hook for {model_name} is not implemented.")

    return activation_hook


def register_pruning_hooks_candidates(model_name, model, candidates):
    hook_handles = []

    for candidate in candidates:  # Start pruning every candidate
        for layer_name, module in model.named_modules():  # Loop over all layers
            if layer_name.lower().endswith(candidate[0]):  # Find the layer in the model which matches with current candidate
                hook_fn = get_pruning_hook_candidates(model_name, layer_name, candidate[1])
                handle = module.register_forward_hook(hook_fn)
                hook_handles.append(handle)
                break  # break because we only need to register on this layer once

    return hook_handles


def get_pruning_hook_candidates(model_name, layer_name, expert_index):
    # print(f"Pruning hook on layer: '{layer_name}' and expert '{expert_index}'")

    def prune_hook(module, input, output):
        if model_name in [
            "Qwen3-30B-A3B-Instruct-2507",
            "Phi-3.5-MoE-instruct",
            "Mixtral-8x7B-Instruct-v0.1",
            "gpt-oss-20b",
            "Qwen1.5-MoE-A2.7B-Chat",
            "Hunyuan-A13B-Instruct",
            "pangu-pro-moe-model",
        ]:
            pruned_output = output.clone()
            pruned_output[..., expert_index] = float("-inf")
            return pruned_output
        elif model_name == "deepseek-moe-16b-chat":
            pruned_output = output[2]
            pruned_output[..., expert_index] = float("-inf")
            scores = pruned_output.softmax(dim=-1)
            topk_weight, topk_indices = torch.topk(scores, k=6, dim=-1, sorted=False)  # [batch_size * seq_len, topk]
            return topk_indices, topk_weight, pruned_output
        else:
            print("---------------------------------------------------------------")
            print(input)
            print("---------------------------------------------------------------")
            print(output)
            print("---------------------------------------------------------------")
            print(output.shape)
            print("---------------------------------------------------------------")
            raise ValueError(f"Pruning hook for {model_name} is not implemented.")

    return prune_hook


def register_global_pruning_hooks(model_name, model, gate_name, experts):
    hook_handles = []

    for layer_name, module in model.named_modules():
        if layer_name.lower().endswith(gate_name):
            hook_fn = get_global_pruning_hook(model_name, layer_name, experts)
            handle = module.register_forward_hook(hook_fn)
            hook_handles.append(handle)

    return hook_handles


def get_global_pruning_hook(model_name, layer_name, experts):
    # print(f"Pruning hook on layer: '{layer_name}' and expert '{expert_index}'")

    def global_prune_hook(module, input, output):
        if model_name in [
            "Qwen3-30B-A3B-Instruct-2507",
            "Phi-3.5-MoE-instruct",
            "Mixtral-8x7B-Instruct-v0.1",
            "gpt-oss-20b",
            "Qwen1.5-MoE-A2.7B-Chat",
            "Hunyuan-A13B-Instruct",
            "pangu-pro-moe-model",
        ]:
            pruned_output = output.clone()
            pruned_output[..., experts] = float("-inf")
            return pruned_output
        elif model_name == "deepseek-moe-16b-chat":
            pruned_output = output[2]
            pruned_output[..., experts] = float("-inf")
            scores = pruned_output.softmax(dim=-1)
            topk_weight, topk_indices = torch.topk(scores, k=6, dim=-1, sorted=False)  # [batch_size * seq_len, topk]
            return topk_indices, topk_weight, pruned_output
        else:
            print("---------------------------------------------------------------")
            print(input)
            print("---------------------------------------------------------------")
            print(output)
            print("---------------------------------------------------------------")
            print(output.shape)
            print("---------------------------------------------------------------")
            raise ValueError(f"Pruning hook for {model_name} is not implemented.")

    return global_prune_hook
