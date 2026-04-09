import data.data_utils as data_utils
import moe_model_files.compute_graph_patcher as compute_graph_patcher

import torch
import inspect
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
from tqdm import tqdm
import types
import gc


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
    elif model_name in ["gpt-oss-20b"]:
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

    if model_name in ["deepseek-moe-16b-chat"]:
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
        print("\nDynamicCache successfully patched for compatibility with deepseek-moe-16b-chat.")
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
        "Hunyuan-A13B-Instruct",
    ]:
        max_new_tokens = 2048  # We give thinking models more token budget
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
            )

            generated_tokens = [output[ids.shape[-1] :] for ids, output in zip(input_ids, output_ids["sequences"])]

            responses = [tokenizer.decode(generated_tokens[i], skip_special_tokens=True).strip() for i in range(len(generated_tokens))]

            all_outputs.extend(responses)

    return all_outputs


def generate_output_with_yield(model, model_name, tokenizer, prompts, batch_size):
    if model_name in [
        "gpt-oss-20b",
        "Hunyuan-A13B-Instruct",
    ]:
        max_new_tokens = 2048  # We give thinking models more token budget
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
            )

            generated_tokens = [output[ids.shape[-1] :] for ids, output in zip(input_ids, output_ids["sequences"])]

            responses = [tokenizer.decode(generated_tokens[i], skip_special_tokens=True).strip() for i in range(len(generated_tokens))]

            # 1. YIELD instead of extending a list
            yield responses

            # 2. Force memory cleanup before the next batch loads
            del input_tokens, input_ids, output_ids, generated_tokens
            torch.cuda.empty_cache()
            gc.collect()


def moderate(model, tokenizer, prompt):
    input_ids = tokenizer.apply_chat_template(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
