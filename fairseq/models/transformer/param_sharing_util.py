from typing import List

import torch.nn as nn

from fairseq.distributed import fsdp_wrap
from fairseq.modules.checkpoint_activations import checkpoint_wrapper


def get_share_index(total_layers: int, share_layer_num: int, share_type: str = "sequence") -> List[int]:
    """
    Args:
        total_layers: N
        share_layer_num: M
        share_type: sequence, cycle, cycle_rev

    Returns:
        a list of indexes
        -   -1:     create a new layer
        -   else i: copy ith layer
    """
    if share_type not in ["sequence", "cycle", "cycle_rev"]:
        raise NotImplementedError(f"share_type: {share_type}")
    if share_layer_num <= 1 or share_layer_num >= total_layers:
        return [-1] * total_layers

    res = []
    if share_type == "sequence":
        last_new_layer_index = -1
        for i in range(total_layers):
            if i % share_layer_num == 0:
                res.append(-1)
                last_new_layer_index = i
            else:
                res.append(last_new_layer_index)
    else:
        unique_layer_num = total_layers // share_layer_num + 0 if total_layers % share_layer_num == 0 else 1
        for _ in range(unique_layer_num):
            res.append(-1)
        if share_type == "cycle":
            for i in range(total_layers - unique_layer_num):
                res.append(i % unique_layer_num)
        elif share_type == "cycle_rev":
            for i in range(total_layers - unique_layer_num * 2):
                res.append(i % unique_layer_num)
            base_layers = range(unique_layer_num)
            for i in range(total_layers - len(res)):
                res.append(base_layers[-((i % unique_layer_num) + 1)])

    return res


def copy_layer(base_layer: nn.Module, target_layer: nn.Module, cfg, decoder: bool = False) -> nn.Module:
    target_layer.self_attn.k_proj = base_layer.self_attn.k_proj
    target_layer.self_attn.v_proj = base_layer.self_attn.v_proj
    target_layer.self_attn.q_proj = base_layer.self_attn.q_proj
    if base_layer.self_attn.bias_k is not None:
        target_layer.self_attn.bias_k = base_layer.self_attn.bias_k
    if base_layer.self_attn.bias_v is not None:
        target_layer.self_attn.bias_v = base_layer.self_attn.bias_v
    target_layer.self_attn.out_proj = base_layer.self_attn.out_proj
    target_layer.fc1 = base_layer.fc1
    target_layer.fc2 = base_layer.fc2
    target_layer.self_attn_layer_norm = base_layer.self_attn_layer_norm
    target_layer.final_layer_norm = base_layer.final_layer_norm
    if decoder:
        target_layer.encoder_attn.k_proj = base_layer.encoder_attn.k_proj
        target_layer.encoder_attn.v_proj = base_layer.encoder_attn.v_proj
        target_layer.encoder_attn.q_proj = base_layer.encoder_attn.q_proj
        target_layer.encoder_attn.out_proj = base_layer.encoder_attn.out_proj
        if base_layer.encoder_attn.bias_k is not None:
            target_layer.encoder_attn.bias_k = base_layer.encoder_attn.bias_k
        if base_layer.self_attn.bias_v is not None:
            target_layer.encoder_attn.bias_v = base_layer.encoder_attn.bias_v
        target_layer.encoder_attn_layer_norm = base_layer.encoder_attn_layer_norm

    checkpoint = cfg.checkpoint_activations
    if checkpoint:
        offload_to_cpu = cfg.offload_activations
        target_layer = checkpoint_wrapper(target_layer, offload_to_cpu=offload_to_cpu)
    # if we are checkpointing, enforce that FSDP always wraps the
    # checkpointed layer, regardless of layer size
    min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
    target_layer = fsdp_wrap(target_layer, min_num_params=min_params_to_wrap)
    return target_layer


def copy_dp_attn(base_layer: nn.Module, target_layer: nn.Module) -> nn.Module:
    if base_layer.dp_attn is not None and target_layer.dp_attn is not None:
        target_layer.dp_attn.k_proj = base_layer.dp_attn.k_proj
        target_layer.dp_attn.v_proj = base_layer.dp_attn.v_proj
        target_layer.dp_attn.q_proj = base_layer.dp_attn.q_proj
        if base_layer.dp_attn.bias_k is not None:
            target_layer.dp_attn.bias_k = base_layer.dp_attn.bias_k
        if base_layer.dp_attn.bias_v is not None:
            target_layer.dp_attn.bias_v = base_layer.dp_attn.bias_v
        target_layer.dp_attn.out_proj = base_layer.dp_attn.out_proj
    target_layer.dp_attn_layer_norm = base_layer.dp_attn_layer_norm
    return target_layer


if __name__ == "__main__":
    n = 12
    m = 2
    t = "sequence"
    print(t, get_share_index(n, m, t), sep="\t")
    t = "cycle"
    print(t, get_share_index(n, m, t), sep="\t\t")
    t = "cycle_rev"
    print(t, get_share_index(n, m, t), sep="\t")
