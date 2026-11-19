from __future__ import annotations

import logging
from types import ModuleType

import torch

logger = logging.getLogger("vllm_kunlun")


def detect_out_major(gate_up, down, hidden: int, inter: int) -> bool | None:
    """Detect the on-disk axis order from shape: True = out-major (Qwen3 family),
    False = in-major (GPT-OSS family).
    """
    if gate_up is not None:
        last = gate_up.lora_a.shape[-1]  # in-major: hidden; out-major: 2*inter
        if last == hidden and last != 2 * inter:
            return False
        if last == 2 * inter and last != hidden:
            return True
    if down is not None:
        last = down.lora_a.shape[-1]  # in-major: inter; out-major: hidden
        if last == inter and last != hidden:
            return False
        if last == hidden and last != inter:
            return True
    return None


def peft_3d_pair_to_stacked(
    lora_weights, num_experts: int, out_major: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    a_flat, b_flat = lora_weights.lora_a, lora_weights.lora_b
    if out_major:
        # (in, r*E) -> (in, r, E) -> (E, r, in)
        a = b_flat.reshape(b_flat.shape[0], -1, num_experts).permute(2, 1, 0)
        # (r*E, out) -> (E, r, out) -> (E, out, r)
        b = a_flat.reshape(num_experts, -1, a_flat.shape[-1]).permute(0, 2, 1)
        scaling = getattr(lora_weights, "_kunlun_preopt_scaling", 1.0)
        if scaling != 1.0:
            a = a / scaling
            b = b * scaling
    else:
        # Upstream original: lora_A is shrink, lora_B is expand.
        a = a_flat.reshape(num_experts, -1, a_flat.shape[-1])
        b = b_flat.reshape(b_flat.shape[0], -1, num_experts).permute(2, 0, 1)
    return a, b


def _convert_3d_to_2d_moe_lora(self, lora_model, module, module_name: str) -> None:
    """Alternative to the upstream method of the same name: adds axis-order
    normalization and zero-fill for the missing side."""
    gate_up = self._get_lora_layer_weights(lora_model, module_name + ".base_layer")
    down = self._get_lora_layer_weights(lora_model, module_name)
    if gate_up is None and down is None:
        return

    hidden = module.hidden_size
    inter = module.intermediate_size_per_partition * module.tp_size
    local_num_experts = module.local_num_experts
    global_num_experts = module.global_num_experts
    expert_start = module.ep_rank * local_num_experts
    expert_end = expert_start + local_num_experts

    out_major = detect_out_major(gate_up, down, hidden, inter)
    if out_major is None:
        logger.warning(
            "[KunlunPlugin] %s: cannot determine the 3D adapter axis order from shape "
            "(hidden=%d, inter=%d), treating it as upstream in-major; "
            "if the output degrades to the base model, the on-disk layout is out-major.",
            module_name,
            hidden,
            inter,
        )
        out_major = False

    def normalize(lora_weights):
        a, b = peft_3d_pair_to_stacked(lora_weights, global_num_experts, out_major)
        return (
            a[expert_start:expert_end].contiguous(),
            b[expert_start:expert_end].contiguous(),
        )

    carrier = down if down is not None else gate_up
    rank = carrier.rank
    zeros_kwargs = {"dtype": carrier.lora_a.dtype, "device": carrier.lora_a.device}

    if gate_up is not None:
        gate_up_a, gate_up_b = normalize(gate_up)
        intermediate_x2 = gate_up_b.shape[1]
        if intermediate_x2 != 2 * inter:
            raise ValueError(
                f"{module_name}: gate_up lora_B output dim should be 2*intermediate="
                f"{2 * inter}, got {intermediate_x2} (out_major={out_major})."
            )
        # GPT-OSS interleaves gate/up on the output dim; other 3D checkpoints concat them (gate first).
        if self.model.config.architectures[0] == "GptOssForCausalLM":
            w1_b = gate_up_b[:, ::2, :].contiguous()
            w3_b = gate_up_b[:, 1::2, :].contiguous()
        else:
            w1_b = gate_up_b[:, :inter, :].contiguous()
            w3_b = gate_up_b[:, inter:, :].contiguous()
    else:
        # Implement the upstream FIXME: treat the missing side as a zero delta, with
        # semantics matching the initial value of reset_lora.
        logger.warning(
            "[KunlunPlugin] %s: 3D adapter is missing gate_up (no .base_layer key), "
            "loading with zero delta; check whether mlp.experts.gate_up_proj was written "
            "into --target_parameters during training.",
            module_name,
        )
        gate_up_a = torch.zeros(local_num_experts, rank, hidden, **zeros_kwargs)
        w1_b = torch.zeros(local_num_experts, inter, rank, **zeros_kwargs)
        w3_b = torch.zeros(local_num_experts, inter, rank, **zeros_kwargs)

    if down is not None:
        down_a, down_b = normalize(down)
    else:
        logger.warning(
            "[KunlunPlugin] %s: 3D adapter is missing down_proj, loading with zero delta.",
            module_name,
        )
        down_a = torch.zeros(local_num_experts, rank, inter, **zeros_kwargs)
        down_b = torch.zeros(local_num_experts, hidden, rank, **zeros_kwargs)

    # A wrong axis order makes set_lora do a partial write without raising, so hard-check it here.
    for name, got, want in (
        ("w1/w3 lora_a", gate_up_a.shape, (local_num_experts, rank, hidden)),
        ("w1 lora_b", w1_b.shape, (local_num_experts, inter, rank)),
        ("w3 lora_b", w3_b.shape, (local_num_experts, inter, rank)),
        ("w2 lora_a", down_a.shape, (local_num_experts, rank, inter)),
        ("w2 lora_b", down_b.shape, (local_num_experts, hidden, rank)),
    ):
        if tuple(got) != want:
            raise ValueError(
                f"{module_name}: {name} shape {tuple(got)} != expected {want} "
                f"(out_major={out_major}, rank={rank})."
            )

    # w1 and w3 share the same shrink: PEFT's fused parameter has only one A side, and
    # set_lora copies it into two separate buffers, so the sharing here only saves CPU
    # memory and does not affect the numerics.
    carrier.lora_a = [gate_up_a, down_a, gate_up_a]
    carrier.lora_b = [w1_b, down_b, w3_b]
    # Keep only the wrapper key: activate_adapter looks up by module_name, and keeping
    # base_layer would make pin_memory copy another one.
    lora_model.loras[module_name] = carrier
    lora_model.loras.pop(module_name + ".base_layer", None)


def _record_preopt_scaling(lora_weights_module: ModuleType) -> None:
    """Make ``LoRALayerWeights.optimize()`` record the scaling before it is folded into lora_b.

    ``add_adapter`` runs in the order "optimize first, then convert to MoE", while the
    out-major on-disk ``lora_B`` is shrink, so the factor was folded onto the wrong side.
    This only adds a field and does not change optimize's behavior.
    """
    cls = getattr(lora_weights_module, "LoRALayerWeights", None)
    if cls is None or getattr(cls, "_kunlun_records_scaling", False):
        return
    original = cls.optimize

    def optimize(self):
        if self.scaling != 1:
            self._kunlun_preopt_scaling = float(self.scaling)
        return original(self)

    cls.optimize = optimize
    cls._kunlun_records_scaling = True


def patch_applied(module: ModuleType) -> bool:
    """Return whether ``LoRAModelManager`` is absent or already patched."""
    cls = getattr(module, "LoRAModelManager", None)
    return cls is None or getattr(module, "_kunlun_3d_moe_lora_patched", False)


def apply_patch(module: ModuleType) -> None:
    """Replace ``LoRAModelManager._convert_3d_to_2d_moe_lora``.

    Applied unconditionally: the in-major branch is equivalent to the upstream
    body, and the method only runs when ``--enable-mixed-moe-lora-format`` is on
    and the request declares ``is_3d_lora_weight=True``.
    """
    cls = getattr(module, "LoRAModelManager", None)
    if cls is None:
        return
    module._kunlun_3d_moe_lora_patched = True
    if not hasattr(cls, "_convert_3d_to_2d_moe_lora"):
        return
    cls._convert_3d_to_2d_moe_lora = _convert_3d_to_2d_moe_lora

    import vllm.lora.lora_weights as lora_weights_module

    _record_preopt_scaling(lora_weights_module)
    logger.info(
        "[KunlunPlugin] patched _convert_3d_to_2d_moe_lora "
        "(PEFT 3D adapter: axis-order detection + missing-side zeros)"
    )
