# SPDX-License-Identifier: Apache-2.0
"""MRV2 DFlash2 proposal capture using the upstream graph scheduler."""

import inspect
from copy import copy

import kunlun_ops
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from vllm.v1.worker.gpu.input_batch import InputBatch

logger = init_logger(__name__)


class DFlash2CudaGraphManager(CudaGraphManager):
    """Capture context KV, query forward and greedy selection together.

    Dispatch uses upstream request padding. Padded queries attend only to the
    reserved null KV block, and their slots never write the live cache.
    """

    def __init__(self, speculator, target_mode):
        self.speculator = speculator
        reasons = self._disabled_reasons(target_mode)
        mode = CUDAGraphMode.NONE if reasons else CUDAGraphMode.FULL_DECODE_ONLY
        config = speculator.vllm_config
        if config.speculative_config.uses_dynamic_speculative_decoding():
            # Dynamic widths apply to target verification. The draft always
            # emits full query blocks; keep its graph descriptors fixed-width.
            config = copy(config)
            config.speculative_config = copy(config.speculative_config)
            config.speculative_config.num_speculative_tokens_per_batch_size = None
        super().__init__(
            config,
            speculator.device,
            mode,
            decode_query_len=speculator.num_query_per_req,
        )
        self.attn_states = {}
        self.attention_buffers = {}
        self._replay_logged = False
        self._reported_fallbacks = set()
        if not reasons and not self.needs_capture():
            reasons.append("no_valid_capture_sizes")
        logger.info(
            "DFlash2 proposal graph configuration: target=%s draft=%s TP=%d "
            "target_enforce_eager=%s draft_enforce_eager=%s reasons=%s",
            target_mode,
            mode if not reasons else CUDAGraphMode.NONE,
            self.tp_size,
            speculator.vllm_config.model_config.enforce_eager,
            speculator.speculative_config.enforce_eager,
            ",".join(reasons) or "enabled",
        )
        if reasons == ["no_valid_capture_sizes"]:
            self._disable_capture(reasons[0])
        if not self.needs_capture():
            speculator._aux_staging = None

    def _disabled_reasons(self, target_mode):
        spec = self.speculator
        checks = (
            (target_mode.decode_mode() != CUDAGraphMode.FULL, "target_decode_not_full"),
            (spec.vllm_config.model_config.enforce_eager, "target_enforce_eager"),
            (spec.speculative_config.enforce_eager, "draft_enforce_eager"),
            (spec.dflash_causal, "causal_draft"),
        )
        return [reason for disabled, reason in checks if disabled]

    def _disable_capture(self, reason):
        self.cudagraph_mode = CUDAGraphMode.NONE
        self._capture_descs.clear()
        self._candidates.clear()
        self._graphs_captured = False
        logger.warning(
            "DFlash2 proposal graph disabled: %s; draft runs eagerly", reason
        )

    def validate_capture(self):
        # MRV2 initializes this manager before set_attn. Inspect actual draft
        # builders/layers here, after set_attn and before touching any KV data.
        if not self.needs_capture():
            return False
        from vllm_kunlun.v1.attention.backends.kunlun_attn import KunlunAttentionBackend

        spec = self.speculator
        layers = [layer.self_attn.attn for layer in spec.model.model.layers]
        groups = [group for bucket in spec.attn_groups for group in bucket]
        if not groups or not layers:
            self._disable_capture("draft_attention_unavailable")
            return False
        for group in groups:
            support = group.get_metadata_builder(0).get_cudagraph_support(
                spec.vllm_config, group.kv_cache_spec
            )
            if support.value < AttentionCGSupport.UNIFORM_BATCH.value:
                self._disable_capture(
                    f"draft_attention_not_full: {group.backend.__name__} ({support.name})"
                )
                return False
            if not issubclass(group.backend, KunlunAttentionBackend):
                self._disable_capture(
                    f"unsupported_draft_backend: {group.backend.__name__}"
                )
                return False
        for layer in layers:
            if layer.query_quant is not None or layer.calculate_kv_scales:
                self._disable_capture(
                    f"query_quant_or_dynamic_kv_scales: {layer.layer_name}"
                )
                return False
        op = getattr(kunlun_ops, "prefill_attention", None)
        # torch.library.custom_op returns a CustomOpDef with a generic __call__;
        # inspect its original Python function, including functools.wraps chains.
        fn = getattr(op, "_init_fn", op)
        location = getattr(kunlun_ops, "__file__", "<unknown>")
        try:
            parameters = inspect.signature(fn).parameters
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Cannot verify DFlash2 graph prefill_attention signature; kunlun_ops={location}"
            ) from exc
        required = ("max_query_len", "max_context_len")
        missing = [
            name
            for name in required
            if name not in parameters
            or parameters[name].kind
            not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ]
        if missing:
            raise RuntimeError(
                f"DFlash2 FULL graph requires prefill_attention keyword arguments {missing}; "
                f"kunlun_ops={location}. Install matching Python and native operator binaries."
            )
        return True

    def dispatch(self, num_reqs, num_tokens, uniform_token_count, num_active_loras=0):
        desc = super().dispatch(
            num_reqs, num_tokens, uniform_token_count, num_active_loras
        )
        if (
            desc.cg_mode != CUDAGraphMode.FULL
            and self.cudagraph_mode != CUDAGraphMode.NONE
        ):
            reason = (
                "no_matching_capture_size"
                if self._graphs_captured
                else "graphs_not_captured"
            )
            key = (reason, num_reqs, num_tokens)
            if key not in self._reported_fallbacks:
                self._reported_fallbacks.add(key)
                logger.info(
                    "DFlash2 proposal graph fallback: reason=%s requests=%d query_tokens=%d",
                    reason,
                    num_reqs,
                    num_tokens,
                )
        return desc

    def run_fullgraph(self, desc):
        super().run_fullgraph(desc)
        if not self._replay_logged:
            logger.info(
                "DFlash2 proposal FULL graph replay activated: padded_requests=%d query_tokens=%d",
                desc.num_reqs,
                desc.num_tokens,
            )
            self._replay_logged = True

    def capture(self):
        spec = self.speculator
        if not self.validate_capture():
            return
        layers = [layer.self_attn.attn for layer in spec.model.model.layers]
        # Block zero is reserved. Cold capture may read it but must never write
        # live KV blocks, including during the upstream warmup pass.
        for layer in layers:
            layer.kv_cache[:, 0].zero_()
        spec.hidden_states.zero_()
        if spec._aux_staging is not None:
            spec._aux_staging.zero_()
        spec.context_positions.zero_()
        spec._context_slot_mappings.fill_(-1)
        spec._query_slots.fill_(-1)
        for table in spec.block_tables.input_block_tables:
            table.zero_()

        def create_forward(desc, warmup):
            count, batch = desc.num_tokens, desc.num_reqs
            InputBatch.make_dummy(batch, count, spec.input_buffers)
            samples = (
                spec._anchor_indices[:batch, None]
                + torch.arange(1, spec.num_query_per_req, device=spec.device)
            ).flatten()
            spec.sample_indices[: samples.numel()].copy_(samples)
            state = spec._build_query_attn_state(batch)
            metadata, slots = state
            # Use fixed host launch bounds for these prefill-only batches.
            # Device LoDs remain exact and are recomputed in the recorded forward.
            for md in {id(md): md for md in metadata.values()}.values():
                md.use_static_lod_bounds = True
                md.max_query_len = spec.num_query_per_req
                md.max_kv_len = spec.max_model_len
                md.seq_lens_tensor_cpu = torch.full_like(
                    md.seq_lens_tensor_cpu, md.max_kv_len
                )
                md.kv_lod_cpu = (
                    torch.arange(batch + 1, dtype=torch.int32) * md.max_kv_len
                )
            buffers = {
                layer: (
                    torch.empty(
                        (count, layer.num_heads, layer.head_size_v),
                        dtype=spec.dtype,
                        device=spec.device,
                    ),
                    torch.empty(
                        (count, layer.num_heads, layer.head_size_v),
                        dtype=spec.dtype,
                        device=spec.device,
                    ),
                )
                for layer in layers
            }
            if not warmup:
                self.attention_buffers[desc] = buffers

            def forward(mode):
                for layer, tensors in buffers.items():
                    layer._kunlun_dflash_graph_buffers = tensors
                try:
                    spec._prepare_graph_forward(batch, metadata)
                    spec._generate_draft(batch, count, metadata, slots, None, mode)
                finally:
                    for layer in buffers:
                        del layer._kunlun_dflash_graph_buffers

            return forward, state

        self.attn_states = super().capture(
            create_forward, "Capturing DFlash2 proposals"
        )
        logger.info(
            "Captured %d DFlash2 proposal FULL graphs; request_sizes=%s aux_context_in_graph=%s",
            len(self.graphs),
            sorted(desc.num_reqs for desc in self.graphs),
            spec._aux_staging is not None,
        )
