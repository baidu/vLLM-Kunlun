# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MTP-first DMTP proposals with a fixed draft block for the Kunlun V2 runner."""

import json
from pathlib import Path

import kunlun_ops
import numpy as np
import torch
from safetensors import safe_open
from vllm.compilation.backends import set_model_tag
from vllm.config import set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_tp_group
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MultiTokenPredictor
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata, init_attn_backend
from vllm.v1.worker.gpu.cudagraph_utils import AttentionState, CudaGraphManager

from .dflash2.speculator import DFlash2Speculator

logger = init_logger(__name__)


def native_weights(folder):
    index = json.loads((folder / "model.safetensors.index.json").read_text())
    shards = {}
    for name, shard in index["weight_map"].items():
        if name.startswith("mtp."):
            shards.setdefault(shard, []).append(name)
    if not shards:
        raise ValueError(f"No native mtp.* weights found in {folder}")
    for shard, names in shards.items():
        with safe_open(folder / shard, framework="pt", device="cpu") as checkpoint:
            for name in names:
                yield name.removeprefix("mtp."), checkpoint.get_tensor(name)


def sample_candidates(scores, ids, temperature, coins):
    """Return the sampled local index, token and processed logits for its q row."""
    greedy = temperature == 0
    logits = scores.float() / temperature.clamp_min(1e-5)[:, None]
    probs = logits.softmax(dim=-1)
    draw = (coins[:, None] >= probs.cumsum(dim=-1)).sum(-1).clamp_max(ids.shape[-1] - 1)
    index = torch.where(greedy, scores.argmax(-1), draw)
    point_mass = torch.full_like(logits, float("-inf")).scatter_(1, index[:, None], 0.0)
    return (
        index,
        ids.gather(1, index[:, None]).squeeze(1),
        torch.where(greedy[:, None], point_mass, logits),
    )


def sample_tail(scores, ids, temperatures, coins, ranks):
    """Draw every possible edge with the same per-position coin, then walk one path."""
    n, steps, k = ids.shape
    scaled = scores.float() / temperatures.clamp_min(1e-5)[:, None, None, None]
    cdf = scaled.softmax(-1).cumsum(-1)
    draw = (coins[:, :, None, None] >= cdf).sum(-1).clamp_max(k - 1)
    greedy = temperatures == 0
    # For sampled rows each conditional distribution supplies one winner.
    # The Kunlun selector returns local ranks so we can recover the exact q
    # used at each realized predecessor, including in mixed-temperature batches.
    path_scores = torch.where(
        greedy[:, None, None, None], scores, ranks == draw[..., None]
    )
    chosen = kunlun_ops.dflash_greedy_select_path(
        ranks.expand(n, steps, k), path_scores
    )
    previous = torch.cat((torch.zeros_like(chosen[:, :1]), chosen[:, :-1]), dim=1)
    q = scaled.gather(2, previous[:, :, None, None].expand(n, steps, 1, k)).squeeze(2)
    point_mass = torch.full_like(q, float("-inf")).scatter_(2, chosen[..., None], 0.0)
    return ids.gather(2, chosen[..., None]).squeeze(2), torch.where(
        greedy[:, None, None], point_mass, q
    )


class NativeMTPGraphs(CudaGraphManager):
    """Fixed-width causal MTP update using the target verifier's rows."""

    def __init__(self, spec):
        self.verify_width = spec.num_speculative_steps + 1
        super().__init__(
            spec.vllm_config,
            spec.device,
            CUDAGraphMode.FULL_DECODE_ONLY,
            decode_query_len=self.verify_width,
        )
        self.spec = spec
        n, width = spec.max_num_reqs, self.verify_width
        self.ids = torch.zeros(n, width, device=spec.device, dtype=torch.int32)
        self.positions = torch.zeros_like(self.ids, dtype=torch.long)
        self.hidden = torch.zeros(
            n * width, spec.hidden_size, device=spec.device, dtype=spec.dtype
        )
        self.anchors = torch.zeros(n, device=spec.device, dtype=torch.int32)
        self.rejected = torch.zeros_like(self.anchors)
        self.lens = torch.full_like(self.anchors, width)
        self.group = next(
            i
            for i, groups in enumerate(spec.native_groups)
            if any(spec.native_name in group.layer_names for group in groups)
        )
        # Never write to live KV during warmup/capture. At replay only the
        # accepted prefix writes to private MTP KV; rejected rows get slot -1.
        self.slots = torch.full_like(
            spec.block_tables.slot_mappings[:, : n * width], -1
        )
        self.target_slots = torch.full_like(self.ids, -1, dtype=torch.long)
        # The recorded XFA kernels reference the per-shape LoD buffers.
        self.attn_states = self.capture(
            self.create_forward, "Capturing DMTP native MTP"
        )
        self.exact = {desc.num_reqs: desc for desc in self.graphs}
        self.replay_logged = False
        logger.info(
            "Captured native DMTP XPU graphs with %d verifier rows per request: %s",
            self.verify_width,
            sorted(self.exact),
        )

    def create_forward(self, desc, warmup):
        spec = self.spec
        n, width = desc.num_reqs, self.verify_width
        count = n * width
        starts = torch.arange(n + 1, dtype=torch.int32) * width
        # Build a prefill-shaped descriptor for the paged XFA kernel. Its
        # host bounds are fixed, its device lengths change at each replay.
        metadata = build_attn_metadata(
            spec.native_groups,
            n,
            count,
            starts.to(spec.device),
            starts,
            width,
            self.lens[:n],
            spec.max_model_len,
            [t[:n] for t in spec.block_tables.input_block_tables],
            self.slots[:, :count],
            spec.kv_cache_config,
            causal=False,
        )
        md = metadata[spec.native_name]
        md.causal = True
        md.use_static_lod_bounds = True
        md.max_kv_len = spec.max_model_len
        md.kv_lod_cpu = torch.arange(n + 1, dtype=torch.int32) * md.max_kv_len
        by_layer = {spec.native_name: md.slot_mapping}

        def forward(mode):
            last = width - 1 - self.rejected[:n].long()
            lane = spec.query_offsets[None, :width]
            slots = torch.where(lane <= last[:, None], self.target_slots[:n], -1)
            self.slots[self.group, :count].copy_(slots.flatten())
            tokens = torch.where(
                lane == last[:, None],
                self.anchors[:n, None],
                self.ids[:n].roll(-1, dims=1),
            )
            torch.cumsum(self.lens[:n], 0, out=md.kv_lod_xpu[1:])
            with set_forward_context(
                metadata,
                spec.vllm_config,
                num_tokens=count,
                slot_mapping=by_layer,
                skip_compiled=True,
            ):
                out = spec.native(
                    input_ids=tokens.flatten(),
                    positions=self.positions[:n].flatten() + 1,
                    hidden_states=self.hidden[:count],
                )
            spec.native_hidden[:n].copy_(
                out.view(n, width, -1)[spec.batch_rows[:n], last]
            )

        return forward, AttentionState(metadata, by_layer)

    def replay(self, batch, hidden, anchors, rejected):
        n, count = batch.num_reqs, batch.num_tokens
        desc = self.exact.get(n)
        if (
            desc is None
            or count != desc.num_tokens
            or not (batch.num_scheduled_tokens == self.verify_width).all()
        ):
            return False
        self.ids[:n].copy_(batch.input_ids[:count].view(n, -1))
        self.positions[:n].copy_(batch.positions[:count].view(n, -1))
        self.hidden[:count].copy_(hidden[:count])
        self.anchors[:n].copy_(anchors)
        self.rejected[:n].copy_(rejected[:n])
        # Keep the padded verifier length: causal attention starts at
        # seq_len - verify_width, matching eager's accepted_len - query_len.
        # This also holds for an eight-row prefill after a prefix cache hit.
        self.lens[:n].copy_(batch.seq_lens[:n])
        self.target_slots[:n].copy_(
            self.spec.block_tables.slot_mappings[self.group, :count].view(n, -1)
        )
        self.run_fullgraph(desc)
        if not self.replay_logged:
            logger.info("Native DMTP XPU graph replay active (requests=%d)", n)
            self.replay_logged = True
        return True


class DMTPSpeculator(DFlash2Speculator):
    _speculator_name = "DMTP"
    _draft_sample_methods = ("greedy", "probabilistic")
    _fixed_draft_query_len = 8

    @classmethod
    def _validate_config(cls, vllm_config):
        super()._validate_config(vllm_config)
        spec = vllm_config.speculative_config
        config = spec.draft_model_config.hf_config
        cfg = config.dflash_config
        if (
            not 3 <= spec.num_speculative_tokens < cls._fixed_draft_query_len
            or spec.uses_dynamic_speculative_decoding()
        ):
            raise ValueError("DMTP requires a fixed proposal length from 3 to 7")
        expected = {
            "input_layout": ["A", "T1"] + ["MASK"] * 6,
            "mtp_prefix": True,
            "mtp_t1_residual": True,
            "mtp_hidden_only": True,
            "mtp_prefix_injection": "add_to_t1_input_embedding",
            "mtp_t1_input_slot": 1,
            "mtp_hidden_index_relative_to_T1": -2,
            "block_size": 8,
            "mtp_proposal_token_count": 1,
            "dflash_proposal_token_count": 6,
            "block_attention": "bidirectional",
            "context_kv_boundary": "strictly_before_anchor",
            "hidden_norm_zero_centered_gamma": True,
            "conv_base_kernel_identity_folded": True,
            "swa_window_size": 4096,
            "draft_layer_context_windows": [4096] * 5,
        }
        if any(cfg.get(key) != value for key, value in expected.items()):
            raise ValueError("Unsupported DMTP checkpoint layout or attention contract")
        target = vllm_config.model_config.hf_text_config
        if (
            target.model_type != "qwen3_5_text"
            or target.mtp_num_hidden_layers != 1
            or target.hidden_size != config.hidden_size
            or target.num_hidden_layers != 64
            or config.head_dim != 256
            or config.partial_rotary_factor != 0.25
        ):
            raise ValueError(
                "Expected the Qwen3.6-27B target and its matching DMTP draft"
            )
        if vllm_config.cache_config.kv_sharing_fast_prefill:
            raise ValueError(
                "DMTP is incompatible with kv_sharing_fast_prefill: "
                "native MTP requires correct hidden states for every prompt token"
            )

    def __init__(self, vllm_config, device):
        super().__init__(vllm_config, device)
        n, steps = self.max_num_reqs, self.num_speculative_steps
        self.native_hidden = torch.zeros(
            n, self.hidden_size, dtype=self.dtype, device=device
        )
        self.proposal_temps = torch.zeros(n, device=device, dtype=torch.float32)
        self.coins = torch.zeros(n, steps, device=device, dtype=torch.float32)
        self.support_ids = torch.zeros(
            n, steps, self.selector_top_k, dtype=torch.long, device=device
        )
        self.support_logits = torch.empty_like(self.support_ids, dtype=torch.float32)
        self.native_graphs = None
        if self.draft_logits is not None:
            self.draft_logits.fill_(float("-inf"))
            self.previous_support = torch.zeros_like(self.support_ids)
        self.window = self.draft_model_config.hf_config.dflash_config["swa_window_size"]
        self.separate_query = False
        self.query_offsets = torch.arange(self.num_query_per_req, device=device)
        self.steps = self.query_offsets[:steps]
        self.batch_rows = torch.arange(n, device=device)
        self.ranks = torch.arange(self.selector_top_k, device=device)

    def load_draft_model(self, target_model, target_attn_layer_names):
        model = super().load_draft_model(target_model, target_attn_layer_names)
        vc = self.vllm_config
        with (
            set_current_vllm_config(vc),
            set_model_tag("dmtp_native"),
            set_default_torch_dtype(self.dtype),
        ):
            with self.device:
                native = Qwen3_5MultiTokenPredictor(
                    vllm_config=vc, prefix="dmtp_native"
                )
            # Share the current target embedding only after the private MTP
            # parameters have gone through their normal vLLM post-load hooks.
            del native.embed_tokens
            loaded = native.load_weights(native_weights(Path(vc.model_config.model)))
            missing = set(dict(native.named_parameters())) - loaded
            if missing:
                raise ValueError(f"Missing native MTP parameters: {sorted(missing)}")
            process_weights_after_loading(native, vc.model_config, self.device)
            native.embed_tokens = model.model.embed_tokens
        self.native = native.eval()
        self.native_name = native.layers[0].self_attn.attn.layer_name
        logger.info(
            "DMTP loaded native MTP from %s with %d draft slots and %d proposals",
            vc.model_config.model,
            self.num_query_per_req,
            self.num_speculative_steps,
        )
        return model

    def set_attn(self, model_state, kv_cache_config, block_tables):
        # Both models have their own allocated KV. Their metadata differs:
        # native MTP appends a ragged causal prefix, DFlash2 queries a full block.
        self.draft_attn_layer_names.discard(self.native_name)
        super().set_attn(model_state, kv_cache_config, block_tables)
        self.native_groups, _, _ = init_attn_backend(
            kv_cache_config,
            self.vllm_config,
            self.device,
            active_layer_names={self.native_name},
        )

    def capture(self, attn_states=None):
        super().capture(attn_states)
        from vllm_kunlun.v1.attention.backends.kunlun_attn import KunlunAttentionImpl

        manager = self.query_cudagraph_manager
        impl = self.native.layers[0].self_attn.attn.impl
        if (
            manager is not None
            and manager.graphs
            and isinstance(impl, KunlunAttentionImpl)
        ):
            self.native_graphs = NativeMTPGraphs(self)

    def _can_replay_query(self):
        return not self.separate_query

    def _stage_context_window(self, batch, rejected):
        # The scheduler allocates lookahead for DFlash. Its host length is
        # an upper bound even with outstanding rejections. Check each real
        # request's *allocated* kernel pages; unused table entries can be stale.
        n = batch.num_reqs
        upper = batch.seq_lens_cpu_upper_bound[:n].numpy()
        required = upper + self.num_query_per_req
        if (required <= min(self.window, self.max_model_len)).all() and all(
            (
                required
                <= self.block_tables.num_blocks.np[gid, batch.idx_mapping_np]
                * self.block_tables.kernel_block_sizes[gid]
            ).all()
            for gid in self.draft_kv_cache_group_ids
        ):
            return
        anchors = batch.seq_lens[:n] - rejected[:n]
        # Resolve ambiguous page ownership and the anchor-relative window
        # against the exact accepted lengths on the exceptional path.
        lengths = anchors.cpu().tolist()
        page_size = min(
            self.block_tables.kernel_block_sizes[g]
            for g in self.draft_kv_cache_group_ids
        )
        self.separate_query = any(
            n > self.window
            or n + self.num_query_per_req > self.max_model_len
            or (n - 1) // page_size != (n + self.num_query_per_req - 1) // page_size
            for n in lengths
        )
        page_indices = {}
        if self.separate_query:
            self.true_query_positions = anchors[:, None] + self.query_offsets
            for gid in self.draft_kv_cache_group_ids:
                table = self.block_tables.input_block_tables[gid]
                size = self.block_tables.kernel_block_sizes[gid]
                per_request = []
                for i, n in enumerate(lengths):
                    pos = torch.arange(max(0, n - self.window), n, device=self.device)
                    per_request.append((table[i, pos // size].long(), pos % size))
                page_indices[gid] = per_request
        for i, layer in enumerate(self.model.model.layers):
            group_index = self._layer_group_idx[i]
            layer.self_attn.context_pages = page_indices.get(
                self.draft_kv_cache_group_ids[group_index]
            )

    def _update_native(self, batch, hidden, anchors, rejected, dummy_run):
        n = batch.num_reqs
        if not dummy_run:
            self._stage_context_window(batch, rejected)
            if self.native_graphs is not None and self.native_graphs.replay(
                batch, hidden, anchors, rejected
            ):
                return
        offsets = batch.query_start_loc_np[: n + 1]
        lengths = np.diff(offsets) - rejected[:n].cpu().numpy()
        if np.any(lengths < 1):
            raise ValueError(
                "DMTP native MTP requires at least one verified target row per request"
            )
        starts = torch.from_numpy(
            np.concatenate(([0], lengths.cumsum())).astype(np.int32)
        )
        rows = np.concatenate(
            [
                np.arange(start, start + length)
                for start, length in zip(offsets, lengths)
            ]
        )
        rows = torch.from_numpy(rows).to(self.device)
        ends = (starts[1:] - 1).to(device=self.device, dtype=torch.long)
        # MTP reads E(token[p+1]) and the final normalized target hidden at p.
        # Its causal KV lives in a private layer at logical slot p; its RoPE uses p+1.
        tokens = batch.input_ids[(rows + 1).clamp_max(batch.num_tokens - 1)]
        tokens[ends] = anchors.to(tokens.dtype)
        positions = batch.positions[rows] + 1
        count = rows.numel()
        metadata, by_layer = None, None
        if not dummy_run:
            # prepare_attn() has gathered these tables in target batch order.
            tables = [t[:n] for t in self.block_tables.input_block_tables]
            slots = self.block_tables.slot_mappings.index_select(1, rows)
            metadata = build_attn_metadata(
                self.native_groups,
                n,
                count,
                starts.to(self.device),
                starts,
                int(lengths.max()),
                batch.seq_lens[:n] - rejected[:n],
                self.max_model_len,
                tables,
                slots,
                self.kv_cache_config,
                causal=True,
            )
            by_layer = {self.native_name: metadata[self.native_name].slot_mapping}
        with set_forward_context(
            metadata,
            self.vllm_config,
            num_tokens=count,
            slot_mapping=by_layer,
            skip_compiled=True,
        ):
            result = self.native(
                input_ids=tokens,
                positions=positions,
                hidden_states=hidden[rows],
            )
        self.native_hidden[:n].copy_(result[ends])

    def propose(
        self,
        input_batch,
        attn_metadata,
        slot_mappings,
        last_hidden_states,
        aux_hidden_states,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
        num_tokens_across_dp=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        mm_inputs=None,
        is_profile=False,
    ):
        n = input_batch.num_reqs
        if n == 0:
            return self.draft_tokens[:0]
        slots = input_batch.idx_mapping[:n].long()
        anchors = torch.where(
            num_sampled[:n] > 0,
            last_sampled.reshape(-1)[slots],
            next_prefill_tokens.reshape(-1)[slots],
        )
        self._update_native(
            input_batch, last_hidden_states, anchors, num_rejected, dummy_run
        )
        if self.draft_logits is not None:
            self.proposal_temps[:n].copy_(temperature[slots])
            coins = torch.rand_like(self.coins[:n])
            if get_tp_group().world_size > 1:
                get_tp_group().broadcast(coins, src=0)
            self.coins[:n].copy_(coins)
        try:
            result = super().propose(
                input_batch,
                attn_metadata,
                slot_mappings,
                last_hidden_states,
                aux_hidden_states,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                temperature,
                seeds,
                num_tokens_across_dp,
                dummy_run,
                skip_attn_for_dummy_run,
                mm_inputs,
                is_profile,
            )
        finally:
            if self.separate_query:
                for layer in self.model.model.layers:
                    layer.self_attn.context_pages = None
                self.separate_query = False
        if self.draft_logits is not None and not dummy_run:
            # The verifier uses persistent request slots. Only the previous
            # 16 candidates can have finite logits. Clear them first so that
            # overlapping supports and recycled request slots stay correct.
            rows, steps = slots[:, None, None], self.steps[None, :, None]
            self.draft_logits[rows, steps, self.previous_support[slots]] = float("-inf")
            self.draft_logits[rows, steps, self.support_ids[:n]] = self.support_logits[
                :n
            ]
            self.previous_support[slots] = self.support_ids[:n]
        return result

    def _generate_draft(
        self,
        num_reqs,
        num_tokens_padded,
        attn_metadata,
        slot_mappings,
        num_tokens_across_dp,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    ):
        n = num_reqs
        if self.separate_query:
            self.input_buffers.positions[:num_tokens_padded].copy_(
                self.true_query_positions.flatten()
            )
        width = self.num_query_per_req
        tokens = self.input_buffers.input_ids[:num_tokens_padded].view(n, width)
        temperatures = self.proposal_temps[:n]
        ids, raw = self.model.compute_candidates(
            self.native_hidden[:n], apply_draft_transform=False
        )
        _, t1, processed = sample_candidates(raw, ids, temperatures, self.coins[:n, 0])
        tokens[:, 1].copy_(t1)
        self.draft_tokens[:n, 0].copy_(t1)
        self.support_ids[:n, 0].copy_(ids)
        self.support_logits[:n, 0].copy_(processed)

        embeddings = self.model.embed_input_ids(tokens.flatten())
        embeddings = self.model.model.mtp_prefix_conditioner(
            embeddings, self.native_hidden[:n]
        )
        block = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
            inputs_embeds=embeddings,
        ).view(n, width, -1)
        # The bidirectional draft sees the full checkpoint block. Score and
        # sample only the positions the target will actually verify.
        tail_steps = self.num_speculative_steps - 1
        tail = block[:, 2 : 2 + tail_steps].contiguous()
        ids, unary = self.model.compute_candidates(tail.flatten(0, 1))
        ids = ids.view(n, tail_steps, self.selector_top_k)
        scores = self.model.model.candidate_selector(ids, unary.view_as(ids), tail, t1)
        self.support_ids[:n, 1:].copy_(ids)
        tail_tokens, tail_logits = sample_tail(
            scores, ids, temperatures, self.coins[:n, 1:], self.ranks
        )
        self.draft_tokens[:n, 1:].copy_(tail_tokens)
        self.support_logits[:n, 1:].copy_(tail_logits)
