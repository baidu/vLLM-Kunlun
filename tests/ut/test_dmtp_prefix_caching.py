# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DMTP scheduler isolation and cached-prefill metadata contracts on CPU."""

import importlib.util
import logging
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_kunlun.registration import compat_patches


class SpecConfig:
    method = "dflash"
    num_speculative_tokens = 7

    def __init__(self, architecture="DFlash2T1ResidualDraftModel"):
        self.draft_model_config = SimpleNamespace(architectures=[architecture])

    def use_eagle(self):
        return False

    def use_dflash(self):
        return self.method == "dflash"


def scheduler_config(spec, prefix=True):
    return SimpleNamespace(
        speculative_config=spec,
        cache_config=SimpleNamespace(enable_prefix_caching=prefix),
    )


@pytest.fixture
def scheduler_module(module_factory):
    class Scheduler:
        def __init__(self, vllm_config, kv_cache_config=None, fail=False):
            spec = vllm_config.speculative_config
            self.use_eagle = False
            self.num_spec_tokens = spec.num_speculative_tokens if spec else 0
            self.num_lookahead_tokens = 0
            if spec is not None:
                if spec.use_eagle():
                    self.use_eagle = True
                    self.num_lookahead_tokens = self.num_spec_tokens
                if spec.use_dflash():
                    self.num_lookahead_tokens = self.num_spec_tokens + 1
            self.kv_cache_manager = SimpleNamespace(
                use_eagle=self.use_eagle,
                eagle_group_ids={
                    i
                    for i, group in enumerate(
                        getattr(kv_cache_config, "kv_cache_groups", ())
                    )
                    if group.is_eagle_group
                },
            )
            # Another config instance must never observe the temporary override.
            assert SpecConfig().use_eagle() is False
            if fail:
                raise RuntimeError("initialization failed")

    class AsyncScheduler(Scheduler):
        pass

    return module_factory(Scheduler=Scheduler, AsyncScheduler=AsyncScheduler)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_scheduler_protects_dmtp_kv_and_restores_config(scheduler_module, asynchronous):
    spec = SpecConfig()
    compat_patches._apply_dmtp_scheduler_patch(scheduler_module)
    cls = (
        scheduler_module.AsyncScheduler if asynchronous else scheduler_module.Scheduler
    )
    scheduler = cls(vllm_config=scheduler_config(spec))
    assert scheduler.use_eagle is True
    assert scheduler.kv_cache_manager.use_eagle is True
    assert scheduler.num_lookahead_tokens == 8
    assert spec.use_eagle() is False
    assert "use_eagle" not in vars(spec)


@pytest.mark.parametrize(
    "spec,prefix",
    [(SpecConfig("DFlash2DraftModel"), True), (SpecConfig(), False), (None, True)],
)
def test_other_scheduler_configs_keep_their_behavior(scheduler_module, spec, prefix):
    compat_patches._apply_dmtp_scheduler_patch(scheduler_module)
    scheduler = scheduler_module.Scheduler(scheduler_config(spec, prefix))
    assert scheduler.use_eagle is False
    assert scheduler.kv_cache_manager.use_eagle is False
    assert scheduler.num_lookahead_tokens == (8 if spec else 0)


def test_scheduler_preserves_upstream_eagle_true(scheduler_module, monkeypatch):
    # Some vLLM versions already include DFlash in use_eagle(). Preserve it.
    monkeypatch.setattr(SpecConfig, "use_eagle", lambda self: True)
    spec = SpecConfig()

    def initialize(self, config):
        self.use_eagle = config.speculative_config.use_eagle()
        self.num_spec_tokens, self.num_lookahead_tokens = 7, 8

    scheduler_module.Scheduler.__init__ = initialize
    compat_patches._apply_dmtp_scheduler_patch(scheduler_module)
    scheduler = scheduler_module.Scheduler(scheduler_config(spec))
    assert scheduler.use_eagle is True
    assert spec.use_eagle() is True
    assert "use_eagle" not in vars(spec)


@pytest.mark.parametrize("existing_override", [False, True])
def test_scheduler_restores_config_when_init_raises(
    scheduler_module, existing_override
):
    spec = SpecConfig()

    def previous():
        return False

    if existing_override:
        spec.use_eagle = previous
    compat_patches._apply_dmtp_scheduler_patch(scheduler_module)
    with pytest.raises(RuntimeError, match="initialization failed"):
        scheduler_module.Scheduler(scheduler_config(spec), fail=True)
    assert spec.use_eagle() is False
    assert ("use_eagle" in vars(spec)) is existing_override
    if existing_override:
        assert spec.use_eagle is previous


def test_scheduler_hook_is_idempotent_and_registered(scheduler_module, module_factory):
    assert compat_patches._dmtp_scheduler_applied(module_factory())
    compat_patches._apply_dmtp_scheduler_patch(module_factory())
    assert not compat_patches._dmtp_scheduler_applied(scheduler_module)
    compat_patches._apply_dmtp_scheduler_patch(scheduler_module)
    original = scheduler_module.Scheduler.__init__
    compat_patches._apply_dmtp_scheduler_patch(scheduler_module)
    assert scheduler_module.Scheduler.__init__ is original
    assert compat_patches._dmtp_scheduler_applied(scheduler_module)
    assert (
        "vllm.v1.core.sched.scheduler",
        compat_patches._dmtp_scheduler_applied,
        compat_patches._apply_dmtp_scheduler_patch,
    ) in compat_patches.DEFAULT_HOOKS


@pytest.mark.parametrize("keyword", [False, True])
def test_only_native_kv_group_gets_successor_boundary_protection(
    scheduler_module, keyword
):
    groups = [
        SimpleNamespace(layer_names=names, is_eagle_group=False)
        for names in [
            ["model.layers.0.attn", "dmtp_native.layers.0.self_attn.attn"],
            ["draft_model.layers.0.attn"],
            ["model.layers.1.mamba"],
        ]
    ]
    kv_config = SimpleNamespace(kv_cache_groups=groups)
    compat_patches._apply_dmtp_scheduler_patch(scheduler_module)
    config = scheduler_config(SpecConfig())
    if keyword:
        scheduler = scheduler_module.Scheduler(config, kv_cache_config=kv_config)
    else:
        scheduler = scheduler_module.Scheduler(config, kv_config)
    assert [group.is_eagle_group for group in groups] == [True, False, False]
    assert scheduler.kv_cache_manager.eagle_group_ids == {0}


def test_missing_native_kv_group_is_rejected(scheduler_module):
    compat_patches._apply_dmtp_scheduler_patch(scheduler_module)
    config = scheduler_config(SpecConfig())
    kv_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=["model.layers.0.attn"],
                is_eagle_group=False,
            )
        ]
    )
    with pytest.raises(ValueError, match="native MTP KV group"):
        scheduler_module.Scheduler(config, kv_config)
    assert "use_eagle" not in vars(config.speculative_config)


@pytest.mark.parametrize(
    "spec,expected",
    [
        (None, False),
        (SimpleNamespace(method="dflash"), False),
        (SimpleNamespace(method="dflash", draft_model_config=None), False),
        (SpecConfig(), True),
        (SpecConfig("DFlash2DraftModel"), False),
        (
            SimpleNamespace(
                method="mtp", draft_model_config=SpecConfig().draft_model_config
            ),
            False,
        ),
    ],
)
def test_dmtp_detection_is_narrow(spec, expected):
    assert compat_patches._is_dmtp_speculative_config(spec) is expected


@pytest.fixture
def dmtp(stub_module_tree):
    class BaseSpeculator:
        @classmethod
        def _validate_config(cls, config):
            pass

    for name, attrs in {
        "kunlun_ops": {},
        "safetensors": {"safe_open": None},
        "vllm.compilation.backends": {"set_model_tag": nullcontext},
        "vllm.config": {"set_current_vllm_config": nullcontext},
        "vllm.config.compilation": {"CUDAGraphMode": SimpleNamespace(NONE=0)},
        "vllm.distributed": {"get_tp_group": None},
        "vllm.forward_context": {"set_forward_context": nullcontext},
        "vllm.logger": {"init_logger": logging.getLogger},
        "vllm.model_executor.model_loader.utils": {
            "process_weights_after_loading": None
        },
        "vllm.model_executor.models.qwen3_5_mtp": {
            "Qwen3_5MultiTokenPredictor": object
        },
        "vllm.utils.torch_utils": {"set_default_torch_dtype": nullcontext},
        "vllm.v1.worker.gpu.attn_utils": {
            "build_attn_metadata": None,
            "init_attn_backend": None,
        },
        "vllm.v1.worker.gpu.cudagraph_utils": {
            "AttentionState": lambda *args: args,
            "CudaGraphManager": object,
        },
        "vllm_kunlun.v1.worker.gpu.spec_decode.dflash2.speculator": {
            "DFlash2Speculator": BaseSpeculator
        },
    }.items():
        stub_module_tree(name, **attrs)
    path = (
        Path(__file__).resolve().parents[2]
        / "vllm_kunlun/v1/worker/gpu/spec_decode/dmtp.py"
    )
    spec = importlib.util.spec_from_file_location(
        "vllm_kunlun.v1.worker.gpu.spec_decode._prefix_test", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def valid_config():
    cfg = dict(
        input_layout=["A", "T1"] + ["MASK"] * 6,
        mtp_prefix=True,
        mtp_t1_residual=True,
        mtp_hidden_only=True,
        mtp_prefix_injection="add_to_t1_input_embedding",
        mtp_t1_input_slot=1,
        mtp_hidden_index_relative_to_T1=-2,
        block_size=8,
        mtp_proposal_token_count=1,
        dflash_proposal_token_count=6,
        block_attention="bidirectional",
        context_kv_boundary="strictly_before_anchor",
        hidden_norm_zero_centered_gamma=True,
        conv_base_kernel_identity_folded=True,
        swa_window_size=4096,
        draft_layer_context_windows=[4096] * 5,
    )
    draft = SimpleNamespace(
        hidden_size=16, head_dim=256, partial_rotary_factor=0.25, dflash_config=cfg
    )
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            num_speculative_tokens=7,
            uses_dynamic_speculative_decoding=lambda: False,
            draft_model_config=SimpleNamespace(hf_config=draft),
        ),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                model_type="qwen3_5_text",
                mtp_num_hidden_layers=1,
                hidden_size=16,
                num_hidden_layers=64,
            )
        ),
        cache_config=SimpleNamespace(
            enable_prefix_caching=True, kv_sharing_fast_prefill=False
        ),
    )


def test_dmtp_allows_prefix_caching(dmtp, valid_config):
    dmtp.DMTPSpeculator._validate_config(valid_config)


def test_dmtp_rejects_fast_prefill(dmtp, valid_config):
    valid_config.cache_config.kv_sharing_fast_prefill = True
    with pytest.raises(ValueError, match="kv_sharing_fast_prefill"):
        dmtp.DMTPSpeculator._validate_config(valid_config)


@pytest.mark.parametrize("rejections", [[0, 0], [3, 7]])
def test_eight_row_cached_prefill_and_decode_graph_match_eager(
    dmtp, monkeypatch, rejections
):
    """Compare accepted inputs, slots and causal KV bounds, including mixed rejection."""
    records, context = [], {}

    def metadata(
        groups,
        n,
        count,
        starts,
        starts_cpu,
        width,
        lens,
        max_len,
        tables,
        slots,
        config,
        causal,
    ):
        md = SimpleNamespace(
            slot_mapping=slots[0, :count],
            starts=starts_cpu.clone(),
            kv_lod_xpu=torch.cat(
                (torch.zeros(1, dtype=torch.int32), lens.cumsum(0).int())
            ),
        )
        return {"native": md}

    @contextmanager
    def forward_context(md, *args, **kwargs):
        context["md"] = md["native"]
        yield

    def native(input_ids, positions, hidden_states):
        md = context["md"]
        lens = md.kv_lod_xpu.diff()
        for i, (start, end) in enumerate(zip(md.starts[:-1], md.starts[1:])):
            for lane, row in enumerate(range(int(start), int(end))):
                if md.slot_mapping[row] >= 0:
                    records.append(
                        (
                            int(input_ids[row]),
                            int(positions[row]),
                            int(md.slot_mapping[row]),
                            int(lens[i] - (end - start) + lane + 1),
                            hidden_states[row].tolist(),
                        )
                    )
        return hidden_states + input_ids[:, None] + positions[:, None]

    monkeypatch.setattr(dmtp, "build_attn_metadata", metadata)
    monkeypatch.setattr(dmtp, "set_forward_context", forward_context)
    batch = SimpleNamespace(
        num_reqs=2,
        num_tokens=16,
        num_scheduled_tokens=np.array([8, 8]),
        input_ids=torch.arange(16, dtype=torch.int32),
        positions=torch.cat((torch.arange(384, 392), torch.arange(512, 520))),
        seq_lens=torch.tensor([392, 520], dtype=torch.int32),
        query_start_loc_np=np.array([0, 8, 16], dtype=np.int32),
    )
    spec = dmtp.DMTPSpeculator.__new__(dmtp.DMTPSpeculator)
    spec.native_name, spec.native_groups = "native", []
    spec.device, spec.max_model_len, spec.kv_cache_config = "cpu", 2048, None
    spec.native_hidden = torch.zeros(2, 2)
    spec.native, spec.native_graphs = native, None
    spec._stage_context_window = lambda *args: None
    spec.query_offsets, spec.batch_rows = torch.arange(8), torch.arange(2)
    spec.vllm_config = None
    spec.block_tables = SimpleNamespace(
        input_block_tables=[torch.zeros(2, 16, dtype=torch.int32)],
        slot_mappings=torch.arange(16).view(1, -1),
    )
    hidden = torch.arange(32, dtype=torch.float32).view(16, 2)
    anchors, rejected = torch.tensor([80, 90]), torch.tensor(rejections)
    spec._update_native(batch, hidden, anchors, rejected, False)
    eager_records, eager_hidden = records.copy(), spec.native_hidden.clone()
    records.clear()
    graph = dmtp.NativeMTPGraphs.__new__(dmtp.NativeMTPGraphs)
    graph.spec, graph.verify_width, graph.group = spec, 8, 0
    graph.ids = torch.zeros(2, 8, dtype=torch.int32)
    graph.positions = torch.zeros(2, 8, dtype=torch.long)
    graph.hidden = torch.zeros(16, 2)
    graph.anchors = torch.zeros(2, dtype=torch.int32)
    graph.rejected = torch.zeros(2, dtype=torch.int32)
    graph.lens = torch.full((2,), 8, dtype=torch.int32)
    graph.slots = torch.full((1, 16), -1, dtype=torch.long)
    graph.target_slots = torch.full((2, 8), -1, dtype=torch.long)
    desc = SimpleNamespace(num_reqs=2, num_tokens=16)
    graph.exact, graph.replay_logged = {2: desc}, False
    forward, _ = graph.create_forward(desc, False)
    graph.run_fullgraph = lambda desc: forward(None)
    assert graph.replay(batch, hidden, anchors, rejected)
    assert records == eager_records
    torch.testing.assert_close(spec.native_hidden, eager_hidden, rtol=0, atol=0)
    batch.num_scheduled_tokens = np.array([7, 9])
    assert not graph.replay(batch, hidden, anchors, rejected)


def test_chunked_prefill_uses_next_prompt_token_with_request_mapping(dmtp, monkeypatch):
    spec = dmtp.DMTPSpeculator.__new__(dmtp.DMTPSpeculator)
    captured = []
    spec._update_native = lambda batch, hidden, anchors, *args: captured.append(
        anchors.tolist()
    )
    spec.draft_logits, spec.separate_query = None, False
    monkeypatch.setattr(
        dmtp.DFlash2Speculator, "propose", lambda *args: "draft", raising=False
    )
    batch = SimpleNamespace(num_reqs=2, idx_mapping=torch.tensor([2, 0]))
    result = spec.propose(
        batch,
        None,
        None,
        None,
        None,
        torch.tensor([0, 1]),
        torch.zeros(2),
        torch.tensor([70, 71, 72]),
        torch.tensor([80, 81, 82]),
        None,
        None,
    )
    assert captured == [[82, 70]]
    assert result == "draft"


def test_cached_context_window_respects_allocated_pages(dmtp):
    spec = dmtp.DMTPSpeculator.__new__(dmtp.DMTPSpeculator)
    spec.window, spec.max_model_len, spec.num_query_per_req = 4096, 20480, 8
    spec.device, spec.separate_query = "cpu", False
    spec.draft_kv_cache_group_ids, spec._layer_group_idx = [0], [0]
    spec.query_offsets = torch.arange(8)
    attn = SimpleNamespace(context_pages=None)
    spec.model = SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attn)])
    )
    tables = torch.tensor([[10, 11, 99, 99], [20, 21, 22, 23]])
    spec.block_tables = SimpleNamespace(
        num_blocks=SimpleNamespace(np=np.array([[2, 4]])),
        kernel_block_sizes=[128],
        input_block_tables=[tables],
    )
    batch = SimpleNamespace(
        num_reqs=2,
        idx_mapping_np=np.array([0, 1]),
        seq_lens_cpu_upper_bound=torch.tensor([253, 511]),
        seq_lens=torch.tensor([253, 511]),
    )
    spec._stage_context_window(batch, torch.tensor([0, 2]))
    assert spec.separate_query
    assert spec.true_query_positions[:, 0].tolist() == [253, 509]
    for i, length in enumerate([253, 509]):
        pages, offsets = attn.context_pages[i]
        positions = torch.arange(length)
        torch.testing.assert_close(pages, tables[i, positions // 128])
        torch.testing.assert_close(offsets, positions % 128)
        assert 99 not in pages.tolist()


@pytest.fixture
def target_runner(load_kunlun_override, stub_module_tree):
    class Runner:
        def add_requests(self, output):
            pass

        def execute_model(self, output, *args):
            result = self.cudagraph_manager.dispatch(2, 16, 8, 0)
            if getattr(self, "fail", False):
                raise RuntimeError("forward failed")
            return result

    class Manager:
        def dispatch(self, n, count, uniform, loras):
            return uniform

    stub_module_tree("vllm.v1.kv_cache_interface", MambaSpec=object)
    stub_module_tree("vllm_kunlun.v1.worker.mamba_utils", MambaStateZeroer=object)
    _, upstream = load_kunlun_override(
        "v1/worker/gpu/model_runner.py",
        "vllm.v1.worker.gpu.model_runner",
        GPUModelRunner=Runner,
    )
    runner = upstream.GPUModelRunner()
    runner.vllm_config = scheduler_config(SpecConfig())
    runner.cudagraph_manager = Manager()
    return runner


@pytest.mark.parametrize("mixed", [False, True])
def test_eight_row_target_prefill_avoids_decode_graph(target_runner, mixed):
    output = SimpleNamespace(
        num_scheduled_tokens={"a": 8, "b": 8},
        scheduled_spec_decode_tokens={"b": list(range(7))} if mixed else {},
    )
    assert target_runner.execute_model(output) is None
    assert "dispatch" not in vars(target_runner.cudagraph_manager)


@pytest.mark.parametrize(
    "dummy,architecture,drafts,expected",
    [
        (True, "DFlash2T1ResidualDraftModel", {}, 8),
        (False, "DFlash2DraftModel", {}, 8),
        (
            False,
            "DFlash2T1ResidualDraftModel",
            {"a": list(range(7)), "b": list(range(7))},
            8,
        ),
    ],
)
def test_target_decode_and_other_drafts_keep_graphs(
    target_runner, dummy, architecture, drafts, expected
):
    target_runner.vllm_config = scheduler_config(SpecConfig(architecture))
    output = SimpleNamespace(
        num_scheduled_tokens={"a": 8, "b": 8}, scheduled_spec_decode_tokens=drafts
    )
    assert target_runner.execute_model(output, dummy_run=dummy) == expected


@pytest.mark.parametrize("existing_override", [False, True])
def test_target_dispatch_is_restored_on_failure(target_runner, existing_override):
    manager = target_runner.cudagraph_manager
    previous = manager.dispatch
    if existing_override:
        manager.dispatch = previous
    target_runner.fail = True
    output = SimpleNamespace(
        num_scheduled_tokens={"a": 8, "b": 8}, scheduled_spec_decode_tokens={}
    )
    with pytest.raises(RuntimeError, match="forward failed"):
        target_runner.execute_model(output)
    assert ("dispatch" in vars(manager)) is existing_override
    assert manager.dispatch == previous
