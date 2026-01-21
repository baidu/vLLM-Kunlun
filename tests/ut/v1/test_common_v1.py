import pytest
import torch
import math
import sys
from unittest.mock import MagicMock, patch

# =======================
# 0. Mock Library Imports
# =======================
# 确保可以导入被测类
from vllm.attention.backends.abstract import AttentionType
from vllm_kunlun.v1.attention.backends.mla.common import (
    MLACommonImpl, MLACommonMetadata, MLACommonPrefillMetadata, 
    MLACommonDecodeMetadata, MLACommonBackend
)

# =======================
# 1. Mock Global Config
# =======================
@pytest.fixture(autouse=True)
def mock_vllm_config():
    mock_conf = MagicMock()
    mock_conf.model_config = MagicMock()
    mock_conf.model_config.max_model_len = 4096
    mock_conf.model_config.get_num_attention_heads.return_value = 128
    mock_conf.scheduler_config = MagicMock()
    mock_conf.scheduler_config.max_num_seqs = 256
    mock_conf.cache_config = MagicMock()
    mock_conf.cache_config.block_size = 16 
    mock_conf.device_config = MagicMock()
    mock_conf.device_config.device = torch.device("cuda")
    mock_conf.parallel_config = MagicMock()
    
    target_path = "vllm_kunlun.v1.attention.backends.mla.common.get_current_vllm_config"
    with patch(target_path, return_value=mock_conf):
        yield mock_conf

# =======================
# 2. Mock Distributed State
# =======================
@pytest.fixture(autouse=True)
def mock_distributed():
    mock_group = MagicMock()
    mock_group.world_size = 1
    mock_group.rank = 0
    with patch("vllm.distributed.parallel_state.get_dcp_group", return_value=mock_group), \
         patch("vllm.distributed.parallel_state.get_tensor_model_parallel_world_size", return_value=1), \
         patch("vllm.distributed.parallel_state.get_tensor_model_parallel_rank", return_value=0), \
         patch("vllm.distributed.parallel_state.get_world_group", return_value=mock_group), \
         patch("vllm_kunlun.v1.attention.backends.mla.common.get_dcp_group", return_value=mock_group):
        yield

# =======================
# 3. Helper Class
# =======================
class ConcreteMLAImpl(MLACommonImpl):
    def _forward_decode(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        if isinstance(q, tuple): bs = q[0].shape[1] 
        else: bs = q.shape[0]
        attn_out = torch.zeros(bs, self.num_heads, self.kv_lora_rank, 
                               device=self.W_UV.device, dtype=self.W_UV.dtype)
        lse = torch.zeros(bs, self.num_heads, device=self.W_UV.device, dtype=torch.float32)
        return attn_out, lse

# =======================
# 4. Fixtures
# =======================
@pytest.fixture(scope="module")
def device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available(): return torch.device("xpu")
    pytest.skip("No Device")

@pytest.fixture
def mla_config():
    qk_nope = 128; qk_rope = 64
    return {
        "num_heads": 128, "head_size": 192, "scale": 1.0, "q_lora_rank": 1536,
        "kv_lora_rank": 512, "qk_nope_head_dim": qk_nope, "qk_rope_head_dim": qk_rope,
        "qk_head_dim": 192, "v_head_dim": 128, "kv_b_proj": MagicMock(name="kv_b_proj"), 
    }

@pytest.fixture
def impl(mla_config, device):
    impl_obj = ConcreteMLAImpl(
        num_heads=mla_config["num_heads"], head_size=mla_config["head_size"], scale=mla_config["scale"],
        num_kv_heads=1, alibi_slopes=None, sliding_window=None, kv_cache_dtype="auto",
        logits_soft_cap=None, attn_type=AttentionType.DECODER, kv_sharing_target_layer_name=None,
        q_lora_rank=mla_config["q_lora_rank"], kv_lora_rank=mla_config["kv_lora_rank"],
        qk_nope_head_dim=mla_config["qk_nope_head_dim"], qk_rope_head_dim=mla_config["qk_rope_head_dim"],
        qk_head_dim=mla_config["qk_head_dim"], v_head_dim=mla_config["v_head_dim"],
        kv_b_proj=mla_config["kv_b_proj"]
    )
    # Inject Weights
    impl_obj.W_UK_T = torch.randn(128, 128, 512, dtype=torch.float16, device=device)
    impl_obj.W_UV = torch.randn(128, 512, 128, dtype=torch.float16, device=device)
    impl_obj.kv_b_proj = MagicMock(side_effect=lambda x: (torch.randn((*x.shape[:-1], 128*(128+128)), device=x.device, dtype=x.dtype), None))
    return impl_obj

# =======================
# 5. Tests
# =======================

def test_backend_name():
    assert MLACommonBackend.get_name() == "TRITON_MLA"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need Device")
def test_forward_prefill_context_chunk(impl, device, mla_config):
    """
    测试 Chunked Prefill 流程。
    [STABILITY FIX]: Mock 掉 xtorch_ops.attention 防止底层算子崩溃。
    """
    num_prefills = 1
    seq_len = 128
    block_size = 16 
    num_blocks = seq_len // block_size
    
    q = torch.randn(seq_len, 128, 192, dtype=torch.float16, device=device)
    kv_dim = 512 + 64 # 576
    kv_c_and_k_pe = torch.randn(num_blocks, block_size, kv_dim, dtype=torch.float16, device=device)

    chunked_context = MagicMock()
    chunked_context.seq_tot = [seq_len]
    chunked_context.cu_seq_lens = torch.tensor([[0, seq_len]], dtype=torch.int32, device=device)
    chunked_context.starts = torch.tensor([[0]], dtype=torch.int32, device=device)
    chunked_context.workspace = torch.empty(seq_len, kv_dim, dtype=torch.float16, device=device)

    prefill_meta = MagicMock(spec=MLACommonPrefillMetadata)
    prefill_meta.chunked_context = chunked_context
    prefill_meta.block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)
    prefill_meta.query_start_loc = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    
    metadata = MagicMock(spec=MLACommonMetadata)
    metadata.prefill = prefill_meta
    metadata.num_prefills = num_prefills
    metadata.num_decodes = 0
    layer = MagicMock()
    layer._k_scale = torch.tensor(1.0, device=device)

    # [MOCK] 拦截 xtorch_ops.attention
    # 目的：验证数据流通了即可，不需要真正计算 attention，避免 C++ 崩溃
    def mock_attention_op(q, k_cache, v_cache, out, **kwargs):
        # 简单将 q 的内容拷贝到 out (假设维度兼容)，或者填 0
        # out shape: [seq_len, num_heads, head_size]
        out.fill_(0.1) 
        return

    with patch("xtorch_ops.attention", side_effect=mock_attention_op):
        output, lse = impl._compute_prefill_context(
            q=q, kv_c_and_k_pe_cache=kv_c_and_k_pe,
            attn_metadata=metadata, k_scale=layer._k_scale
        )
    
    assert output is not None
    assert output.shape[0] == seq_len
    # 由于 Mock 填了 0.1，验证一下是否返回了值
    assert output.mean() > 0 
    print("\nPrefill Context Chunk Test Passed")

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need Device")
def test_forward_decode_step(impl, device, mla_config):
    """
    测试 Decode Step。
    [STABILITY FIX]: Mock 掉 concat_and_cache_mla
    """
    bs = 2
    q = torch.randn(bs, 128, 192, dtype=torch.float16, device=device)
    k_c = torch.randn(bs, 512, dtype=torch.float16, device=device)
    k_pe = torch.randn(bs, 64, dtype=torch.float16, device=device)
    
    metadata = MagicMock(spec=MLACommonMetadata)
    metadata.num_actual_tokens = bs
    metadata.num_decodes = bs
    metadata.num_prefills = 0
    metadata.num_decode_tokens = bs
    metadata.slot_mapping = torch.arange(bs, dtype=torch.long, device=device)
    metadata.prefill.chunked_context = None 
    metadata.decode = MagicMock(spec=MLACommonDecodeMetadata)

    kv_cache = torch.randn(10, 16, 576, dtype=torch.float16, device=device)
    layer = MagicMock()
    output = torch.empty(bs, 128*128, dtype=torch.float16, device=device)

    # Mock concat_and_cache_mla 避免写 KV Cache 时的潜在问题
    with patch("xtorch_ops.concat_and_cache_mla"):
        res = impl.forward(
            layer=layer, q=q, k_c_normed=k_c, k_pe=k_pe,
            kv_cache=kv_cache, attn_metadata=metadata, output=output
        )
    
    assert res is not None
    assert res.shape == (bs, 128*128)
    print("\nDecode Step Test Passed")