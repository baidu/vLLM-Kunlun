import pytest
import torch
import math
from unittest.mock import MagicMock, PropertyMock
from typing import Optional


from vllm_kunlun.v1.attention.backends.kunlun_attn import KunlunAttentionImpl, KunlunAttentionBackend

# =======================
# 1. 基础配置与 Fixtures
# =======================

@pytest.fixture(scope="module")
def device():
    """检测并返回可用设备，优先 XPU"""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        pytest.skip("No XPU or CUDA device available for hardware testing")

@pytest.fixture
def attn_config():
    """定义 Attention 的基础参数"""
    return {
        "num_heads": 8,
        "head_size": 128,
        "scale": 1.0 / math.sqrt(128),
        "num_kv_heads": 8, # MHA 模式
        "block_size": 16,
        "max_seq_len": 128,
    }

@pytest.fixture
def impl(attn_config):
    """实例化 KunlunAttentionImpl"""
    return KunlunAttentionImpl(
        num_heads=attn_config["num_heads"],
        head_size=attn_config["head_size"],
        scale=attn_config["scale"],
        num_kv_heads=attn_config["num_kv_heads"],
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto"
    )

# =======================
# 2. 辅助函数：构造 Mock Metadata
# =======================

def create_mock_metadata(
    device, 
    is_prefill: bool, 
    num_seqs: int, 
    num_heads: int,
    block_size: int
):
    """
    构造一个符合 KunlunAttentionImpl 接口要求的 Metadata Mock 对象。
    关键在于：其中的 Tensor 属性必须是真实的 Device Tensor，以便底层算子计算。
    """
    metadata = MagicMock()
    
    # 基础计数
    num_tokens = num_seqs * 10  # 假设平均长度
    metadata.num_prefill_tokens = num_tokens if is_prefill else 0
    metadata.num_decode_tokens = num_tokens if not is_prefill else 0
    metadata.num_actual_tokens = num_tokens
    
    # 构造 Mock 的子对象 (prefill_metadata 或 decode_metadata)
    phase_meta = MagicMock()
    
    # 构造真实的 Tensor 数据
    # Block Tables: [num_seqs, max_blocks]
    block_tables = torch.arange(num_seqs * 4, dtype=torch.int32, device=device).view(num_seqs, 4)
    phase_meta.block_tables = block_tables
    
    # Seq Lens
    seq_lens = torch.full((num_seqs,), 10, dtype=torch.int32, device=device)
    phase_meta.seq_lens_tensor = seq_lens
    phase_meta.seq_lens_tensor_cpu = seq_lens.cpu()
    
    if is_prefill:
        # Prefill 特有的 Tensor
        # context_qlen_lod (batch_size + 1)
        lod = torch.arange(0, num_tokens + 1, 10, dtype=torch.int32, device=device)
        phase_meta.query_start_loc = lod
        phase_meta.query_start_loc_host = lod.cpu()
        
        # context_kvlen_lod
        phase_meta.kv_lod_xpu = lod
        phase_meta.kv_lod_cpu = lod.cpu()
        
        # 将子对象挂载到 mock 的 prefill_metadata 属性上
        type(metadata).prefill_metadata = PropertyMock(return_value=phase_meta)
        type(metadata).decode_metadata = PropertyMock(return_value=None)
    else:
        # Decode 特有的 Tensor
        # 将子对象挂载到 mock 的 decode_metadata 属性上
        type(metadata).prefill_metadata = PropertyMock(return_value=None)
        type(metadata).decode_metadata = PropertyMock(return_value=phase_meta)

    # Slot mapping (common)
    metadata.slot_mapping = torch.zeros(num_tokens, dtype=torch.long, device=device)
    
    return metadata

# =======================
# 3. 测试用例
# =======================

def test_initialization(impl):
    """测试初始化是否正常"""
    assert impl.num_heads == 8
    assert impl.head_size == 128
    # 使用 Backend 类检查名称，而不是实例
    assert KunlunAttentionBackend.get_name() == "Kunlun_v1"

@pytest.mark.skipif(not torch.cuda.is_available() and not hasattr(torch, 'xpu'), reason="Need device")
def test_forward_prefill(impl, device, attn_config):
    """
    测试 Prefill 阶段的 Forward
    验证点：代码能够跑通 xtorch_ops.prefill_attention 且输出形状正确
    """
    num_seqs = 2
    seq_len = 10
    total_tokens = num_seqs * seq_len
    
    # 1. 准备输入 Tensor
    q = torch.randn(total_tokens, attn_config["num_heads"], attn_config["head_size"], 
                   dtype=torch.float16, device=device)
    k = torch.randn(total_tokens, attn_config["num_kv_heads"], attn_config["head_size"], 
                   dtype=torch.float16, device=device)
    v = torch.randn(total_tokens, attn_config["num_kv_heads"], attn_config["head_size"], 
                   dtype=torch.float16, device=device)
    
    # KV Cache: [num_blocks, num_kv_heads, head_size/x, block_size, x] 
    # 或者 vllm 的标准格式 [num_blocks, num_heads, head_size, block_size] (取决于 layout)
    # 这里我们按照代码逻辑，只需要创建一个足够大的 buffer 即可
    kv_cache = torch.randn(100, attn_config["num_kv_heads"], attn_config["block_size"], attn_config["head_size"],
                          dtype=torch.float16, device=device)

    # 2. 构造 Metadata
    # 注意：我们不需要手动实例化 KunlunMetadata，而是使用 Mock 对象并填充真实的 Tensor
    # 这样可以绕过复杂的构造逻辑，直接测试 forward 里的计算流
    attn_metadata = create_mock_metadata(device, is_prefill=True, num_seqs=num_seqs, 
                                         num_heads=attn_config["num_heads"], 
                                         block_size=attn_config["block_size"])
    
    # 3. 执行 Forward
    output = impl.forward(
        layer=None,
        query=q,
        key=k,
        value=v,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata
    )
    
    # 4. 验证
    assert output is not None
    # Output shape 应该是 [total_tokens, num_heads * head_size]
    expected_shape = (total_tokens, attn_config["num_heads"] * attn_config["head_size"])
    assert output.shape == expected_shape
    print(f"\nPrefill Output Check Passed: Shape {output.shape}")

@pytest.mark.skipif(not torch.cuda.is_available() and not hasattr(torch, 'xpu'), reason="Need device")
def test_forward_decode(impl, device, attn_config):
    """
    测试 Decode 阶段的 Forward
    验证点：代码能够跑通 xtorch_ops.speculative_attention 或 paged_attention
    """
    num_seqs = 4
    # Decode 阶段通常只有最新的 token
    total_tokens = num_seqs 
    
    # 1. 准备输入 Tensor
    q = torch.randn(total_tokens, attn_config["num_heads"], attn_config["head_size"], 
                   dtype=torch.float16, device=device)
    
    # Decode 阶段通常 K/V 已经 cache 住了，这里传入 None 或 dummy 均可，主要看 forward 逻辑
    # 代码中: if key is not None... reshape_and_cache.
    # 我们这里模拟只有 Q，KV 已经在 Cache 中的情况
    k = None
    v = None
    
    kv_cache = torch.randn(100, attn_config["num_kv_heads"], attn_config["block_size"], attn_config["head_size"],
                          dtype=torch.float16, device=device)

    # 2. 构造 Metadata (Decode Mode)
    attn_metadata = create_mock_metadata(device, is_prefill=False, num_seqs=num_seqs, 
                                         num_heads=attn_config["num_heads"], 
                                         block_size=attn_config["block_size"])
    
    # 3. 执行 Forward
    # 注意：vllm forward 签名中 k, v 是 Optional
    output = impl.forward(
        layer=None,
        query=q,
        key=k,
        value=v,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata
    )
    
    # 4. 验证
    expected_shape = (total_tokens, attn_config["num_heads"] * attn_config["head_size"])
    assert output.shape == expected_shape
    print(f"\nDecode Output Check Passed: Shape {output.shape}")