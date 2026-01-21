import pytest
import torch
import math
from unittest.mock import MagicMock, PropertyMock, patch

# 导入被测类
from vllm.attention.backends.abstract import AttentionType
from vllm_kunlun.v1.attention.backends.mla.flashmla import FlashMLAImpl, FlashMLABackend

# 尝试导入 ops
try:
    from vllm_kunlun.ops.attention.flashmla import get_mla_metadata
except ImportError:
    get_mla_metadata = None

# =======================
# 1. Mock Global Config
# =======================
@pytest.fixture(autouse=True)
def mock_vllm_config():
    """防止初始化时检测硬件报错"""
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
    """Mock 分布式环境，防止 get_dcp_group() 报错"""
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
# 3. Fixtures
# =======================
@pytest.fixture(scope="module")
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    pytest.skip("CUDA device not available")

@pytest.fixture
def mla_config():
    qk_nope = 128
    qk_rope = 64
    return {
        "num_heads": 128,
        "head_size": qk_nope + qk_rope,
        "scale": 1.0 / math.sqrt(qk_nope + qk_rope),
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": qk_nope,
        "qk_rope_head_dim": qk_rope,
        "qk_head_dim": qk_nope + qk_rope,
        "v_head_dim": 128,
        "kv_b_proj": MagicMock(name="kv_b_proj"), 
    }

@pytest.fixture
def impl(mla_config):
    mla_args = {
        "q_lora_rank": mla_config["q_lora_rank"],
        "kv_lora_rank": mla_config["kv_lora_rank"],
        "qk_nope_head_dim": mla_config["qk_nope_head_dim"],
        "qk_rope_head_dim": mla_config["qk_rope_head_dim"],
        "qk_head_dim": mla_config["qk_head_dim"],
        "v_head_dim": mla_config["v_head_dim"],
        "kv_b_proj": mla_config["kv_b_proj"],
    }
    impl_obj = FlashMLAImpl(
        num_heads=mla_config["num_heads"],
        head_size=mla_config["head_size"],
        scale=mla_config["scale"],
        num_kv_heads=1, 
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type=AttentionType.DECODER,
        kv_sharing_target_layer_name=None,
        **mla_args
    )
    return impl_obj

def create_mock_decode_metadata(device, num_seqs, num_heads, block_size, max_seq_len=1024):
    if get_mla_metadata is None:
        pytest.fail("Cannot import get_mla_metadata")

    num_blocks = (max_seq_len + block_size - 1) // block_size
    block_tables = torch.arange(num_seqs * num_blocks, dtype=torch.int32, device=device).view(num_seqs, num_blocks)
    seq_lens = torch.randint(1, max_seq_len, (num_seqs,), dtype=torch.int32, device=device)
    
    tile_scheduler_metadata, num_splits = get_mla_metadata(seq_lens, num_heads, 1)
    slot_mapping = torch.arange(num_seqs, dtype=torch.long, device=device)

    metadata = MagicMock()
    metadata.num_actual_tokens = num_seqs
    metadata.num_decodes = num_seqs
    metadata.num_prefills = 0
    metadata.num_decode_tokens = num_seqs
    metadata.slot_mapping = slot_mapping 
    
    decode_meta = MagicMock()
    decode_meta.block_table = block_tables
    decode_meta.seq_lens = seq_lens
    decode_meta.tile_scheduler_metadata = tile_scheduler_metadata
    decode_meta.num_splits = num_splits

    type(metadata).decode = PropertyMock(return_value=decode_meta)
    type(metadata).prefill = PropertyMock(return_value=None)
    return metadata

# =======================
# 4. Tests
# =======================

def test_backend_name():
    assert FlashMLABackend.get_name() == "FLASHMLA"

def test_initialization(impl, mla_config):
    assert impl.num_heads == mla_config["num_heads"]
    assert impl.can_return_lse_for_decode is True

def test_unsupported_features(mla_config):
    valid_kwargs = {
        "num_heads": 128, 
        "head_size": 128, 
        "scale": 1.0, 
        "num_kv_heads": 1,
        "kv_cache_dtype": "auto",
        "attn_type": AttentionType.DECODER,
        "kv_sharing_target_layer_name": None,
        "q_lora_rank": mla_config["q_lora_rank"],
        "kv_lora_rank": mla_config["kv_lora_rank"],
        "qk_nope_head_dim": mla_config["qk_nope_head_dim"],
        "qk_rope_head_dim": mla_config["qk_rope_head_dim"],
        "qk_head_dim": mla_config["qk_head_dim"],
        "v_head_dim": mla_config["v_head_dim"],
        "kv_b_proj": mla_config["kv_b_proj"],
    }

    with pytest.raises(NotImplementedError, match="alibi"):
        FlashMLAImpl(
            **valid_kwargs,
            alibi_slopes=[0.1], 
            sliding_window=None, 
            logits_soft_cap=None,
        )

    invalid_encoder_kwargs = valid_kwargs.copy()
    invalid_encoder_kwargs["attn_type"] = AttentionType.ENCODER
    
    with pytest.raises(NotImplementedError, match="Encoder"):
        FlashMLAImpl(
            **invalid_encoder_kwargs,
            alibi_slopes=None,
            sliding_window=None,
            logits_soft_cap=None,
        )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA")
def test_forward_decode(impl, device, mla_config):
    num_seqs = 4
    bs = num_seqs
    
    # 注入 Up-Key Transpose Matrix
    if not hasattr(impl, "W_UK_T"):
        impl.W_UK_T = torch.randn(
            mla_config["num_heads"], 
            mla_config["qk_nope_head_dim"], 
            mla_config["kv_lora_rank"], 
            dtype=torch.float16, 
            device=device
        )
    # 注入 Up-Value Matrix
    if not hasattr(impl, "W_UV"):
        impl.W_UV = torch.randn(
            mla_config["num_heads"],
            mla_config["kv_lora_rank"],
            mla_config["v_head_dim"],
            dtype=torch.float16,
            device=device
        )

    impl.q_pad_num_heads = None
    
    q_dim = mla_config["num_heads"] * mla_config["head_size"]
    
    # Q 必须是 3D 形状 (bs, num_heads, head_size)
    q = torch.randn(
        bs, 
        mla_config["num_heads"], 
        mla_config["head_size"], 
        dtype=torch.float16, 
        device=device
    )
    
    k_c_normed = torch.randn(bs, mla_config["kv_lora_rank"], dtype=torch.float16, device=device)
    k_pe = torch.randn(bs, mla_config["qk_rope_head_dim"], dtype=torch.float16, device=device)

    block_size = 16
    num_blocks = 100
    kv_dim = mla_config["kv_lora_rank"] + mla_config["qk_rope_head_dim"]
    
    kv_cache = torch.randn(
        num_blocks, block_size, kv_dim,
        dtype=torch.float16, device=device
    )

    attn_metadata = create_mock_decode_metadata(
        device, 
        num_seqs=num_seqs, 
        num_heads=mla_config["num_heads"], 
        block_size=block_size
    )

    layer = MagicMock()
    layer._q_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    layer._k_scale = torch.tensor([1.0], dtype=torch.float32, device=device)

    output = torch.empty(bs, q_dim, dtype=torch.float16, device=device)

    # [CRITICAL FIX] 只接收单个返回值
    res_output = impl.forward(
        layer=layer,
        q=q,
        k_c_normed=k_c_normed,
        k_pe=k_pe,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
        output=output
    )

    # 验证
    assert res_output is not None
    assert res_output.shape == (bs, q_dim)
    
    print(f"\nForward Success! Output shape: {res_output.shape}")