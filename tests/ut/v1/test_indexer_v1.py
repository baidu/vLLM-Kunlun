import pytest
import torch
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, patch

# =======================
# 1. Real Imports Check
# =======================
try:
    import vllm
    # 尝试导入真实类型用于 Type Hint（非必须，但有了更好）
    from vllm.v1.attention.backends.utils import CommonAttentionMetadata
except ImportError:
    pytest.fail("vLLM not found! This test requires a real vLLM environment.")

# =======================
# 2. Source Code Injection (Code Under Test)
# =======================
# 假设这些类是本次新增的，尚未合并到 vllm 库中，因此在此定义
@dataclass
class DeepseekV32IndexerPrefillMetadata:
    chunks: list

@dataclass
class DeepSeekV32IndexerDecodeMetadata:
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    decode_lens: torch.Tensor
    requires_padding: bool
    schedule_metadata: torch.Tensor

@dataclass
class DeepseekV32IndexerMetadata:
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    num_reqs: int
    max_query_len: int
    max_seq_len: int
    num_actual_tokens: int
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    head_dim: int
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    decode: Optional[DeepSeekV32IndexerDecodeMetadata] = None
    prefill: Optional[DeepseekV32IndexerPrefillMetadata] = None

# 被测函数
def kunlun_build(self,
              common_prefix_len: int,
              common_attn_metadata,
              fast_build: bool = False) -> DeepseekV32IndexerMetadata:

    # [Real Import] 这里会尝试从真实的 vLLM 库中导入函数
    # 在测试中，我们会 patch 这些路径
    from vllm.v1.attention.backends.utils import split_decodes_and_prefills
    from vllm.v1.attention.backends.mla.indexer import split_prefill_chunks

    num_reqs = common_attn_metadata.num_reqs
    num_tokens = common_attn_metadata.num_actual_tokens

    query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
    
    # 调用 split_decodes_and_prefills (测试时会被 Mock)
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = \
        split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.reorder_batch_threshold)

    # 这里的 Assert 是我们要验证的逻辑之一
    assert num_decodes + num_prefills == num_reqs
    assert num_decode_tokens + num_prefill_tokens == num_tokens

    prefill_metadata = None
    if num_prefills > 0:
        # 调用 split_prefill_chunks (测试时会被 Mock)
        chunk_seq_ids = split_prefill_chunks(
            common_attn_metadata.seq_lens_cpu,
            self.max_prefill_buffer_size,
            num_decodes,
        )
        chunks = [
            self.build_one_prefill_chunk(
                reqs_start, reqs_end, query_start_loc_cpu,
                common_attn_metadata.seq_lens_cpu,
                common_attn_metadata.block_table_tensor)
            for reqs_start, reqs_end in chunk_seq_ids
        ]
        prefill_metadata = DeepseekV32IndexerPrefillMetadata(chunks=chunks)

    decode_metadata = None
    if num_decodes > 0:
        # 使用真实 Tensor 操作
        torch.diff(common_attn_metadata.query_start_loc[:num_decodes + 1],
                   out=self.decode_lens_buffer[:num_decodes])
        decode_lens = self.decode_lens_buffer[:num_decodes]
        decode_lens_cpu = torch.diff(
            common_attn_metadata.query_start_loc_cpu[:num_decodes + 1])

        requires_padding = (decode_lens_cpu.max()
                            > decode_lens_cpu.min()).item()

        decode_metadata = DeepSeekV32IndexerDecodeMetadata(
            block_table=common_attn_metadata.
            block_table_tensor[:num_decodes, ...],
            seq_lens=common_attn_metadata.seq_lens[:num_decodes],
            seq_lens_cpu=common_attn_metadata.seq_lens[:num_decodes].cpu(),
            decode_lens=decode_lens,
            requires_padding=requires_padding,
            schedule_metadata=self.scheduler_metadata_buffer,
        )

    attn_metadata = DeepseekV32IndexerMetadata(
        seq_lens=common_attn_metadata.seq_lens,
        seq_lens_cpu=common_attn_metadata.seq_lens.cpu(),
        num_reqs=common_attn_metadata.num_reqs,
        max_query_len=common_attn_metadata.max_query_len,
        max_seq_len=common_attn_metadata.max_seq_len,
        num_actual_tokens=common_attn_metadata.num_actual_tokens,
        query_start_loc=common_attn_metadata.query_start_loc,
        slot_mapping=common_attn_metadata.slot_mapping,
        head_dim=128,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        prefill=prefill_metadata,
        decode=decode_metadata,
    )
    return attn_metadata

# =======================
# 3. Fixtures
# =======================

@pytest.fixture(scope="module")
def device():
    """自动检测真实设备"""
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available(): return torch.device("xpu")
    pytest.skip("No CUDA or XPU device available")

@pytest.fixture
def builder_mock(device):
    builder = MagicMock()
    builder.reorder_batch_threshold = 1
    builder.max_prefill_buffer_size = 8192
    # [Real Tensor] 使用真实设备上的 Tensor，确保 torch.diff 等算子正常运行
    builder.decode_lens_buffer = torch.zeros(256, dtype=torch.int32, device=device)
    builder.scheduler_metadata_buffer = torch.zeros(10, dtype=torch.int32, device=device)
    builder.build_one_prefill_chunk.side_effect = lambda start, end, *args: f"chunk_{start}_{end}"
    return builder

@pytest.fixture
def common_metadata_mock(device):
    # 我们使用 MagicMock 来模拟 CommonAttentionMetadata 对象
    # 但里面的数据字段使用真实的 Tensor
    meta = MagicMock()
    meta.max_query_len = 10
    meta.max_seq_len = 128
    
    # 默认值，将在具体的 test case 中被覆盖以匹配 mock 的 split 结果
    meta.num_reqs = 0 
    meta.num_actual_tokens = 0 
    
    meta.slot_mapping = torch.arange(1024, dtype=torch.long, device=device)
    meta.seq_lens = torch.tensor([10, 20, 30, 40], dtype=torch.int32, device=device)
    meta.seq_lens_cpu = meta.seq_lens.cpu()
    meta.query_start_loc = torch.tensor([0, 1, 2, 17, 42], dtype=torch.int32, device=device)
    meta.query_start_loc_cpu = meta.query_start_loc.cpu()
    meta.block_table_tensor = torch.zeros((4, 16), dtype=torch.int32, device=device)
    
    return meta

# =======================
# 4. Test Cases
# =======================

def test_kunlun_build_mixed_batch(builder_mock, common_metadata_mock, device):
    """
    测试混合 Batch (Decode + Prefill)。
    使用 patch 拦截真实的 vllm 工具函数调用。
    """
    # [Patch] 拦截 split_decodes_and_prefills
    # 参数: (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens)
    with patch("vllm.v1.attention.backends.utils.split_decodes_and_prefills", return_value=(2, 2, 2, 40)), \
         patch("vllm.v1.attention.backends.mla.indexer.split_prefill_chunks", return_value=[(2, 3), (3, 4)]):
        
        # [Sync] 确保 Metadata 与 Mock 返回值一致，通过 Assert 检查
        common_metadata_mock.num_reqs = 4  # 2 + 2
        common_metadata_mock.num_actual_tokens = 42 # 2 + 40
        
        # Execute
        metadata = kunlun_build(builder_mock, 0, common_metadata_mock)
        
        # Verification
        assert isinstance(metadata, DeepseekV32IndexerMetadata)
        assert metadata.num_decodes == 2
        assert metadata.num_prefills == 2
        
        # Decode Check
        assert metadata.decode is not None
        # query_start_loc 前三个是 [0, 1, 2], diff 结果应为 [1, 1]
        expected_lens = torch.tensor([1, 1], dtype=torch.int32, device=device)
        assert torch.equal(metadata.decode.decode_lens, expected_lens)
        assert metadata.decode.requires_padding is False
        
        # Prefill Check
        assert metadata.prefill is not None
        assert len(metadata.prefill.chunks) == 2
        
        print("\nMixed Batch Test Passed!")

def test_kunlun_build_decode_padding(builder_mock, common_metadata_mock, device):
    """测试 Decode Padding 逻辑 (真实 Tensor 计算)"""
    # 构造不均匀 Query: [0, 1, 3] -> lens [1, 2]
    common_metadata_mock.query_start_loc = torch.tensor([0, 1, 3], dtype=torch.int32, device=device)
    common_metadata_mock.query_start_loc_cpu = common_metadata_mock.query_start_loc.cpu()
    
    with patch("vllm.v1.attention.backends.utils.split_decodes_and_prefills", return_value=(2, 0, 3, 0)):
        
        # [Sync]
        common_metadata_mock.num_reqs = 2
        common_metadata_mock.num_actual_tokens = 3
        
        metadata = kunlun_build(builder_mock, 0, common_metadata_mock)
        
        assert metadata.decode is not None
        expected_lens = torch.tensor([1, 2], dtype=torch.int32, device=device)
        assert torch.equal(metadata.decode.decode_lens, expected_lens)
        # 1 != 2 -> Requires Padding
        assert metadata.decode.requires_padding is True
        print("\nDecode Padding Test Passed!")

def test_kunlun_build_prefill_only(builder_mock, common_metadata_mock, device):
    """测试纯 Prefill 场景"""
    with patch("vllm.v1.attention.backends.utils.split_decodes_and_prefills", return_value=(0, 4, 0, 100)), \
         patch("vllm.v1.attention.backends.mla.indexer.split_prefill_chunks", return_value=[(0, 4)]):
        
        # [Sync]
        common_metadata_mock.num_reqs = 4
        common_metadata_mock.num_actual_tokens = 100
        
        metadata = kunlun_build(builder_mock, 0, common_metadata_mock)
        
        assert metadata.num_decodes == 0
        assert metadata.decode is None
        assert metadata.prefill is not None
        assert len(metadata.prefill.chunks) == 1
        
        print("\nPrefill Only Test Passed!")