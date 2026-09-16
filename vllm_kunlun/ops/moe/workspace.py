"""Centralized metadata and data workspace layouts for Kunlun MoE."""

from dataclasses import dataclass

import torch
from vllm.v1.worker.workspace import current_workspace_manager

MOE_BLOCK_STATISTIC_ROWS = 12


@dataclass(frozen=True)
class MoeMetadata:
    """Typed views over the single int32 metadata allocation."""

    buffer: torch.Tensor
    num_experts: int
    num_rows: int

    @staticmethod
    def numel(num_experts: int, num_rows: int) -> int:
        return (
            MOE_BLOCK_STATISTIC_ROWS * num_experts
            + num_experts
            + num_experts
            + 1
            + num_rows
        )

    @property
    def block_statistic(self) -> torch.Tensor:
        end = MOE_BLOCK_STATISTIC_ROWS * self.num_experts
        return self.buffer[:end].view(MOE_BLOCK_STATISTIC_ROWS, self.num_experts)

    @property
    def expert_m(self) -> torch.Tensor:
        start = MOE_BLOCK_STATISTIC_ROWS * self.num_experts
        return self.buffer[start : start + self.num_experts]

    @property
    def sorted_tokens_num_lod(self) -> torch.Tensor:
        start = (MOE_BLOCK_STATISTIC_ROWS + 1) * self.num_experts
        return self.buffer[start : start + self.num_experts + 1]

    @property
    def sorted_tokens_idx(self) -> torch.Tensor:
        start = (MOE_BLOCK_STATISTIC_ROWS + 2) * self.num_experts + 1
        return self.buffer[start : start + self.num_rows]


@dataclass(frozen=True)
class MoeWorkspaces:
    """All scratch buffers for one MoE invocation; returned output never aliases them."""

    a: torch.Tensor
    b: torch.Tensor
    metadata: MoeMetadata
    q: torch.Tensor | None = None
    scales: torch.Tensor | None = None

    @staticmethod
    def _expanded_numel(num_rows: int, hidden_size: int, use_sorted: bool) -> int:
        return num_rows * hidden_size if use_sorted else 0

    @classmethod
    def fp(
        cls,
        *,
        num_rows: int,
        num_experts: int,
        hidden_size: int,
        gate_up_size: int,
        output_size: int,
        dtype: torch.dtype,
        use_sorted: bool,
        fuse_swiglu: bool,
        activation_allocates: bool,
    ) -> "MoeWorkspaces":
        inter_size = gate_up_size // 2
        activated = 0 if activation_allocates else num_rows * inter_size
        expert_output = num_rows * output_size
        if fuse_swiglu:
            a_numel, b_numel = expert_output, activated
        else:
            a_numel = activated
            b_numel = max(num_rows * gate_up_size, expert_output)
        a_numel = max(a_numel, cls._expanded_numel(num_rows, hidden_size, use_sorted))
        metadata_numel = MoeMetadata.numel(num_experts, num_rows)
        a, b, metadata = current_workspace_manager().get_simultaneous(
            ((a_numel,), dtype),
            ((b_numel,), dtype),
            ((metadata_numel,), torch.int32),
        )
        return cls(a, b, MoeMetadata(metadata, num_experts, num_rows))

    @classmethod
    def int8(
        cls,
        *,
        num_rows: int,
        num_experts: int,
        hidden_size: int,
        gate_up_size: int,
        output_size: int,
        dtype: torch.dtype,
        use_sorted: bool,
    ) -> "MoeWorkspaces":
        inter_size = gate_up_size // 2
        expanded = cls._expanded_numel(num_rows, hidden_size, use_sorted)
        metadata_numel = MoeMetadata.numel(num_experts, num_rows)
        a, b, q, scales, metadata = current_workspace_manager().get_simultaneous(
            ((max(expanded, num_rows * inter_size),), dtype),
            ((max(num_rows * gate_up_size, num_rows * output_size),), dtype),
            ((num_rows * max(hidden_size, inter_size),), torch.int8),
            ((num_rows,), torch.float32),
            ((metadata_numel,), torch.int32),
        )
        return cls(a, b, MoeMetadata(metadata, num_experts, num_rows), q, scales)

    @staticmethod
    def view(
        buffer: torch.Tensor, rows: int, width: int, *leading: int
    ) -> torch.Tensor:
        return buffer[: rows * width].view(*leading, width)

    @staticmethod
    def output(num_tokens: int, output_size: int, like: torch.Tensor) -> torch.Tensor:
        return torch.empty(
            (num_tokens, output_size), dtype=like.dtype, device=like.device
        )
