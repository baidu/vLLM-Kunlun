# SPDX-License-Identifier: Apache-2.0
"""Q38 XPU PORT: PLE conv-state 的行 gather/scatter。

## 问题（算子问题 #6）

PLE 的 conv_state 是整块 KV cache 的一个**非连续视图**，实测布局：

    shape=(152889, 10240, 9)  stride=(131072, 1, 10240)  contiguous=False
    dtype=bfloat16  numel=1.409e10  逻辑大小 26.25 GiB
    每行逻辑元素 92160 个，但行间跨度 131072（行后有 38912 个元素的填充/他用区）

对它直接调 ``index_select`` / ``index_copy_`` 会 OOM（实测
"Tried to allocate 27.20 GiB"）。根因在 torch_xmlir 的 aten 包装层
``python/torch_xmlir/csrc/aten_capture/eager_customized/index_select.cpp``::

    at::Tensor& CUDANativeFunctions::index_select_out(...) {
        at::Tensor tmp_self_tensor = self.contiguous();   // <= 无条件整块物化

下游 kernel ``xdnn_pytorch::index_select`` 只按 ``self.sizes`` 取数、不认 stride，
所以包装层必须先把输入拍连续。输入本来连续时这句是 no-op；输入是大 cache 的
非连续视图时，它就变成一次整块拷贝。

## device 路径（``Q38_PLE_STATE_DEVICE=1``，默认）

不去动那个 26 GiB 的非连续视图，改成在**连续的 ``(n, stride0)`` 视图**上按行取数：

    full = src.as_strided((n, stride0), (stride0, 1), off)   # 这个视图是连续的
    rows = full.index_select(0, idx)                         # 整行(含行后填充)一起取
    rows[:, :row_numel] -> reshape/permute 回逻辑形状            # 填充部分切掉

写回是"整行读出 -> 覆盖行内前 row_numel 个元素 -> 整行写回"，
填充区被原样读出再原样写回，内容不变（单测用哨兵值校验）。

``full`` 连续 => ``self.contiguous()`` 成为 no-op，只分配结果大小；
整行是连续块拷贝，比按元素偏移 gather 快一个量级。全程无 host 同步，可进 CUDA Graph。

实测（n=4096、8 行、bf16，输出 1.41 MB）：

    原生 src.index_select      峰值 +1441.41 MB   <= 按整块输入分配
    本实现（整行 index_select）  峰值 +   3.41 MB   延迟 66.6 us
    host 逐行兜底              峰值 +   2.81 MB   延迟 163.9 us

## host 路径（``Q38_PLE_STATE_DEVICE=0``，退路，兼容旧行为）

indices 落 host 后逐行拷贝。正确，但有一次 device→host 同步，且
``torch.stack`` 会把捕获时刻的行地址写死进 CUDA Graph：单测里换掉 indices
再 replay，host 路径结果不跟随（静默算错），device 路径跟随。

两条路径逐位相同，见 ut_ple_conv_state_gather.py。
"""

import os

import torch


def row_plan(src):
    """判断 dim0 的每一行是否是存储里的一个连续块，并给出内存序信息。

    返回 ``(stride0, row_numel, mem_shape, perm)``，做不到返回 None。
    ``perm`` 是 dim1.. 按 stride 降序（= 内存序）排出来的逻辑维编号。
    """
    if src.dim() < 2:
        return None
    perm = sorted(range(1, src.dim()), key=lambda d: -src.stride(d))
    expected = 1
    for d in reversed(perm):
        if src.stride(d) != expected:
            return None
        expected *= int(src.shape[d])
    row_numel = expected
    if row_numel <= 0:
        return None
    stride0 = src.stride(0)
    if stride0 < row_numel:
        return None
    mem_shape = tuple(int(src.shape[d]) for d in perm)
    return stride0, row_numel, mem_shape, tuple(perm)


def _storage_numel(src):
    try:
        return src.untyped_storage().nbytes() // src.element_size()
    except Exception:
        return src.storage().size()


def _full_view(src, plan):
    """连续的 ``(n, stride0)`` 行矩阵视图：行长等于行间跨度，所以它是连续的。

    视图末行可能越过存储末尾（当 ``storage_offset > 0`` 且存储恰好只装下
    n 行时，例如 GDN 的 ssm_state 视图 offset=2560）。此时把视图起点向前
    平移到存储内，行内数据改从列偏移 ``col0 = offset - v_off`` 处取。

    返回 ``(view, col0)``；无法构造在界内的视图时返回 ``(None, 0)``。
    """
    stride0, row_numel = plan[0], plan[1]
    n = int(src.shape[0])
    off = src.storage_offset()
    total = _storage_numel(src)
    v_off = off if off + n * stride0 <= total else total - n * stride0
    col0 = off - v_off
    if v_off < 0 or col0 < 0 or col0 + row_numel > stride0:
        return None, 0
    return src.as_strided((n, stride0), (stride0, 1), v_off), col0


def _safe_idx(src, idx1d):
    """把索引钳进 [0, n)。

    上游已把 ``NULL_BLOCK_ID`` 重映射到 slot 0，但 CUDA Graph 捕获用的 dummy run
    里 ``state_indices`` 可能是未初始化的垃圾值（实测出现过
    ``index 9205498103099064256``），既不等于 ``NULL_BLOCK_ID`` 也过不了上游的
    ``torch.where``。原生 GPU kernel 不查越界会直接读脏内存；这里钳一下，
    避免越界读写打到无关显存。只有 8~16 个元素，开销可忽略。
    注意必须 out-of-place：``idx1d`` 是调用方还要用的 ``state_indices``。
    """
    return torch.clamp(idx1d.to(torch.int64), 0, int(src.shape[0]) - 1)


def _use_device_path():
    """默认走 device 路径：eager 下与旧兜底逐位相同、快 2 倍多，且解除了开图阻塞。
    出问题时置 ``Q38_PLE_STATE_DEVICE=0`` 退回旧的 host 逐行兜底。"""
    return os.environ.get("Q38_PLE_STATE_DEVICE", "1") == "1"


def _to_logical_axes(ndim, perm):
    """gathered (k, *mem_shape) -> 逻辑 (k, *src.shape[1:]) 的 permute 轴序。"""
    return (0,) + tuple(perm.index(i) + 1 for i in range(1, ndim))


def gather_rows(src: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """等价于 ``src.index_select(0, indices)``，但不对大张量整体做拷贝。"""
    idx1d = indices.reshape(-1)
    if _use_device_path():
        plan = row_plan(src)
        if plan is not None:
            _, row_numel, mem_shape, perm = plan
            k = int(idx1d.numel())
            if k == 0:
                return src.new_empty((0, *src.shape[1:]))
            full, col0 = _full_view(src, plan)
            if full is not None:
                rows = torch.index_select(full, 0, _safe_idx(src, idx1d))
                got = rows[:, col0 : col0 + row_numel].contiguous().view(k, *mem_shape)
                return got.permute(*_to_logical_axes(src.dim(), perm)).contiguous()
    idx = idx1d.tolist()
    if not idx:
        return src.new_empty((0, *src.shape[1:]))
    return torch.stack([src[int(i)] for i in idx], dim=0)


def scatter_rows_(
    dst: torch.Tensor, indices: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    """等价于 ``dst.index_copy_(0, indices, values)`` 的行级写回。"""
    idx1d = indices.reshape(-1)
    if _use_device_path():
        plan = row_plan(dst)
        if plan is not None:
            _, row_numel, _, perm = plan
            k = int(idx1d.numel())
            if k == 0:
                return dst
            full, col0 = _full_view(dst, plan)
            if full is not None:
                idx = _safe_idx(dst, idx1d)
                # 整行读出 -> 覆盖行内数据段 -> 整行写回，
                # 行内填充区原样读出再原样写回，内容不变。
                rows = torch.index_select(full, 0, idx)
                rows[:, col0 : col0 + row_numel] = (
                    values.to(dst.dtype).permute(0, *perm).reshape(k, row_numel)
                )
                full.index_copy_(0, idx, rows)
                return dst
    for pos, i in enumerate(idx1d.tolist()):
        dst[int(i)].copy_(values[pos])
    return dst
