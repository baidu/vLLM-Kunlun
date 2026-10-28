# 07 · 分布式：kccl / bccl / DTensor / DDP

[← 首页](README.md) · [← 06 融合算子](06-custom-ops-nn.md) · [→ 08 编译](08-compile-dynamo.md)

先说一个反直觉的事实：**`$PKG/distributed/` 这 3878 行 Python 里没有一行 XCCL/BKCL 代码，
也没有一个环境变量。** 通信后端完全在 C++，Python 侧的接入靠 symbrewrite 插件。

## NCCL → kccl/bccl 的三层改名

### 第一层：类别名

```python
# $PKG/symbrewrite/plugins/torch/__init__.py:50-60
SYMBOL_REPLACE("torch.distributed.ProcessGroupNCCL", torch.distributed.ProcessGroupXCCL),
SYMBOL_REPLACE("torch._C._distributed_c10d.ProcessGroupNCCL", torch.distributed.ProcessGroupXCCL),
SYMBOL_REPLACE("torch.distributed.distributed_c10d.ProcessGroupNCCL", torch.distributed.ProcessGroupXCCL),
```

三个 import 位置全覆盖。运行时确认
`torch.distributed.ProcessGroupNCCL is torch.distributed.ProcessGroupXCCL`。**[已验证]**

配套：`torch.distributed.is_nccl_available → lambda: True`（`:83-86`，运行时确认返回 `True`）。

### 第二层：backend 字符串双向翻译

```python
# $PKG/symbrewrite/plugins/torch/mock_torch.py:21-31
ENABLE_NEW_PG = os.environ.get("XMLIR_ENABLE_NEW_PG", "1").lower() in ("1","true","yes","on")

def _get_backend_name():
    """Get the backend name based on ENABLE_NEW_PG environment variable"""
    return "kccl" if ENABLE_NEW_PG else "bccl"
```

- `mock_init_process_group`（`:46-69`）和 `mock_new_group`（`:153-179`）
  把所有 str 参数里的 `"nccl"` 子串替换成 `"kccl"`/`"bccl"`
- `mock_get_backend`（`:72-77`）**反向**映射回 `"nccl"`，让用户代码里
  `if dist.get_backend() == "nccl"` 依然成立
- `mock_dist_name`（`:430-434`）把 `"custom"` 也报成 `"nccl"`
- `mock_barrier`（`:420-427`）在 custom backend 上裸调时注入 `BarrierOptions(device=cuda)`

**这是「兼容性伪装」的典型手法：对内改名，对外装作没改。** 好处是绝大多数框架代码
不用改；代价是日志和报错信息可能自相矛盾。**[源码]**

### 第三层：C++ 里的两套 ProcessGroup

XMLIR Python 扩展中同时存在**两个**实现 **[已验证]**：

| 代次 | C++ 类 | 注册函数 | backend 名 | 源码路径（编进 .rodata） |
|---|---|---|---|---|
| 新 | `torch_xmlir::kccl::ProcessGroupXCCL` | `torch_kccl_python_init` | `kccl` | 新 ProcessGroup 实现 |
| 旧 | `c10d::ProcessGroupXCCL` | `torch_xccl_python_init` | `bccl` | 旧 ProcessGroup 实现 |

对 `torch_kccl_python_init` 的反汇编可以看到注册序列 **[已验证]**：

```
ProcessGroupXCCL::cclInitOnce()
PyImport_ImportModule("torch.distributed")
  .attr("Backend").attr("register_backend")
  cpp_function(createKLProcessGroup)   # args: dist_backend_options, pg_options
  (... "kccl" ..., extended_api=...)
PyImport_ImportModule("torch._C").attr("_distributed_c10d")
  pybind11::class_<torch_xmlir::kccl::ProcessGroupXCCL>(scope, "ProcessGroupXCCL")
```

即：`torch.distributed.Backend.register_backend("kccl", createKLProcessGroup, extended_api=...)`，
然后把类绑成 `torch._C._distributed_c10d.ProcessGroupXCCL`。
`torch.distributed.ProcessGroupXCCL` 在 `import _XMLIRC` 之后即可解析
（运行时确认可访问），早于插件加载。**[已验证]**

集合通信组件中**没有** `ProcessGroupXCCL` 符号，说明它只提供底层集合通信能力。PyTorch 侧的集合操作位于 XMLIR 的编译组件中。**[已验证]**：

```
torch_xmlir::kccl::ProcessGroupXCCL::allreduce_impl(at::Tensor&, char const*, c10d::AllreduceOptions const&)
…::gather_impl(…, BKCLDataType, void*, c10::cuda::CUDAStream&, int, int)
…::scatter_impl / alltoall_base / recvAnysource
…::initXCCLComm / getXCCLComm / groupStart / endCoalescing / abortComms / runHookLoop / workEnqueue
```

注意签名里的 `c10::cuda::CUDAStream` —— **ABI 层面直接暴露了 CUDA dispatch key 劫持**。

`.so` 内的通信相关环境变量（名字推断，默认值未验证）**[.so 内]**：
`TORCH_XCCL_NAN_CHECK`、`TORCH_XCCL_AVOID_RECORD_STREAMS`、
`TORCH_XCCL_DEFAUTL_PG_TIMEOUT_MILSEC`（拼写错误在二进制里）、
`TORCH_XCCL_ENV_STORE_KEY_WITH_PG_NAME`、`TORCH_XCCL_ENV_WAIT_GDB`、
`XMLIR_XCCL_USE_ASYNC_OP`、`XMLIR_DIST_ASYNC_ISEND_IRECV`、
`XMLIR_DIST_CHECK`、`XMLIR_DIST_CHECK_INF_NAN(_ASYNC)`、`XMLIR_DIST_CHECK_INF_NAN_SAVE_DIR`。

## `distributed/tensor/`：真正在干活的部分

`$PKG/distributed/` 四个子树里，只有 `tensor/`（约 3200 行）是活的、承重的。
它的全部工作是：**让昆仑的 out-variant 自定义 kernel 在 DTensor dispatch 下存活**。

由 `$PKG/__init__.py:218` 的 `from torch_xmlir.distributed import tensor` 触发，
`tensor/__init__.py` 星号导入 5 个模块，导入即注册。**[源码]**

模块命名有误导性：`_random_ops.py` 放 dropout，`_pointwise_ops.py` 放 RoPE/SwiGLU，
`_tensor_ops.py` 放优化器和 MoE 路由。

### 三种注册机制

**(a) `register_op_strategy` —— 只有 1 个算子**（`_matrix_ops.py:23-24`）：

```python
@register_op_strategy(torch.ops.custom_ops.linear.default)
def custom_linear_op_strategy(op_schema: OpSchema) -> OpStrategy:
```

从**输出**placement 反推输入 placement（`:67-97`）：
`Shard(1)` out → column-parallel（`Replicate` × `Shard(1)`，bias `Shard(0)`）；
`Shard(0)` out → data-parallel；`Partial` out → row-parallel 需 reduce。
读 `op_schema.kwargs_schema["out"]`（`:56`），所以调用时必须带 `out=`。

**(b) `register_sharding` —— 只有 1 个算子**，且只允许 replicate（`_math_ops.py:16-23`）：

```python
@register_sharding(torch.ops.custom_ops.findmax.default)
def custom_findmax_sharding(input, max=None):
    acceptable_shardings = []
    replicate_only = ([Replicate()], [Replicate()])
    acceptable_shardings.append(replicate_only)
    return acceptable_shardings
```

**(c) 主力机制 —— 直接改 DTensor 的私有 handler dict（29 个算子）**：

```python
# $PKG/distributed/tensor/_math_ops.py:841-845
# Register all custom handlers into DTensor's OpDispatcher. (see _dispatch.py line 148).
_custom_handlers = dtensor.DTensor._op_dispatcher._custom_op_handlers
_custom_handlers[
    torch.ops.custom_ops.multi_tensor_l2norm.default
] = _multi_tensor_l2norm_handler
```

引用的 hook 是真实存在的：`torch/distributed/tensor/_dispatch.py:148-149`
`if op_call in self._custom_op_handlers: return self._custom_op_handlers[op_call](...)`。**[已验证]**

注册清单：

| 文件 | 行 | handler |
|---|---|---|
| `_matrix_ops.py` | 491-492 | `mha_varlen_fwd`、`mha_varlen_bwd` |
| `_pointwise_ops.py` | 1083-1111 | `fused_rope_forward/backward`、`fused_rope_thd_forward/backward`、`multi_tensor_scale`、`rotary_no_freqs_pos_emb_forward/backward`、`rotary_pos_emb`、`rotary_pos_emb_backward`、`swiglu_forward/backward`（11 个） |
| `_math_ops.py` | 843-868 | `multi_tensor_l2norm`、`rms_layer_norm(_backward)`、`scaled_softmax_forward/backward`、`softmax_with_mask(_backward)`、`te_layer_norm(_backward)`、`tensor_checksum`（10 个） |
| `_tensor_ops.py` | 490-501 | `optimizer_AdamW`、`combine_convert`、`dispatch_convert`、`multi_tensor_adam` |
| `_random_ops.py` | 173-178 | `dropout_add_forward`、`dropout_add_backward` |

### 为什么必须用 handler 而不是 strategy

因为这些 kernel 是 out-variant / void 返回的原地算子，**违反 PyTorch 的原地命名约定
（没有 `_` 后缀、返回 void）**，DTensor 默认传播机制无法把变更写回。
`_tensor_ops.py:24-32` 说得很直白：

```
This op mutates mom1, mom2, and param inplace, but since it does not
follow PyTorch's inplace naming convention (no '_' suffix, returns void),
DTensor's default dispatch cannot propagate inplace mutations back to
the original DTensors. This handler:
  1. Redistributes all data tensors to match param's placement
  2. Calls the underlying op on local tensors
  3. Reverse-redistributes inplace-mutated tensors back to their original
     placements when redistribution created new local tensors
```

每个 handler 都是同一套骨架，建在 `_tensor_utils.py`（72 行，
`find_mesh` / `redistribute_dt` / `writeback_output`）之上：
选锚点 placement（几乎总是取**输出** buffer 的）→ `redistribute_dt` 每个参数 →
在裸 local tensor 上调 `op_call` → `writeback_output` 反向 redistribute 并 `copy_` 回去。

### 切分约束（重要）

每个 kernel 的维度约束都写在 docstring 里，不满足时回落到 `Replicate` **[源码]**：

- **RoPE**：head_dim 永不可切（`_pointwise_ops.py:663-669`）
- **softmax / RMSNorm**：最后一维永不可切（`_math_ops.py:347-353`, `178-190`）
- **`mha_varlen_fwd`：三个维度全都不能切**，原因写得异常坦诚（`_matrix_ops.py:167-173`）：

```
Sharding constraints:
  - T (dim 0): CANNOT be sharded — lod_seqlens index into T
  - D (dim 2): CANNOT be sharded — dot product couples all D elements
  - H (dim 1): CANNOT be sharded — the kernel writes softmax_lse using a
               global head-index offset baked into its accumulation logic;
               running with a local head_num produces incorrect lse values
               for any rank whose local heads don't start at index 0.
```

- **`dispatch_convert`**：kernel 里有全局 running counter，强制全 replicate（`_tensor_ops.py:216-228`）

**用 DTensor + 昆仑融合算子做张量并行时，这张约束表是必读的** ——
很多你以为能切的维度其实会被静默 replicate，性能不符合预期时先查这里。

31 个引用到的 `custom_ops` 算子已确认全部存在于自定义算子编译组件中。**[已验证]**

## 集合通信调用点

`distributed/` 里只有 `dist.all_reduce`，6 处，全是「local kernel 算完后补一次 reduce」**[源码]**：

| 位置 | 做什么 |
|---|---|
| `_math_ops.py:118-121` | `multi_tensor_l2norm`：`square_()` → `all_reduce(SUM)` → `sqrt_()`（正确合并 per-rank 偏 L2 范数） |
| `_math_ops.py:300-301` | `rms_layer_norm_backward`：`dweight` 做 `SUM`，最后一维被 reshard 成 Replicate 时跳过 |
| `_math_ops.py:744-746` | `te_layer_norm_backward`：`dgamma`/`dbeta` 做 `SUM` |
| `_math_ops.py:828-836` | `tensor_checksum`：`has_nan_or_inf`/`max_val`/`max_history` 做 `MAX`，`min_*` 做 `MIN` |

**P2P 在 `distributed/` 里完全没有。** `send`/`recv`/`isend`/`irecv`/`batch_isend_irecv`
一个都不出现。P2P 依赖 C++ 的 `ProcessGroupXCCL::send/recv/recvAnysource` 实现，加一个 Python 补丁：

```python
# $PKG/symbrewrite/plugins/torch/mock_batch_isend_irecv.py:74
# XMLIR_DIST_DISABLE_ASYNC_ISEND_IRECV 为真时才替换
```

替换实现（`:16-69`）把 `_coalescing_manager` 桩成裸 `yield`（`:32-34`），逐个发起 op ——
**这是个刻意关闭 coalescing/overlap 的正确性逃生阀**，不是优化。默认 `"0"` 不启用。

## 通信-计算重叠

`distributed/` 里的 handler **全部同步**：redistribute → kernel → 阻塞 all_reduce → writeback。

重叠机制在别处：

- `nn/parallel/distributed.py` 有分桶 reducer（`_compute_bucket_assignment_by_size` @ `:549`，
  `_rebuild_buckets` @ `:892`），但 `:1009` 记了一条缺口：
  `# TODO: need support ProcessGroupXCCL::WorkXCCL::getFuture`，
  紧跟 `raise RuntimeError("xpu not support dist._register_comm_hook")` ——
  **DDP comm hook 在这个后端上不可用**。
- C++ 侧有 coalescing（`groupStart` / `endCoalescing` / `workEnqueue`）
- `xpytorch_import_hook.py:58-59` 把 `CUDA_DEVICE_MAX_CONNECTIONS` 拉到 8，
  注释明说是为了多流重叠（「llama 70b 多流优化至少需要 5 个流」）

## `distributed/` 的死代码（三个子树）

| 路径 | 行数 | 状态 |
|---|---|---|
| `distributed/distributed.py` | 130 | **broken**。`:119`/`:122` 调 `_XMLIRC._xpu_sync_multi` 和 `._xpu_reset_tokens`，两个符号在 XMLIR Python 扩展中都**不存在**。`_sync_params_and_buffers` 在 `__init__` 里无条件调用（`:57`），所以**构造这个类就会 AttributeError**。`torch_xmlir.nn.parallel.DistributedDataParallel` 也存在独立导入问题，见下一节。 |
| `distributed/utils.py` | 45 | torch `_pack_kwargs`/`_unpack_kwargs` 的拷贝，`__all__ = []`，无人引用 |
| `distributed/fsdp/` | 465 | **什么都没 patch**。`wrap.py` 是 PyTorch ~1.12/1.13 `fsdp.wrap` 的逐字拷贝（Facebook BSD 头 `:1-4`），签名是老 API（`unwrapped_params` 而非 `nonwrapped_numel`），与 torch 2.9 的 `_Policy` 体系不兼容。存在的唯一合理理由是让 `ParamExecOrderWrapPolicy` 还能 import —— 它已从上游 torch 移除。全 site-packages grep 无人引用；也找不到任何 `_flat_param` / `fully_shard` / `FullyShardedDataParallel` 符号 |

**即：本适配层没有对 FSDP 做任何专门适配。** FSDP 能不能用取决于官方实现在
CUDA 伪装下是否恰好工作，本次未验证。

`distributed/__init__.py:11` 还有个明显 bug：

```python
__all__ = ["DistributedDataParallel, tensor"]   # 一个含逗号的字符串，不是两个元素
```

所以 `from torch_xmlir.distributed import *` 一个名字都导不出来。**[源码]**

## `nn/parallel/`：XLA 时代的 DDP，静态推断已 import 不进来

`nn/parallel/`（2175 行）是 PyTorch 1.6~1.8 时代 `torch/nn/parallel/` 的移植，
目标设备类型硬编码为 `"xla"` —— 早于 CUDA key 劫持的产物。**[源码]**

适配要点：

- `data_parallel.py:99,184` `device_type = "xla"`；`comm.py:91` 断言 `inp.device.type == "xla"`
- 集合通信走 `_XMLIRC._broadcast` / `_scatter` / `_gather` 等（`comm.py:47,49,68,208,220,257,264`）
- **没有 D2D 路径**：`broadcast` 绕主机内存（`comm.py:43-44`
  `# TODO:When inter-card D2D is supported, remove the following line` / `tensor = tensor.cpu()`），
  `gather` 同（`:249`）
- `reduce_add`（`:71`）NCCL 快路径被注释掉（`:113-116`），改成串行 `+`/`add_`
- DDP 在 XPU 上拒绝 `gradient_as_bucket_view=True`（`distributed.py:395-402`），
  注释：「1:xpu not support is_alias_of；2:xpu view not share storage, can not save memory use」，
  且 `:411` 无条件强制 `False`
- bucket rebuild 被注释掉（`:640-642`）
- `_dump_DDP_relevant_env_vars`（`:40-92`）读约 38 个 `NCCL_*` 变量**仅用于打日志**，不影响行为
- 默认 `bucket_cap_mb=50`（`:347`），broadcast bucket 250MB（`:424`）

**[静态推断]** `distributed.py:15` 做 `import torch_xmlir.distributed as dist`，`:17` 调
`dist.is_available()`，但 `torch_xmlir/distributed/__init__.py` 只导出
`DistributedDataParallel` 和 `tensor` —— 没有 `is_available`、`Reducer`、`ReduceOp`、
`_compute_bucket_assignment_by_size`（`:549` 用到）、`_broadcast_coalesced_with_group`（`:1018` 用到）。
静态看 `import torch_xmlir.nn.parallel` 应在 `:17` 抛 `AttributeError`。
旁证：全 site-packages 无人 import 它，没有插件替换
`torch.nn.parallel.DistributedDataParallel`，`nn/__init__.py:9` 的 `DataParallel`
re-export 也被注释掉了。**因此两个遗留 DDP 实现均不应视为可用。官方 `torch.nn.parallel.DistributedDataParallel` 在 CUDA 兼容层下是否可用，本次未验证。**

---

下一页：[08-compile-dynamo.md](08-compile-dynamo.md)
