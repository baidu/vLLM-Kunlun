# 05 · 设备、流、显存与运行时

[← 首页](README.md) · [← 04 Dispatcher 劫持](04-dispatch-hijack.md) · [→ 06 融合算子](06-custom-ops-nn.md)

本页最重要的一条结论，先说：

> **`torch.cuda.*` 基本没有被重映射。它是原封不动的官方 PyTorch，跑在 CUDA ABI 垫片上。**
> `torch_xmlir.xpu.*` 是一套**平行的、另起名字的**设备 API，没有被拼接进 `torch.cuda`。

运行时验证 **[已验证]**：

```
cuda avail True   count 8
torch.cuda.Stream       -> <class 'torch.cuda.streams.Stream'>      # 官方
torch.cuda.Event        -> <class 'torch.cuda.streams.Event'>       # 官方
torch.cuda.memory_stats -> torch/cuda/memory.py                     # 官方
torch.cuda.CUDAGraph    -> <class 'torch.cuda.graphs.CUDAGraph'>    # 官方
get_device_name(0)       -> 'GPU'          # 来自垫片，不是 torch_xmlir
get_device_capability(0) -> (8, 6)         # 同上
```

两套 API 底下是**同一个** device context，互相可见 **[已验证]**：

```
torch_xmlir.xpu.set_device(3)  →  torch.cuda.current_device() == 3
torch.cuda.set_device(5)       →  torch_xmlir.xpu.current_device() == 5
torch.zeros(4, device='cuda').device  →  cuda:5
```

## `torch_xmlir.xpu` 设备 API

`$PKG/xpu/device.py`（292 行），全部薄封装 `_XMLIRC` **[源码]**：

| 函数 | 行 | 底层 pybind 调用 |
|---|---|---|
| `is_available()` | 109 | `_xpu_get_devices_number() > 0` |
| `device_count()` | 113 | `_xpu_get_devices_number()` |
| `current_device()` | 117 | `_xpu_get_device_id()` |
| `set_device(d)` | 121 | `_xpu_set_default_device(str)` |
| `synchronize()` | 132 | `_xpu_synchronize_default_stream()` |
| `get_device_properties(d)` | 137 | `_xpu_get_device_properties(d)` |
| `get_device_name(d)` | 146 | 上者的 `.name` |
| `get_device_capability(d)` | 150 | 上者的 `.major`/`.minor` |

`set_device` 有个值得注意的分支 —— 写进去的**设备类型串取决于运行模式**：

```python
# $PKG/xpu/device.py:121-129
def set_device(device):
    device_index = _get_device_index(device, optional=False)
    device_type = "xla"
    if param_scope.xacc.eager("false") == "true":
        device_type = "xpu"
    device_str = device_type + ":{}".format(device_index)
    torch_xmlir._XMLIRC._xpu_set_default_device(device_str)
    device_str = torch_xmlir._XMLIRC._xpu_get_default_device()
    return torch.device(device_str)
```

默认（lazy/XLA 模式）产出 `xla:N`；`hyperparameter` 的 `param_scope` 里
`xacc.eager == "true"` 时产出 `xpu:N`。运行时确认 `torch_xmlir.xpu.set_device(1)`
返回 `device(type='xla', index=1)`。**[已验证]**

而顶层的 `torch_xmlir.device()`（`$PKG/__init__.py:228-268`）**硬断言** `device_type == "xpu"`
（`:264`），返回 `device(type='xpu', index=0)`。两个同名概念不一致，注意别混。

`_XMLIRC._xpu_get_device_properties(0)` 的 repr 对自己的不匹配很坦诚 **[已验证]**：

```
_XpuDeviceProperties(name='KL3', major=0(N/A for XPU), minor=0(N/A for XPU),
                     total_memory=98304MB, multi_processor_count=12)
```

所以 `torch_xmlir.xpu.get_device_capability()` 返回 `(0, 0)`，
而 `torch.cuda.get_device_capability()` 返回 `(8, 6)`。**依赖 capability 做分支的库要小心。**

## `torch.cuda` 里真正被换掉的三个

`plugins/torch/__init__.py` 无条件替换的只有 **[源码]**：

| 符号 | 行 | 换成 |
|---|---|---|
| `torch.cuda._sleep`、`torch._C._cuda_sleep` | 26-33 | `_XMLIRC._xpu_sleep` |
| `torch.cuda._raw_device_count_nvml` | 110-113 | `mock_torch.mock_raw_device_count_nvml`（ctypes → NVML 垫片） |
| `torch.cuda.utilization` | 114-117 | `mock_torch.mock_torch_utilization` |

其余 `torch.cuda.*` 全是原版。

## 流与事件：Python 侧完全没有实现

全包搜 `class .*Stream` / `class .*Event` 只命中 docstring 和一个 benchmark 用的
`FakeCudaEvent`（`plugins/torchbenchmark/mock_torch_benchmark.py:125`）。**[已验证]**

映射发生在 C/C++。`XMLIR Python extension` 引用了这些 CUDA runtime 符号 **[.so 内]**：

```
cudaStreamCreate / Destroy / Query / Synchronize / WaitEvent
cudaEventCreate / CreateWithFlags / Destroy / ElapsedTime / Query / Record / Synchronize
cudaIpcGetEventHandle / cudaDeviceSynchronize / cudaHostPointerGetAttributes
```

以及 XPU 原生的 `xpu_stream_destroy`、`xpu_event_create/destroy/record/wait`、
`stream_synchronize`。这些最终由 CUDA ABI 兼容库解析到 Kunlun XPU 运行时。
**翻译逻辑完全在编译产物中，不可见。**

Python 侧唯一可见的同步原语是 `_XMLIRC._xpu_synchronize_default_stream()`，
暴露为 `xpu/device.py:132 synchronize()` 和 `:291 xpu_synchonize_default_stream()`
（注意后者拼写少个 `r`，是源码里的笔误）。

相关 `.so` 内环境变量 **[.so 内]**：`XMLIR_DIST_SINGLETON_STREAM`、
`XMLIR_DIST_USE_DEFAULT_STREAM`、`XMLIR_OP_USE_DEFAULT_STREAM`、`XPU_SUPPORT_IPC_EVENT`。

## CUDA Graph

`$PKG/xpu/graphs.py`（143 行）提供 `CUDAGraph`（`:8`）和 `graph`（`:80`）。

```python
# $PKG/xpu/graphs.py:8-36
class CUDAGraph(torch_xmlir._XMLIRC._CUDAGraph):
    def __new__(cls):
        return super().__new__(cls)
    def capture_begin(self, pool=None, capture_error_mode="global"):
        super().capture_begin(pool=pool, capture_error_mode=capture_error_mode)
```

所有方法都是纯 docstring 转发，Python 里没加任何 XPU 特有逻辑。
`_CUDAGraph` 绑的是 **PyTorch 自己的 `at::cuda::CUDAGraph`** —— `XMLIR Python extension` 里有
mangled 符号 `_ZN2at4cuda9CUDAGraph13capture_beginESt4pairIyyE21cudaStreamCaptureMode` 等。
即 graph capture 是官方 ATen 代码跑在昆仑 `cudaStreamCapture*` 垫片上。**[已验证]**

`graph`（`:80`）是官方 `torch.cuda.graph` 的逐字拷贝，内部还是用真的
`torch.cuda.Stream()`（`:116`）、`torch.cuda.synchronize()`、`torch.cuda.empty_cache()`（`:127-138`）。

**默认不安装**：

```python
# $PKG/symbrewrite/plugins/torch/__init__.py:157-164
use_xpu_graph = int(os.getenv("XMLIR_FORCE_USE_XPU_GRAPH", 0))
if use_xpu_graph:
    import torch_xmlir.xpu.graphs as xpu_graphs
    symbol_replacements += [
        SYMBOL_REPLACE("torch.cuda.CUDAGraph", xpu_graphs.CUDAGraph),
        SYMBOL_REPLACE("torch.cuda.graph", xpu_graphs.graph),
    ]
```

而且 `xpu/graphs.py` 不在 `xpu/__init__.py` 的星号导入里，要显式 import。**[源码]**

## 显存：`xpu/memory.py` 在本构建整体不可用

这是本次分析里最值得警惕的发现之一。`$PKG/xpu/memory.py`（441 行）声明了一整套
CUDA 形状的显存 API 并写了 `__all__`（`:11-30`），但**底层 pybind 符号有四个不存在** **[已验证]**：

```
empty_cache      → AttributeError: module 'torch_xmlir._XMLIRC' has no attribute '_xpu_empty_cache'
memory_stats     → AttributeError: ... '_xpu_memory_stats'
memory_snapshot  → AttributeError: ... '_xpu_memory_snapshot'
mem_get_info     → AttributeError: ... '_mem_get_info'
```

对包内 23 个 `.so` 逐一 grep `_xpu_empty_cache` / `_xpu_memory_stats` / `_mem_get_info` /
`_xpu_set_per_process_memory_fraction` / `_xpu_memory_snapshot`：**零命中**。

因为 `memory_allocated`（`:223`）、`max_memory_allocated`、`memory_reserved`、
`memory_cached`、`memory_summary`（`:308`）等全部经由 `memory_stats` →
`memory_stats_as_nested_dict`（`:145`），**整个模块都会抛 `AttributeError`**。

好消息是这只在一个开关后面才被接进 `torch.cuda`：

```python
# $PKG/symbrewrite/plugins/torch/mock_allocator.py:6-11
XMLIR_DISABLE_CUDA_ALLOCATOR = os.environ.get("XMLIR_DISABLE_CUDA_ALLOCATOR") == "1"

def return_allocator_register():
    if XMLIR_DISABLE_CUDA_ALLOCATOR:
        allocator_apis = [ ... 14 个 SYMBOL_REPLACE ... ]
        return allocator_apis
    return []
```

**结论：不要设 `XMLIR_DISABLE_CUDA_ALLOCATOR=1`** —— 会把 14 个
`torch.cuda.memory_*` 指到这些坏掉的函数上。默认不设时，用的是**官方 PyTorch CUDA
caching allocator**，`PYTORCH_CUDA_ALLOC_CONF` 按官方语义正常生效
（torch_xmlir 的 Python 代码里完全没引用它）。**[已验证]**

`xpu/memory.py` 里唯一能用的是 `use_l3`（`:434-441`），一个把 `use_l3 = True`
推进 `param_scope` 的上下文管理器，供 C++ kernel 读取。

另有一条独立可用的路径：`core/xpu_model.py:1107 get_memory_info(device)` 调
`_XMLIRC._xpu_memory_info(str(device))`，返回 `{kb_free, kb_total}` —— 与坏掉的
`_mem_get_info` 是不同符号。**[静态推断]** 未实际调用验证。

### `_XMLIRC` 自己的 allocator

`$PKG/__init__.py:200` 调 `_XMLIRC._xpu_init_caching_allocator()`，这个符号**存在**。
行为在 C++ 里。从 `.so` 字符串表能看到它读的旋钮 **[.so 内]**：

| 变量 | 名字暗示 |
|---|---|
| `XMLIR_CACHING_ALLOC_ENABLED` | XMLIR caching allocator 总开关 |
| `XMLIR_D_XPU_L3_SIZE` | L3 scratch 大小 |
| `XMLIR_D_XPU_ENABLE_L3_SHARE` | L3 共享 |
| `XMLIR_F_XPU_RESERVED_L3_INT` | 预留 L3 字节数 |
| `XMLIR_F_XPU_RESERVED_WORKSPACE_INT` | 预留 workspace 字节数 |
| `XMLIR_LRU_CACHE_SIZE` | LRU cache 容量 |
| `XMLIR_MEMCPY_RETRY_SYNC` | memcpy 重试/同步行为 |
| `XMLIR_EMPTY_OP_INIT_ZERO` | `empty` 系列分配是否置零 |
| `XMLIR_DISK_CACHE_DIR` / `XMLIR_PERSISTENT_CACHE_ENABLED` | 编译图磁盘缓存 |

默认值与解析方式**未验证** —— 只是字符串表证据。

## L3 内存

昆仑芯有片上 L3 scratch，是这套栈里 GPU 没有的概念。三个接触点：

- `torch_xmlir.xpu.use_l3()` 上下文管理器（`xpu/memory.py:434`）
- `nn/linear.py:23` 的 `USE_L3` 环境变量：非空时把 legacy Linear 的输出 buffer 分配到 L3（`:49-51`）
- C++ 侧 `XMLIR_D_XPU_L3_SIZE` / `XMLIR_D_XPU_ENABLE_L3_SHARE` / `XMLIR_F_XPU_RESERVED_L3_INT`
- `plugins/torchbenchmark/test_torch_benchmark.py:41-43` 读伪造 memory-info 上的
  `totalL3Memory` / `usedL3Memory` / `freeL3Memory`

## XPU3 运行模式：会写硬件寄存器

`$PKG/_xpu3_run_mode.py`（107 行）通过两个 CLI 工具直接读写**硬件寄存器**，
切换 matmul/FP 流水的精度模式。

```python
# $PKG/_xpu3_run_mode.py:15-20
self._mode_map = {
    "TRAIN-BF16": "3",
    "TRAIN-FP16": "0",
    "INFER-BF16": "-1",
    "INFER-FP16": "-1",
}
```

两个 INFER 模式写的是同一个值，寄存器层面无法区分。**[源码]**

- 工具：运行时提供的寄存器读取与写入工具。分析环境确认这些工具存在并可被调用。**[已验证]**
- 寄存器：实现会写入多组硬件寄存器，具体地址和拓扑映射不在本文公开。
- 设备选择（`:56-64`）：默认 `0..7`；有 `LOCAL_RANK` 就只改那一张；
  否则按 `CUDA_VISIBLE_DEVICES` 的逗号数量

触发点在 `init()`：

```python
# $PKG/__init__.py:170-176
global xpu_run_mode
if xpu_run_mode is not None:
    assert isinstance(xpu_run_mode, str)
    xpu_run_mode = xpu_run_mode.upper()
    set_xpu3_run_mode(xpu_run_mode)
    check_xpu3_run_mode(xpu_run_mode)
```

**注意事项** **[源码]**：

1. 只有显式设置 `XPU_RUN_MODE` 才会写寄存器。
2. 这是**在 import 期修改多张卡的共享硬件状态**。同一节点上并发跑多个任务时，
   一个进程设 `XPU_RUN_MODE` 会影响所有选中的设备。
3. 拼错的模式名会被**静默忽略** —— `write()` 遇到未知 mode 只 warn 就 return（`:72-76`），
   `check()` 也 warn 后 return。所以 `XPU_RUN_MODE=TRAIN_BF16`（下划线而非连字符）不会报错，
   只是什么都没做。
4. `check()` 读回不匹配时抛 `RuntimeError("XPU run mode check failed.")`（`:92-93`）。
5. `_get_device` 声明时没有 `self` 也没有 `@staticmethod`，靠在类上访问才能工作（`:14`）。

关于「XPU3 vs P800」：该文件没有公开的代次分支逻辑。本文分析的运行时构建包含 P800 专用内核产物，因此 **该构建面向 P800，`_xpu3_run_mode.py` 是继承下来的 XPU3 代次寄存器接口**。**[静态推断]**

## `core/`：XLA 时代的化石

`core/`（2758 行）来自 torch_xla 血统，现在大部分是死的 **[源码]**：

| 文件 | 行数 | 状态 |
|---|---|---|
| `core/xpu_builder.py` | 1260 | XLA/HLO 风格图构建 DSL，~110 个方法，全部落到 `_XMLIRC._xpu_op_*` |
| `core/xpu_model.py` | 1191 | 运行时/分布式模块。**`mark_step` 在 `:678` 无条件 `raise NotImplementedError`**，`xpu_device` 在 `:184` 同样 raise |
| `core/functions.py` | 164 | autograd 感知的集合通信包装 + `nms` + `distributed_mm` |
| `core/xpu_op_registry.py` | 95 | `register(name, opfn)` 用户自定义算子注册，op 名前缀 `"xla::_op_"` |
| `core/xpu_env_vars.py` | 42 | 22 个 XRT/TPU 风格环境变量名常量，其中 **19 个无人消费**，纯 torch_xla 遗留 |

因为 `optimizer_step` 在 `:919`/`:947` 调 `mark_step`，**整条 lazy-tensor
`optimizer_step` 路径在本构建不可用**。

`core/` 里还活着的环境变量（经 `utils/utils.py:175-182` 的 `getenv_as` 读取）：

| 变量 | 位置 | 默认 | 含义 |
|---|---|---|---|
| `ALLREDUCE_FUSION` | `xpu_model.py:801` | `True` | all-reduce 前做梯度分桶 |
| `ALLREDUCE_BUCKET_SIZE` | `:802-804` | 32MB | 桶大小 |
| `ALLREDUCE_CHECK` | `:809,838` | `False` | 调试：拿 gloo CPU all-reduce 对账 |
| `XLA_SYNC_WAIT` | `:696,716` | `False` | 阻塞同步（`:716` 在 `get_xpu_data_ptr` 里是活的） |
| `XRT_SHARD_WORLD_SIZE` / `_ORDINAL` / `_LOCAL_ORDINAL` | `:113,129,145` | `1`/`0`/`-1` | 复制组 world size 与 ordinal |
| `RATE_TRACKER_SMOOTHING` | `:269` | `0.4` | `RateTracker` 的 EMA 系数 |

---

下一页：[06-custom-ops-nn.md](06-custom-ops-nn.md)
