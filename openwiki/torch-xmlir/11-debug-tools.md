# 11 · 调试、诊断与工具链

[← 首页](README.md) · [← 10 环境变量](10-env-vars.md) · [→ 12 已知坑](12-pitfalls.md)

## `validate_aten`：CPU vs XPU 逐算子双跑

`$PKG/debug/validate_aten/`（约 1300 行）是**精度排查的主力工具**。
整个实现建在 `TorchDispatchMode` 上（`validate/contexts.py:14`）。
**全包没有用到 `TorchFunctionMode`。** **[已验证]**

### 四个 Mode

| 类 | 位置 | 作用 |
|---|---|---|
| `Validate` / `ValidateCoreContext` | `contexts.py:193` / `:417` | 双跑引擎 |
| `Check` | `contexts.py:329` | 只在 XPU 上跑，扫结果里的 NaN/Inf（`:397-399`），可选 ipdb + `exit(-1)` |
| `LoggingEachOp` | `contexts.py:63` | 逐算子打日志 |
| `InsertDelimiters` | `contexts.py:83` | 每算子打 `[START_SYMBOL]/[END_SYMBOL]`，可选注册 fwd/bwd module hook（`:125-130`） |

### 双跑流程

`ValidateCoreContext.__torch_dispatch__`（`contexts.py:441-600`）**[源码]**：

1. `OpStats.total_cnt` 自增；黑名单 / 全 CPU（`is_cpu_op`）/ meta（`is_meta_op`）跳过（`:457-473`）
2. 深拷贝输入到 CPU：`tree_map(lambda x: tensor_utils.to_cpu_device(x, is_clone=True), xpu_args)`（`:492-497`），
   另存一份 `saved_args` 供原地算子用（`:502-503`）
3. **CPU 参考跑把 fp16/bf16 输入升到 fp32**（`:510`，实现 `:174-189`）。
   注释给的两个理由：torch 2.0 有很多 aten 算子没有 fp16 版本；CPU fp16 本身也不准
4. 两边种同一个种子：`seed = fix_seed()` → `op(*xpu_args)` → `fix_seed(seed)` → `op(*cpu_args)`（`:516-536`）。
   CPU 侧抛异常会降级成 warning 并返回 XPU 结果（`:537-544`）
5. `comparer(...)` 按 config 的 `atol`/`rtol`/`equal_nan` 比对（`:546-553`）。
   比较顺序（`comparer.py:135-212`）：长度 → 非 tensor 逐元素 → shape（`:123`）→ dtype（`:128`）
   → `allclose_comparer`（`:43-107`），后者手算 `|cpu-xpu| - rtol*|xpu|` 以便报出
   **错误个数、最大差值下标、两侧具体值**
6. 不一致时：打日志，**并提示用 `XMLIR_D_FORCE_FALLBACK_STR='<op>'` 把该算子按到 CPU**（`:566-569`），
   可选打 Python 栈、可选 `codegen.gen_unittest(...)`（`:591`）、可选 ipdb（`:593-599`）
7. `Validate.__exit__` 打一张表：`算子 / 双跑次数 / CPU XPU结果一致次数 / 不一致次数`（`:221`, `:246-249`）

### 用法

**不由环境变量启用，纯 opt-in。** 全包 grep `validate_aten` / `ValidateCoreContext`
只命中它自己的文件 —— `debug/__init__.py` 是个裸 docstring，不 import 任何子模块。**[已验证]**

```python
from torch_xmlir.debug.validate_aten import Validate

with Validate(atol=1e-5, rtol=1e-5, dump=True, whitelist=["aten.add.Tensor"]):
    train_step()
```

构造参数（`contexts.py:193-207`）：`atol, rtol, equal_nan, pdb, dump, dump_dir,
blacklist, whitelist, log_level, print_config, print_traceback, print_results`。

- 黑名单在 `Context.__init__` 合并（`common/contexts.py:126-135`）：默认表 +
  一个 `"用户自定义"` 桶
- **`whitelist` 非空时完全覆盖黑名单**（`common/utils.py:8-16`）
- ⚠️ `dump_dir` 默认 `validate_dump_<YYYY_MM_DD_HH_MM_SS>`，**如果已存在会被 `shutil.rmtree`**（`common/contexts.py:151-153`）

默认黑名单（`validate/config.py:17-69`）按原因分组：非计算/搬运类算子、
未初始化内存类（`aten.empty*`）、RNG 类（CPU 与 XPU 算法不同）、
`aten.max_pool2d_with_indices[_backward]`。torch 2.0.x 额外加 `aten.gather.default`（`:72-75`）。

### 复现脚本生成

`codegen.gen_unittest`（`codegen.py:14-30`）在 `<dump_dir>/<id>-<op>/` 下写
`cpu_args.pkl`、`cpu_kwargs.pkl`、`test.py`。生成的 `test.py` 用
`torch.device("cuda")` 作为 XPU 设备（`codegen.py:76`，
`tensor_utils.py:71: to_xpu_device = functools.partial(to_target_device, torch.device("cuda"))`）
—— 与 CUDA key 劫持一致。

### 一处必要的 hack

`Check` 和 `ValidateCoreContext` 都临时 monkeypatch `torch.Tensor.record_stream`
以绕过 dispatch（`contexts.py:418-439`，wrapper `TorchFuncMockNoDispatch` @ `:252-266`
用 `_pop_mode_temporarily`），代码里引了 pytorch#94403。
这也解释了为什么 `$PKG/__init__.py:29` 的 `ignore_deregister_op` 唯独保留了
`aten::record_stream` —— 同一个问题的两端。

### 已知缺陷

`validate/contexts.py:2` 有 `from crypt import methods`、`:4` 有 `import nntplib`，
两个都没用到，且 `crypt`/`nntplib` 在新版 CPython 已废弃/移除（3.10 上没问题，3.13 会炸）。
`common/logger.py:1` import 了 `auto_param` 却不用。
`get_module_index()`（`contexts.py:78-80`）改 `MODULE_COUNTER` 但没写 `global`，会 `UnboundLocalError`；
`InsertDelimiters.__init__` 的 `elif isinstance(model, torch.nn.Module)` 分支（`:96-99`）
在未定义的 `module` 上迭代 —— 所以 **`InsertDelimiters(model=...)` 是坏的，
`InsertDelimiters()` 不带参数可用**。**[源码]**

## XDNN 调试等级

`$PKG/debug/dump_level.py`（117 行）**[源码]**：

`dump_level.py` 定义 `trace`、`checksum`、`dump` 和 `profiling` 四个调试等级，并将其映射到运行时位掩码。具体掩码值不在本文公开。

掩码经 `_XMLIRC._xpu_set_xdnn_debug_level` 推给运行时（`:46-47`）。两种用法：

```python
from torch_xmlir.debug.dump_level import dump_inner, dump_function

with dump_inner(mode="trace,dump"):     # :94-105
    step()

@dump_function(mode="checksum")          # :108-117
def f(): ...
```

另有 `set_xlog_level` / `unset_xlog_level`（走 `_XMLIRC._xpu_set_xlog_level`，
不是环境变量，虽然 docstring 里提到 `XLOG_LEVEL`）。

## profiler 工具链（插件，默认关）

`XFLAGS --enable profiler` 后启用。运行时足迹只有一处注册：

```python
# $PKG/symbrewrite/plugins/profiler/__init__.py:9-16
"resource": {
    "cuda symbol replace": [
        SYMBOL_REPLACE("torch.autograd.profiler_util.EventList.dump", dump),
    ]
},
```

⚠️ torch 2.9 的 `EventList` **有 `table` 但没有 `dump`**。所以这是一次**新增**而非替换 ——
`get_source` 抛 `AttributeError`，被 `symbol_rewrite.py:91-94` 的裸 except 吞掉，
结果 `__origin_dump = None`，`EventList.dump` 是新定义的。**[已验证]**

### 四个组件

| 文件 | 行数 | console_script | 作用 |
|---|---|---|---|
| `profiler/statistic_dump.py` | 436 | — | `dump` 的实现 |
| `profiler/csv_diff.py` | 498 | `XPU_Profiler_Comparison` | baseline vs current CSV 对比 |
| `profiler/trace_clean.py` | 314 | `XPU_Profiler_TraceClean` | 清理时间戳损坏的 Chrome trace 事件 |
| `profiler/trace_merger.py` | 361 | `XPU_Profiler_TraceMerge` | 多 rank trace 合并成一条时间线 |

**`statistic_dump.dump`** —— 把 `EventList.table` 的渲染器改成写文件。
`save_format` ∈ `{"txt","csv","all"}`（`:78` 校验），输出到
`output_dir or "torch_profiler_output"`，文件名
`profiler_<%Y%m%d_%H%M%S>_<pid>.{txt,csv}`（`:96-98`），且有写权限预检（`:85-94`）。
CUDA→XPU 适配在排序键里，把三套设备词汇统一到 torch 内部命名：

```python
# $PKG/symbrewrite/plugins/profiler/statistic_dump.py:116-122
key=lambda e: getattr(
    e,
    sort_by.replace("cuda", "device")
    .replace("xpu", "device")
    .replace("privateuse1", "device"),
),
```

默认 `sort_by="self_device_time_total"`、`row_limit=math.inf`（`:20-21`）。
表头用 `use_device.upper()`，所以列名是 "Self XPU" 而非 "Self CUDA"（`:160`）。
device time 汇总把 `CUDA` / `PrivateUse1` / `MTIA` 事件都算上，排除 user annotation（`:133-140`）。

**`XPU_Profiler_Comparison`** —— 按 `Name` 列做 key（`csv_diff.py:11`），
比 10 个指标（`DEFAULT_FIELDS`，`:13-24`）。注意**这些字段名仍是 CUDA 命名**
（`Self CUDA`、`CUDA total`、`CUDA time avg`），即它消费的是「CUDA 标签的那份 CSV」。
`parse_value`（`:75-100`）把 `12.3us`/`4.5ms`/`1.2s` 归一到微秒，`45.6%` 转 float。
`--min-change 0.2` 只保留 `Self CUDA` 变化 ≥20% 的条目（`:176-184`）；
baseline 为 0 的会保留而非丢弃（`:148-155`）。输出
`diff_<YYYY-MM-DD-HH-MM-SS>_<PID>.{txt,csv}`（`:475-485`）。裸调打完整 help。

**`XPU_Profiler_TraceClean`** —— 修复部分事件带垃圾 `ts` 的 trace
（大概是未初始化的 XPU 时间戳）。启发式基于**位数**而非数值：
`calculate_ts_digits` 数整数部分位数（`:7-38`），`determine_threshold` 建直方图并返回
累计占比达到 `threshold_percentage`（默认 90）的最小位数（`:41-88`，
完全没有 `ts` 时回落 15，`:68`），然后丢掉超宽的事件（`:91-196`）。
兼容裸数组和 `{"traceEvents": [...]}` 两种形状，用 `data.copy()` 保留其它顶层键（`:113`），
没有 `ts` 或 `ts` 非数值的事件一律保留（`:148-153`）。`--analyze` 只报分布不写文件（`:199-273`）。

**`XPU_Profiler_TraceMerge`** —— 分布式 trace 合并。rank 来源优先
`distributedInfo.rank`，回落文件名正则 `rank[-_]?(\d+)` / `TP[-_]?(\d+)` / `world[-_]?(\d+)`，
最后取文件名里最后一个数字（`:92-100`）。
`validate_same_execution`（`:138-209`）在这些情况下**直接 exit**：rank 重复、
`world_size` 或 `backend` 不一致、rank 数超过 `world_size`、
`default_pg` 的 rank 集合覆盖不了发现的 rank。
每 rank `process_events`（`:212-247`）按该 rank 自己的 `min_ts` 做时间窗过滤、
丢掉 `threading.py(` 前缀噪声、给 flow event id 加偏移防跨 rank 冲突、
给每个 `pid` 加 `[Rank NN]` 前缀。flow-id 偏移在单独的预扫描里算（`:265-275`），
正是因为 id 会跨 rank 重叠。合并时流式写 gzip 而不是在内存里拼大 dict（`:288-321`），
每个 rank 处理完就 `del trace` —— 刻意的内存措施。已存在的输出文件不会被覆盖（`:339-342`）。

## 其它调试工具（XLA 时代，多数已坏）

| 文件 | 行数 | 状态 |
|---|---|---|
| `debug/graph_saver.py` | 38 | `save_tensors_graph()` dump lazy-tensor 图（text/dot/hlo），受 `SAVE_GRAPH_FMT` 控制 |
| `debug/metrics.py` | 53 | `_XMLIRC._xpu_counter_*` / `_xpu_metric_*` / `_xpu_metrics_report` 的薄封装 |
| `debug/metrics_saver.py` | 71 | ⚠️ `:65` 用了 `sys` 但没 import，`XLA_METRICS_FILE=STDERR` 会 `NameError` |
| `debug/metrics_compare_utils.py` | 224 | ⚠️ `:197` 调不存在的 `_parse_metrics_report`（真名是 `:85` 的 `parse_metrics_report`），`compare_metrics` **是坏的** |
| `debug/model_comparator.py` | 243 | 独立 CLI：`python -m torch_xmlir.debug.model_comparator DIR1 DIR2`。⚠️ `:169` 把 `collections.namedtuple(...)` 当元组构造器用了 4 个位置参数，所以 `_parse_path` 会 raise，不匹配时的 `tensor_file_compare` 也跟着炸 |

## 版本自检

```bash
python -m torch_xmlir --doctor
```

`__main__.py:11-12` 调 `utils/version.py:60-68 get_version_string()`，
在 `_versions.txt` 内容之外补上 `XTE`（transformer_engine）、
`XFlashAttention`（flash_attn 的 importlib.metadata）、`Hydrax` 版本。

⚠️ `__main__.py` 在**导入期**就解析 argv（`:7`），传未知参数会直接硬失败。**[源码]**

`torch_xmlir.__version__` 是 `get_version_dict()` 的返回值，
**是一个 `OrderedDict` 而不是字符串**（`utils/version.py:46-57`，`__init__.py:64`）。
本机值见 [首页](README.md)。解析规则：跳过 `#` 行，按**第一个** `:` 切分（值里可以含冒号）。

## 构建自定义 XPU 扩展

`$PKG/utils/cpp_extension.py`（823 行），`__all__` = `["xpu_include_paths",
"xpu_library_paths", "CppExtension", "XPUExtension", "BuildExtension"]`（`:34-40`）。

`XPUExtension`（`:737`）硬链这些库（`:783-795`）：

```
c10, torch, torch_cpu, torch_python, XMLIRRuntime,
xdnn_pytorch, xpuapi, xlog_adapter, xpurt, cudart
```

- 编译器发现需要 `XPYTORCH_XTDK`，缺失时抛 `NotImplementedError` 并给下载链接（`:155-170`）
- `.cu` 源码额外需要 `XTRANS_PATH`（`:171-178`）
- 除调用方自带 `-On` 外，xtdk 一律强制 `-O2`（`:806-819`）
- dev 模式检测靠 `pip show xmlir` 找 `Editable project location`（`:343-359`），`functools.cache` 缓存

## 其它 utils

| 文件 | 行数 | 说明 |
|---|---|---|
| `utils/custom_op_loader.py` | 17 | `torch.ops.load_library` 的断言包装，见 [06](06-custom-ops-nn.md) |
| `utils/version.py` | 68 | 上面已述 |
| `utils/utils.py` | 410 | `getenv_as`（`:175-182`，`core/` 读环境变量都走它）、`SampleGenerator`、`TimedScope`、`get_free_tcp_ports`、`parallel_work` |
| `utils/gcsfs.py` | 424 | `gs://` 文件系统垫片，全部走 `_XMLIRC._xpu_tffile_*`（`:38-44`）。**[.so 内]** |
| `utils/serialization.py` | 132 | `save()`/`load()` 把 XPU tensor 溢出到同级 `<path>.tensors/` 目录存 CPU `.pt`；`:51-53` 用 `_XMLIRC._xpu_sync_multi` |
| `utils/cached_dataset.py` | 169 | `CachedDataset` 文件/GCS 样本缓存 |
| `utils/tf_record_reader.py` | 69 | `TfRecordReader` 走 `_XMLIRC._xpu_create_tfrecord_reader` 等 |
| `utils/keyd_queue.py` | 125 | `KeydQueue` 按 key 寻址的阻塞队列 |
| `utils/checkpoint_tagger.py` | 61 | 引用计数的 name→path 标签 + JSON 往返 |
| `utils/data/distributed.py` | 138 | `DistributedSampler`，官方实现但换用 `torch_xmlir.distributed` 取 world size/rank（`:10,73-79`）。**不是 DataLoader 适配** |

## 多进程与 CUDA IPC

`$PKG/multiprocessing.py`（203 行）两半 **[源码]**：

**进程启动器**（XLA 血统）：`spawn()`（`:30-64`）转发 `torch.multiprocessing.start_processes`，
wrapper `_mp_start_fn`（`:19-27`）打印 `device=xpu:{index}` 且异常时 `exit(17)`；
`MpSerialExecutor`（`:67-98`）是把「只让一个 rank 下载数据集」这类操作串行化的锁。

**CUDA IPC 对**（由 `__init__.py:62` 导入命名空间）：

```python
# $PKG/multiprocessing.py:117-127
(
    device, handle, storage_size_bytes, storage_offset_bytes,
    ref_counter_handle, ref_counter_offset, event_handle, event_sync_required,
) = _XMLIRC._share_cuda_(storage)
```

`_reduce_cuda_tensor`（`:101-147`）返回 15 元组；`_rebuild_cuda_tensor`（`:150-203`）
反向重建，零长度 storage 短路（`:168-169`），否则
`torch.cuda._lazy_init()` + `_XMLIRC._new_shared_cuda(...)`（`:171-181`）。

**适配策略：完整保留官方 CUDA IPC 协议和元组布局，只把两个碰驱动 handle 的原语
（`_share_cuda_`、`_new_shared_cuda`）改路由到 `_XMLIRC`。** 实现在 `.so` 里。

拒绝条件：非 leaf 的 `requires_grad` tensor（`:104-110`）、
`storage.device.type != "cuda"`（`:111-112`）。

真正把这两个函数接进 `ForkingPickler` 的是 symbrewrite
（`plugins/torch/__init__.py:98-109` 替换 `reduce_tensor`/`rebuild_cuda_tensor`/`init_reductions`），
`multiprocessing.py` 自己不注册。

## `test_utils/`

3669 行，包含 `automated_test/`（含 `dynamo_parser.py`，会生成
`torch.compile(fn, backend=XMLIRCompiler())` 形式的测试）和 `random_test/`。
本次未深入分析。

---

下一页：[12-pitfalls.md](12-pitfalls.md)
