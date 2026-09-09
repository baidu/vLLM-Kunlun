# 12 · 已知坑、死代码与 latent bug

[← 首页](README.md) · [← 11 调试工具](11-debug-tools.md)

按「会不会咬到你」排序。**遇到诡异行为时先扫这一页。**

---

## A · 行为陷阱（最可能咬人）

### A1 · `torch.compile` 静默变成空装饰器

`XMLIR_ENABLE_MOCK_TORCH_COMPILE` 默认 `true` → `torch.compile` 是
`mock_torch.empty_decorator`（运行时确认）。**不报错、不警告，模型静默以 eager 运行。**

**影响**：任何「编译前后性能对比」的结论都是错的。
**规避**：`XMLIR_ENABLE_MOCK_TORCH_COMPILE=false`（此时走官方 Inductor，昆仑上是否可用未验证）。
详见 [08](08-compile-dynamo.md)。**[已验证]**

### A2 · 所有补丁失败都是静默的

三处静默失败路径 **[源码]**：

| 位置 | 行为 |
|---|---|
| `xpytorch_import_hook.py:123-125,137-139` | import hook 异常 → 只打 `WARNING: hook error!` 到 stderr |
| `symbrewrite/symbol_rewrite.py:96-105` | 中间层 getattr 失败 → 打 warning 并 skip 这一条替换 |
| `$PKG/__init__.py:274-281` | 整个插件初始化失败 → `traceback.print_exc()` + warn |

**影响**：你可能在完全没打补丁的官方 torch 上跑，而只有 stderr 里一行 warning。
**规避**：启动日志必须看。正常启动应该看到 `SYMBOL_REWRITE torch success`。
用 `LOG_SYMBOL_REPLACE=1` 确认关键符号真的被替换了。

### A3 · `xpu/memory.py` 整体不可用

`_xpu_empty_cache` / `_xpu_memory_stats` / `_xpu_memory_snapshot` / `_mem_get_info`
四个 pybind 符号在本构建的**任何** `.so` 里都不存在（23 个库逐一扫描确认）。
整个 `torch_xmlir.xpu.memory` 模块（除 `use_l3`）都会抛 `AttributeError`。**[已验证]**

**影响**：`XMLIR_DISABLE_CUDA_ALLOCATOR=1` 会把 14 个 `torch.cuda.memory_*` 指到这些坏函数上。
**规避**：**不要设这个变量**。默认路径用官方 caching allocator，`PYTORCH_CUDA_ALLOC_CONF` 正常生效。
详见 [05](05-device-runtime.md)。

### A4 · `XPU_RUN_MODE` 在 import 期改共享硬件状态

`$PKG/__init__.py:170-176` → `_xpu3_run_mode.py` 通过运行时提供的寄存器写入工具
写 12 个硬件寄存器，默认作用于设备 `0..7`。**[源码]**

三个后果：

1. 同节点并发多任务时，一个进程设 `XPU_RUN_MODE` 会影响**所有选中设备**。
   用 `LOCAL_RANK` 可以缩到单卡。
2. **拼错的模式名被静默忽略** —— `write()` 遇未知 mode 只 warn 就 return（`:72-76`），
   `check()` 也 warn 后 return。`XPU_RUN_MODE=TRAIN_BF16`（下划线）什么都不做，不报错。
3. `INFER-BF16` 和 `INFER-FP16` 写同一个值 `-1`（`:15-20`），寄存器层面无法区分。

### A5 · `config` 在 import 期冻结

`config = Config.from_env()` 在 `config.py:163` 模块导入期执行。
`import torch_xmlir` 之后再 `os.environ[...] = ...` 对这 13 个 flag **无效**。**[源码]**

而且 TE 和 hydrax 各自直接读环境变量、解析方式不一致
（`transformer_engine/.../gemm.py:135` 用 `in ("true","1","True")`，大小写敏感且无默认值）。
详见 [06](06-custom-ops-nn.md)。

### A6 · 精确字符串比较的开关

这些开关**不接受常见的等价写法** **[源码]**：

| 变量 | 判断 | 陷阱 |
|---|---|---|
| `XMLIR_USE_HYDRA_LINEAR` | `== "1"` | 写 `"true"` 无效，会走 legacy 路径 |
| `XMLIR_PERSISTENT_CACHE_ENABLED` | `== "true"` | 写 `"1"` / `"True"` 无效 |
| `XMLIR_ENABLE_LINEAR_FC_FUSION` | `int(...)` | 写 `"false"` 直接抛 `ValueError` |
| `XMLIR_FORCE_USE_XPU_GRAPH` | `int(...)` | 同上 |
| `XPYTORCH_RUN_ENHANCE` | `int(...)` | 同上 |
| `CUDA_DEVICE_ORDER` 注入条件 | `CUDA_VISIBLE_DEVICES == "0,1,2,3,4,5,6,7"` | 多空格、4 卡、乱序都不触发 |

### A7 · `XTORCHRUN_CLEAN` 会杀别人的进程

`enhance_launch_torch2_5.py:142-156`（和 `…2_0.py:201`）的实现是
`lsof | grep /dev/xpu | ... | xargs kill -9`，**会杀掉本机上除自己以外任何持有 XPU 设备的进程**。
共享节点上不要开。需先 `XPYTORCH_RUN_ENHANCE=1` 才走到这里。**[源码]**

### A8 · DTensor 切分约束会静默回落

RoPE 的 head_dim、softmax/RMSNorm 的最后一维、`mha_varlen_fwd` 的**全部三个维度**
都不可切，不满足时静默 `Replicate`。张量并行性能不达预期时先查
[07](07-distributed.md) 的约束表。**[源码]**

### A9 · `capability` 两套值不一致

`torch.cuda.get_device_capability()` → `(8, 6)`（垫片编的），
`torch_xmlir.xpu.get_device_capability()` → `(0, 0)`（`_XpuDeviceProperties` 里 major/minor
标注 `N/A for XPU`）。依赖 capability 做分支的第三方库要注意。**[已验证]**

### A10 · `dump_dir` 会被 rmtree

`validate_aten` 的 `dump_dir` 默认 `validate_dump_<时间戳>`，
**若目录已存在会被 `shutil.rmtree`**（`common/contexts.py:151-153`）。
不要把它指向有用数据的目录。**[源码]**

### A11 · `plugins.xflags` 写在 site-packages 内部

`XFLAGS --enable` 修改的是 `<pkg>/symbrewrite/plugins/plugins.xflags`，
不是 `$HOME` 下的文件。需要对已安装包的写权限，**且会被重装/升级覆盖**。
容器镜像里改了要注意持久化。**[源码]**

### A12 · `XFLAGS --reset` 与 `--enable` 同时给会互相打架

`xflags_cli.py:198` 先把 default 写盘，`:201` 的 `update_flags()` 又用**加载 reset 之前**
的内存状态覆盖回去 —— reset 等于被丢弃。**[源码]**

---

## B · 已确认的 latent bug

### B1 · `WITHOUT_MLIR` 布尔解析（最讽刺的一个）

```python
# $PKG/__init__.py:47
WITHOUT_MLIR = bool(os.environ.get("WITHOUT_MLIR", "0"))
```

`bool("0")` 是 `True`。**除了设成空字符串，任何值（包括 `"0"`）都让它为真。**
运行时确认 `torch_xmlir.WITHOUT_MLIR == True`。**[已验证]**

**讽刺之处**：正因为它永真，`dynamo/` 那条 import 必炸的路径（缺
`torch_xmlir.ir` / `dialects`，且用了 torch 2.9 已删的 API）才从不执行。
**「修好」这个 bug 会让整个包 import 失败。** 见 [08](08-compile-dynamo.md)。

### B2 · `xpu/device.py:84` 正则的运算符优先级

```python
re.match(r"xla|xpu:(\d+)$", device_str)
```

`|` 结合比预期松，实际含义是 `xla` **或** `xpu:(\d+)$`。
走 `"xla"` 分支时 `m.group(1)` 是 `None`，`:87` 执行 `int(None)`。
运行时确认 `xpu_device_hw('xla')` 抛
`TypeError: int() argument must be ... not 'NoneType'`。
应写作 `r"(?:xla|xpu):(\d+)$"`。**[已验证]**

### B3 · `_DEVICES` 格式与解析正则不匹配

`_XMLIRC._xpu_get_devices()` 返回 `['XLA: 0', 'XPU: 0', 'XLA: 1', 'XPU: 1', ...]`
（**冒号后有空格**），但 `parse_xpu_device`（`xpu/device.py:262`）用
`r"(CPU|TPU|GPU|XPU):(\d+)$"`，`get_xpu_supported_devices`（`:285`）用 `kind + r":\d+$"`，
两个都匹配不上。于是 `get_xpu_supported_devices()` 返回 `None`，
导致 `xpu_replication_devices`（`:229`）在 `:244` 的 `len(None)` 处失败。**[已验证]**

### B4 · `_DEVICES` 索引错位

列表把 XLA 和 XPU 条目交错排列，所以索引 `N` 不是设备 `N`。
`xpu_device_hw('xpu:0')` 返回 `'XLA'`（因为 `_DEVICES.value[0]` 是 `'XLA: 0'`）。**[已验证]**

### B5 · `distributed/__init__.py` 的 `__all__`

```python
__all__ = ["DistributedDataParallel, tensor"]   # 一个含逗号的字符串
```

`from torch_xmlir.distributed import *` 一个名字都导不出来。**[源码]**

### B6 · `fused_adam.py` 的 step 泄漏

`optimizer/fused_adam.py:149,166` 的 `state` 从 `for p in group["params"]` 循环里泄漏出来，
`state["step"] += 1` 只递增**最后一个参数**的 step，却把这个值喂给整个 param group。
另外因为 fp16 在 `:135` 就 raise 了，`:148` 的 `if len(g_16) > 0` 分支不可达。**[源码]**

### B7 · 字符串用 `is` 比较

`nn/rotary_pos_emb_index.py:18,73` 的 `layout is "BLHD"`。
依赖 CPython 字符串 interning，不保证成立。**[源码]**

### B8 · `__all__` 拼写不匹配

`dynamo/passes/replace_missemantice_op.py:6` 的
`__all__ = ["ReplaceMisSemanticeOp"]` 多了个 `e`，类名是 `ReplaceMisSemanticOp`。
`from ... import *` 会失败。**[源码]**

### B9 · `metrics_saver.py` 缺 `import sys`

`debug/metrics_saver.py:65` 用了 `sys`，所以 `XLA_METRICS_FILE=STDERR` 会 `NameError`。**[源码]**

### B10 · `compare_metrics` 调不存在的函数

`debug/metrics_compare_utils.py:197` 调 `_parse_metrics_report`，
真名是 `:85` 的 `parse_metrics_report`。**该函数是坏的。[源码]**

### B11 · `model_comparator.py` 误用 namedtuple

`debug/model_comparator.py:169` 把 `collections.namedtuple(...)` 当元组构造器用了
4 个位置参数，所以 `_parse_path` 会 raise，不匹配时的 `tensor_file_compare` 跟着炸。**[源码]**

### B12 · `validate_aten` 的几处

- `get_module_index()`（`validate/contexts.py:78-80`）改 `MODULE_COUNTER` 但没 `global` → `UnboundLocalError`
- `InsertDelimiters.__init__` 的 `elif isinstance(model, torch.nn.Module)` 分支（`:96-99`）
  在未定义的 `module` 上迭代 → **`InsertDelimiters(model=...)` 坏的**
- `:2` `from crypt import methods`、`:4` `import nntplib` 未使用，且在 CPython 3.13 已移除

**[源码]**

### B13 · `_xpu3_run_mode.py` 的 `_get_device`

声明时既没 `self` 也没 `@staticmethod`，只因为在类上访问才能工作（`:14`）。**[源码]**

### B14 · `xpu_synchonize_default_stream` 拼写

`xpu/device.py:291` —— `synchonize` 少个 `r`，是公开 API 名字的一部分。**[源码]**

### B15 · `distributed/distributed.py` 构造即崩

`:119`/`:122` 调 `_XMLIRC._xpu_sync_multi` 和 `._xpu_reset_tokens`，
两个符号在 `XMLIR Python extension` 里都不存在（用 `_xpu_set_default_device` 等做对照命中确认）。
`_sync_params_and_buffers` 在 `__init__` 里无条件调用（`:57`），
所以**构造 `torch_xmlir.distributed.DistributedDataParallel` 就会 `AttributeError`**。**[已验证]**

---

## C · 死代码清单

| 模块/函数 | 行数 | 为什么死 | 证据 |
|---|---|---|---|
| `dynamo/` 全部 | 1284 | 缺 `torch_xmlir.ir` / `dialects`；用了 torch 2.9 已删 API；且被 `WITHOUT_MLIR` 门死 | [已验证] |
| `dynamo/affine.py` | 316 | 唯一消费者 `ir.py:199-207` 算完 affine 表达式就丢掉，直接 `return "?"` | [源码] |
| `_dynamo_patched_functions.py` | 137 | 由 `WITHOUT_MLIR` 门死；文件头注释自称 torch 2.0.1 workaround | [已验证] |
| `_patched_functions.py` 的 distributed 补丁 | ~1000 | `_apply_patches_201()` 只在 `200 <= dv < 250` 调用，torch 2.9 → `dv=290` | [已验证] |
| `enable_pytorch_storage_replacement()` | — | 条件 `torch.__version__.find("cu") == -1`，版本串含 `cu129` | [已验证] |
| `core/xpu_model.py:mark_step` | — | `:678` 无条件 `raise NotImplementedError`，连带 `optimizer_step`（`:919,947` 调它）整条路径不可用 | [源码] |
| `core/xpu_model.py:xpu_device` | — | `:184` 同样 raise | [源码] |
| `core/xpu_env_vars.py` 的 19 个常量 | — | 22 个里只有 `WORLD_SIZE`/`ORDINAL`/`LOCAL_ORDINAL` 被消费，其余是 torch_xla 遗留 | [源码] |
| `xpu/memory.py` | 441 | 见 A3 | [已验证] |
| `amp/` 全部 | 701 | 全 site-packages 无人 import；且 fp16 only、拒绝 `device_type="cuda"` | [已验证] |
| `nn/parallel/` | 2175 | `distributed.py:17` 调不存在的 `torch_xmlir.distributed.is_available()`；无人 import | [静态推断] |
| `distributed/distributed.py` | 130 | 见 B15 | [已验证] |
| `distributed/fsdp/` | 465 | 老 API 拷贝，与 torch 2.9 FSDP 不兼容，无人引用。**本适配层对 FSDP 无任何专门适配** | [已验证] |
| `distributed/utils.py` | 45 | `__all__ = []`，无人引用 | [源码] |
| `_op_select.py:load_xtrans()` | — | `__init__.py:21` 的调用被注释掉；且 `optional xtrans component` 未发货 | [源码] |
| `_utils.py:_init_graph_disk_cache()` | 33 | `<pkg>/builtin_modules/` 不存在，`:547` 直接 return，无论环境变量如何都是 no-op | [已验证] |
| `plugins/torch_vision/` 的替换 | — | 6 处 `SYMBOL_REPLACE` **全被注释掉**（`__init__.py:13-36`），虽然默认 `True` 但什么都不改。`mock_torchvision.py` 里的实现还活着 | [源码] |
| `config.py` 的 11 个属性 | — | 只有 `enable_param_state` / `enable_activ_state` 有消费者；`use_cast_fc_fusion` 硬编码 `0` 且无人读 | [已验证] |
| `nn/rms_norm.py:12-16` 的 `stats_dtype` | — | 算完不用 | [源码] |
| `optimizer/` 里 4 个 kernel | — | `multi_tensor_l2norm`（在 `optimizer/` 内）、`multi_tensor_cuda_lamb`、`optimizer_LAMB_fused` 在 `.so` 里有但 `optimizer/` 无调用者 | [源码] |
| `nn/mmcv_ops/` | 96 | 无人 import（只有 wheel RECORD 里的条目）；`mean`/`sum` backward 直接 raise | [已验证] |
| loss 类 custom_ops | — | `hard_softmax_with_cross_entropy`、`te_cross_entropy`、`sigmoid_focal_loss_*` 在 `.so` 里但无 Python 包装 | [已验证] |
| 大量 mmcv custom_ops | — | `deform_conv`、`ms_deform_attn`、`roi_align`、`nms_rotated`、`ball_query`、`voxel_pooling_train` 等有 C++ 无 Python | [已验证] |

---

## D · 精度不对时的排查顺序

综合各页信息，给一条实用路径：

1. **确认补丁真的生效** —— 看启动日志有 `SYMBOL_REWRITE torch success`；
   `LOG_SYMBOL_REPLACE=1` 确认关键符号
2. **确认走的是哪条 Linear 路径** —— `XMLIR_USE_HYDRA_LINEAR` 默认走 **hydrax 包**，
   不在 torch_xmlir 里。可用 `FORCE_NN_LINEAR=1` 退到官方实现做对照
3. **确认 dispatch 目标** —— `torch._C._dispatch_dump('aten::xxx')`，
   看是 `RegisterCUDA.cpp` 还是 `the PyTorch build directory`
4. **逐算子双跑** —— `with Validate(atol=..., rtol=...)`，看 `__exit__` 打出的不一致表
5. **定点回落** —— 对可疑算子设 `XMLIR_D_FORCE_FALLBACK_STR='<op>'` 按到 CPU 验证
6. **检查 fast-FC 开关** —— `XMLIR_ENABLE_FAST_FC` 的覆盖关系（[06](06-custom-ops-nn.md)），
   注意 TE / hydrax 的解析差异
7. **检查 RNG 差异** —— 如果需要与 GPU 位对齐，`TORCH_NN_INIT_IS_GPU_ALIGNED=1` +
   `GPU_NAME` / `RANDPERM_KEY_BITS`（[03](03-symbrewrite.md)）
8. **检查 autocast dtype** —— `torch_xmlir.amp.autocast` 是 fp16 only 的死路径，
   应该用官方 `torch.amp.autocast("cuda")`（[09](09-amp-optimizer.md)）
9. **注意 `lamb.py:118-128`** —— trust_ratio 的零值守卫被刻意注释掉，可能出 inf/NaN

## E · 关于代码的一个整体观察

这个包里能读出**三个时代的地层** **[源码]**：

| 时代 | 特征 | 代表 |
|---|---|---|
| XLA / lazy-tensor 时代 | 设备类型硬编码 `"xla"`，`mark_step`，XRT 环境变量，`core/xpu_builder.py` 的 HLO DSL | `core/`、`nn/parallel/`、`utils/gcsfs.py`、`_utils.py:469` 返回 `"xla"` |
| MLIR / xgraph 时代 | `xgraph.aten.*` op 名、`_XMLIRC.compile(ir_str)`、affine 表达式 | `dynamo/`（依赖未发货） |
| CUDA 劫持时代（当前） | 注销 `DispatchKey::CUDA`、CUDA ABI 垫片、`assert device.type == "cuda"` | `__init__.py:26-40`、`symbrewrite/`、`optimizer/fused_sgd.py:73-74` |

同一个包里 `nn/parallel/comm.py:91` 断言 `"xla"` 而 `optimizer/fused_sgd.py:73` 断言 `"cuda"`
——这是判断某段代码属于哪个时代（因而是否还活着）最快的启发式。

---

[← 返回首页](README.md)
