# 01 · 总体架构

[← 返回首页](README.md)

## 四层劫持

昆仑芯适配层的核心设计决策是：**不新增 PyTorch 设备类型，直接冒充 CUDA**。
全包搜索 `privateuse1` / `rename_privateuse1_backend` / `_register_device_module`
只命中 profiler 插件里两处字符串处理（`symbrewrite/plugins/profiler/statistic_dump.py:121,137`），
没有任何自定义 `c10::DeviceType` 或 `DeviceGuardImplInterface` 注册。**[已验证]**

代价是必须在四个层面同时伪装：

```
┌─ L4  Python 符号层 ── builtins.__import__ hook + symbrewrite 改写 torch.* 属性
│                       见 02-import-hook.md / 03-symbrewrite.md
├─ L3  Dispatcher 层 ── 注销 DispatchKey::CUDA 上的官方 kernel，换成 XPU kernel
│                       见 04-dispatch-hijack.md
├─ L2  ATen/C++ 层 ──── XMLIR Python extension 内 codegen 的 RegisterCUDA.cpp + custom_ops
│                       见 06-custom-ops-nn.md
└─ L1  驱动 ABI 层 ──── CUDA ABI 兼容库 → Kunlun XPU 驱动与运行时垫片
```

L1 的直接后果：`torch.cuda.Stream` / `Event` / caching allocator / CUDA Graph
**全部是未经修改的官方 PyTorch 代码**，只是底下的 `cudaStreamCreate` 打到了昆仑运行时。
这也是为什么 `torch.cuda.get_device_name(0)` 返回 `'GPU'`、
`get_device_capability(0)` 返回 `(8, 6)` —— 这些值来自垫片，不是 torch_xmlir 编的。**[已验证]**

## `import torch` 的完整启动时序

用户代码里 **不需要** `import torch_xmlir`。整条链由 `.pth` 触发：

| # | 动作 | 位置 |
|---|---|---|
| 1 | 解释器启动执行 `.pth` → `xpytorch_import_hook.hook_torch()` | `site-packages/torch_xmlir.pth:1` |
| 2 | `builtins.__import__ = _custom_import` | `xpytorch_import_hook.py:151` |
| 3 | 首次 `import torch`（或 torch_xmlir/torchvision/transformers）触发 bootstrap | `xpytorch_import_hook.py:30,37` |
| 4 | 强制注入 5 个 CUDA 兼容环境变量 | `xpytorch_import_hook.py:48-72` |
| 5 | `torch_plugin.initialize_runtime()` 预载 xcudart 垫片 | `xpytorch_import_hook.py:75-77` |
| 6 | `import torch`（官方包，此时 CUDA 垫片已就位） | `xpytorch_import_hook.py:78` |
| 7 | `import torch_xmlir` → 进入下面的 8~15 | `xpytorch_import_hook.py:80` |
| 8 | `op_deregister_C.deregister_all_op("CUDA", "aten", {...})` 注销官方 CUDA kernel | `$PKG/__init__.py:26-40` |
| 9 | `import torch_xmlir.xpu`、`from . import _XMLIRC`（44MB pybind 扩展） | `$PKG/__init__.py:51-55` |
| 10 | `init()`：关 nvfuser/GPU fusion、`_XMLIRC._init_xpu_backend()`、`init_op_select()` | `$PKG/__init__.py:150-168` |
| 11 | `XPU_RUN_MODE` 若设置 → 写硬件寄存器切精度模式 | `$PKG/__init__.py:170-176` |
| 12 | `_XMLIRC._xpu_init_caching_allocator()` | `$PKG/__init__.py:200` |
| 13 | `load_custom_ops_library(custom-op component)` | `$PKG/__init__.py:215` |
| 14 | `from torch_xmlir.distributed import tensor` 注册 DTensor handler | `$PKG/__init__.py:218` |
| 15 | `initialize_plugin_with_xflags()` 按 xflags 导入 symbrewrite 插件 | `$PKG/__init__.py:275` |
| 16 | 回到 hook：`enable_rewrite_for_module("torch")` 落地全部符号改写 | `xpytorch_import_hook.py:95` |

第 13 步必须在第 14 步之前 —— DTensor 注册代码在模块导入期就引用
`torch.ops.custom_ops.*`，`$PKG/__init__.py:217` 的注释明确说明了这个顺序依赖。**[源码]**

## 版本门控：本构建上哪些代码根本不跑

`init()` 里有三处版本判断，在 torch 2.9 下的实际结果 **[已验证]**：

```python
# $PKG/__init__.py:181-185
digit_version = int("".join(list(filter(str.isdigit, torch.__version__))[0:3]))
if digit_version >= 200:
    if digit_version < 250:
        _apply_patches_201()
```

`"2.9.0+cu129"` 取数字前 3 位 → `"290"` → `290`。所以：

| 门控 | 条件 | 实际结果 | 后果 |
|---|---|---|---|
| `_apply_patches_1121()` | `"1.12.1" in torch.__version__` | **不执行** | — |
| `_apply_patches_201()` | `200 <= dv < 250` → `290` | **不执行** | `_patched_functions.py` 里 1122 行的 `broadcast_object_list` / `barrier` / `all_gather_object` 等 distributed 补丁全部休眠 |
| `_apply_patches_201_dynamo()` + `xmlir` backend 注册 | `not WITHOUT_MLIR` | **不执行** | 见下 |
| `enable_pytorch_storage_replacement()` | `torch.__version__.find("cu") == -1` | **不执行**（版本串含 `cu129`） | storage 替换未启用 |

`WITHOUT_MLIR` 是一个已确认的 latent bug：

```python
# $PKG/__init__.py:47
WITHOUT_MLIR = bool(os.environ.get("WITHOUT_MLIR", "0"))
```

`bool("0")` 是 `True`，所以 `WITHOUT_MLIR` **默认永远为真**（运行时确认 `torch_xmlir.WITHOUT_MLIR == True`）。
整个 `dynamo/` 后端注册路径（`__init__.py:187-197`）从不执行。
详见 [08-compile-dynamo.md](08-compile-dynamo.md) 和 [12-pitfalls.md](12-pitfalls.md)。**[已验证]**

## 本构建上真正生效的 Python 层改写

运行时探测结果 **[已验证]**：

| 符号 | 实际指向 |
|---|---|
| `torch.compile` | `symbrewrite.plugins.torch.mock_torch.empty_decorator`（空装饰器！） |
| `torch.nn.Linear` | `torch_xmlir.nn.linear.Linear` |
| `torch.nn.functional.linear` | `torch_xmlir.nn.linear.linear` |
| `torch.backends.cudnn.enabled` | `False` |
| `torch.distributed.ProcessGroupNCCL` | `torch.distributed.ProcessGroupXCCL` |
| `torch.distributed.is_nccl_available()` | `True`（硬编码 lambda） |
| `torch.jit.script` | `mock_torch.empty_decorator` |

## 模块规模速览

243 个 `.py` 文件、约 43k 行 Python，加 23 个 `.so`（最大 `XPU API component` 287MB、
`XPU BLAS component` 259MB、`XDNN component` 249MB）。按行数：

| 模块 | 行数 | 状态 | 页面 |
|---|---|---|---|
| `symbrewrite/` | 13259 | **核心，活跃** | [03](03-symbrewrite.md) |
| `nn/` | 5614 | 部分活跃（Linear 默认生效） | [06](06-custom-ops-nn.md) |
| `distributed/` | 3878 | `tensor/` 活跃，其余死代码 | [07](07-distributed.md) |
| `test_utils/` | 3669 | 测试生成工具 | — |
| `core/` | 2758 | XLA 时代遗留，大量 `NotImplementedError` | [05](05-device-runtime.md) |
| `utils/` | 2454 | 活跃（custom_op_loader / version / cpp_extension） | [11](11-debug-tools.md) |
| `debug/` | 2292 | 全部 opt-in | [11](11-debug-tools.md) |
| `xflags/` | 2014 | 活跃（其中 1806 行是 vendored tabulate） | [03](03-symbrewrite.md) |
| `optimizer/` | 1861 | opt-in | [09](09-amp-optimizer.md) |
| `dynamo/` | 1284 | **死代码**（缺 `torch_xmlir.ir`） | [08](08-compile-dynamo.md) |
| `xpu/` | 1052 | 部分活跃（`memory.py` 整体不可用） | [05](05-device-runtime.md) |
| `amp/` | 701 | 无人 import，opt-in 死路径 | [09](09-amp-optimizer.md) |
| `pd/` | 43 | — | — |

---

下一页：[02-import-hook.md](02-import-hook.md) · 环境变量总表：[10-env-vars.md](10-env-vars.md)
