# 04 · Dispatcher 劫持：偷走 DispatchKey::CUDA

[← 首页](README.md) · [← 03 符号改写](03-symbrewrite.md) · [→ 05 设备与运行时](05-device-runtime.md)

Python 层的 monkeypatch 只能改几十个符号。真正让**上千个 aten 算子**跑到昆仑芯上的是这一层。

## 步骤一：注销官方 CUDA kernel

`$PKG/__init__.py` 的**第 26 行**，比 `import _XMLIRC` 还早：

```python
# $PKG/__init__.py:26-40
from . import op_deregister_C

ignore_deregister_op = {
    "aten::record_stream",
}

res = op_deregister_C.deregister_all_op("CUDA", "aten", ignore_deregister_op)
assert res, "Deregister cuda aten ops error"

nested_op_list = {"aten::to_padded_tensor"}
op_deregister_C.deregister_list_ops("NestedTensorCUDA", nested_op_list)

sparse_cuda_op_list = {"aten::_coalesce"}
op_deregister_C.deregister_list_ops("SparseCUDA", sparse_cuda_op_list)
```

把 `DispatchKey::CUDA` 上 `aten` 命名空间的**所有** kernel 从 dispatcher 里摘掉，
只留 `aten::record_stream` 一个例外。另外单点注销 `NestedTensorCUDA::to_padded_tensor`
和 `SparseCUDA::_coalesce`。**[源码]**

Dispatcher helper extension 导出的 C++ 符号：

```
torch_xmlir::deregister_all_op(std::string, std::string, std::unordered_set<std::string>)
torch_xmlir::deregister_list_ops(...)
```

**[.so 内]** dispatcher 内部到底怎么摘除注册项的，Python 侧看不到。

注意 `assert res`（`:33`）—— 这一步失败会直接让 `import torch` 抛 AssertionError。

## 步骤二：重新注册 XPU kernel

这是 codegen 生成的 C++，编译到 XMLIR Python 扩展中。运行时 `torch._C._dispatch_dump('aten::add.Tensor')` 显示 CPU 注册来自 PyTorch 的 ATen 实现，而 CUDA 注册来自 XMLIR 的 `aten_codegen/generated/RegisterCUDA.cpp`。**[已验证]**

`RegisterCUDA.cpp` 的路径串只出现在 XMLIR Python 扩展中，不出现在其他计算组件中。由此可见，ATen 层注册集中在该扩展，计算实现位于下游编译组件。**[已验证]**

调试意义：**排查算子问题时 `_dispatch_dump` 是最直接的判据**。
看到 `RegisterCUDA.cpp` 表示走昆仑 kernel，看到 PyTorch 构建目录下的实现则表示走官方实现。

## 步骤三：CUDA ABI 垫片

运行时通过一组 CUDA ABI 兼容组件将 CUDA 驱动、运行时和管理接口转接到 Kunlun XPU 实现。兼容组件提供 CUDA 驱动、CUDA 运行时和 NVML 接口，并在底层调用 XPU 驱动与运行时。**[已验证]**

兼容组件中可见异步 launch 与 wait 相关符号，以及驱动版本错误信息。**[已验证]**

这一层的存在解释了 [05](05-device-runtime.md) 的核心结论：
**`torch.cuda` 的流、事件、显存分配器、CUDA Graph 全是官方 PyTorch 原码**。

## 步骤四：backend init

`init()` 里与 dispatcher 相关的部分：

```python
# $PKG/__init__.py:161-168
if parse_version(torch.__version__) < Version("2.2.0"):
    torch._C._jit_set_nvfuser_enabled(False)
torch._C._jit_texpr_set_fallback_allowed(True)
torch._C._jit_override_can_fuse_on_gpu(False)

_XMLIRC._init_xpu_backend()
atexit.register(_prepare_to_exit)
init_op_select()
```

关掉 GPU 上的 JIT fusion（融合出来的 kernel 是 CUDA 代码，昆仑跑不了），
初始化后端，注册退出钩子。**[.so 内]** `_init_xpu_backend` / `_prepare_to_exit` /
`_xpu_init_caching_allocator` 在 `XMLIR Python extension` 里确认存在，实现不可见。

## op_select：逐算子分派策略

`$PKG/_op_select.py` 有两个职责。

### 职责一：预载 xtrans 库 —— **已被注释掉**

```python
# $PKG/_op_select.py:44-50
def load_xtrans():
    try:
        for path in (api_path, llvm15_path, xtrans_path):
            _load_shared_library_if_exists(path)
    except Exception as e:
        warnings.warn(f"Failed to load optional xtrans component: {e}")
```

`$PKG/__init__.py:19` import 了它，但 `:21` 的调用是 `# load_xtrans()`。
而且三个目标里 `optional xtrans component` **本构建没有发货**，
`_load_shared_library_if_exists`（`:39-41`）会用 `os.path.exists` 静默跳过。
所以 `XPU API component` / `LLVM runtime component` 的 `RTLD_GLOBAL` 预载从不发生。**[源码]**

### 职责二：注册 op-select 配置（这个是活的）

```python
# $PKG/_op_select.py:53-58
def init_op_select():
    torch_xmlir._XMLIRC._forceRegisterOpsectTrans()
    config_path = os.getenv("XMLIR_XTRANS_OP_CONFIG", DEFAULT_CONFIG_PATH)
    data = load_yaml(config_path)
    processed_data = process_data(data)
    register_ops(processed_data)
```

`register_ops`（`:33-36`）逐条推到 C++：

```python
torch_xmlir._XMLIRC._registerOPSelectConfig(opname, select_type, dtype)
```

配置文件 `$PKG/op_config.yml`（可用 `XMLIR_XTRANS_OP_CONFIG` 换掉），有效内容：

```yaml
blacklist:
  - opname: empty_strided
  - opname: nonzero
  - opname: native_batch_norm
  - opname: convolution_backward
no_guard_op:
  - opname: fill_
```

文件前 5 行是注释掉的模板，记录了另外两个类别 `force_dispatch_trans` 和
`force_fallback_trans`。四个类别名在 `XMLIR Python extension` 里都能逐字匹配到。**[已验证]**

按名字与相邻环境变量（`XMLIR_XTRANS_OP_PRIOR`、`XMLIR_XTRANS_REGISTER_OPSELECT`）推断：
`blacklist` 把算子排除出 xtrans 路径（所以这四个算子会回落），
`no_guard_op` 对 `fill_` 跳过 device guard，两个 `force_*_trans` 分别把算子钉死在
dispatch-translation / fallback-translation。dtype 字段支持逐 dtype 粒度
（模板示例是 `add` 配 `dtype: [float16]`）。**[静态推断]** —— 精确语义在 C++ 里，未验证。

## eager fallback

包里有 `eager-fallback component`（496KB），且 `XMLIR Python extension` 内有这些环境变量字符串
**[.so 内]**：

| 变量 | 名字暗示的作用 |
|---|---|
| `XMLIR_ENABLE_FALLBACK_TO_CPU_BOOL` | 允许算子回落到 CPU |
| `XMLIR_D_FORCE_FALLBACK_STR` | 强制指定算子回落（`validate_aten` 报错时会提示用它，见 [11](11-debug-tools.md)） |
| `XMLIR_DEBUG_FALLBACK_ALL` / `XACC_DEBUG_FALLBACK_ALL` | 全量回落调试 |
| `XMLIR_DUMP_FALLBACK_OP_LIST_BOOL` / `XMLIR_FALLBACK_OP_LIST_FILE_PATH` | 导出回落算子清单 |
| `XMLIR_XDNN_PYTORCH_CHECK_ENABLE_FALLBACK_BOOL` | XDNN 侧回落检查 |

**精度对不上先怀疑某个算子，用 `XMLIR_D_FORCE_FALLBACK_STR='<op>'` 把它按到 CPU 上验证**
—— 这是 `validate_aten` 内建的诊断建议（`debug/validate_aten/validate/contexts.py:566-569`）。

## 特殊 dispatch 点

`XMLIR Python extension` / `custom-op component` 里有一个值得单独点出的符号：
`wrapper_CUDA___fused_sdp_choice`。这说明 **SDPA（`scaled_dot_product_attention`）
的后端选择是在 C++ dispatcher 层截获的**，不在 Python 层，
可用 `XMLIR_FUSED_SDP_CHOICE` 调整。**[.so 内]**

---

下一页：[05-device-runtime.md](05-device-runtime.md)
