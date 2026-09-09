# 08 · torch.compile 与 dynamo

[← 首页](README.md) · [← 07 分布式](07-distributed.md) · [→ 09 AMP 与优化器](09-amp-optimizer.md)

本页两条结论，都不太好听：

> **1. 本构建上 `torch.compile` 是一个空装饰器。它不编译任何东西。**
> **2. `dynamo/`（1284 行）是死代码，缺少的依赖模块没有随包发货。**

## `torch.compile` 被替换成空装饰器

```python
# $PKG/symbrewrite/plugins/torch/__init__.py:128
# XMLIR_ENABLE_MOCK_TORCH_COMPILE 默认 "true"
```

`:132` 把 `torch.compile` 换成 `mock_torch.empty_decorator`。运行时确认 **[已验证]**：

```
torch.compile -> torch_xmlir.symbrewrite.plugins.torch.mock_torch  empty_decorator
```

也就是说，用户代码里的 `@torch.compile` 或 `model = torch.compile(model)`
**静默地什么都不做，模型以 eager 模式运行**。不报错、不警告。

这在工程上是可以理解的选择（编译路径不成熟，宁可静默 eager 也别炸），
但**调优时必须知道**：你以为在测编译性能，其实测的是 eager。

要关掉这个 mock：`XMLIR_ENABLE_MOCK_TORCH_COMPILE=false`。此时走的是
`with_import_hook_disabled` 包过的**真实官方 `torch.compile`**（`:136-139`），
即 Inductor 路径 —— 而 Inductor 会生成 Triton CUDA kernel，在昆仑上能否工作本次未验证。

## `dynamo/` 为什么是死代码

三重死亡 **[已验证]**：

**(1) 依赖模块不存在。** `dynamo/` import `torch_xmlir.ir` 和 `torch_xmlir.dialects`
（`dynamo/xmlir_builder.py:30-31`、`dynamo/ir.py:6-7`、`dynamo/affine.py:50`），
而这两个模块**都不存在** —— 没有 `ir.py`、没有 `dialects/`，
`xmlir-1.0.0.1.dist-info/RECORD` 里也没有对应条目。**MLIR/xgraph 层没有随包发货。**

**(2) 用了 torch 2.9 已删除的 API：**

| API | 位置 |
|---|---|
| `torch._dynamo.config_utils.install_config_module` | `dynamo/builder_config.py:8` |
| `torch._dynamo.utils.fake_mode_from_tensors` | `dynamo/xmlir_builder.py:14` |
| `torch.fx.experimental.symbolic_shapes.{magic_methods, method_to_operator}` | `dynamo/xmlir_builder.py:9-13` |

**(3) 门控 bug 意外救了它。** `$PKG/__init__.py:47`：

```python
WITHOUT_MLIR = bool(os.environ.get("WITHOUT_MLIR", "0"))
```

`bool("0") is True` —— 所以 `WITHOUT_MLIR` **默认永远为真**
（运行时确认 `torch_xmlir.WITHOUT_MLIR == True`）。于是 `if not WITHOUT_MLIR:` 下面的
所有东西从不执行，包括 `_apply_patches_201_dynamo()` 和 backend 注册：

```python
# $PKG/__init__.py:187-197
if not WITHOUT_MLIR:
    from ._dynamo_patched_functions import _apply_patches_201_dynamo
    _apply_patches_201_dynamo()

    from torch_xmlir.dynamo import compile_fx
    from torch._dynamo import register_backend

    @register_backend
    def xmlir(*args, **kwargs):
        return compile_fx(*args, **kwargs)
```

**讽刺之处：正是这个 bug 让 `import torch` 不至于因为 (1)(2) 而崩掉。**
如果哪天有人「修好」了 `WITHOUT_MLIR` 的解析，这个包会立刻 import 失败。
见 [12-pitfalls.md](12-pitfalls.md)。

`if not WITHOUT_MLIR` 还守着 `:201-202` 的
`_XMLIRC._xpu_set_xmlir_opt_path(...)` —— 也从不执行。

## `dynamo/` 的设计（供参考，当前不可运行）

即使不能跑，架构值得记录，因为它揭示了昆仑原本的编译路线。

### 后端名与三条注册路径

后端名是 **`xmlir`**（靠 `register_backend` 读函数名）。三条入口：

1. `torch.compile(m, backend="xmlir")` —— 走 `register_backend`（`__init__.py:195-197`）
2. patch 过的 `torch.compile` 自己特判字符串并直接构造 `XMLIRCompiler`
   （`_dynamo_patched_functions.py:121-124`）
3. `torch.compile(fn, backend=XMLIRCompiler())` —— 包内测试生成器用的就是这个
   （`test_utils/automated_test/dynamo_parser.py:78`）

### 不是适配 Inductor，是替换 Inductor

`dynamo/` 里没有任何 Inductor codegen / scheduling / Triton 定制。
`_dynamo_patched_functions.py:117-120` 把 `backend="inductor"` 原样留给官方
`_TorchCompileInductorWrapper`。XPU 路径是完全独立的后端，终点是 C++ 编译调用而不是 Triton：

```python
# $PKG/dynamo/xmlir_builder.py:371-378
def compile_ir_module(self, func_name, ir_module_str):
    # Here we use graph id as ir module id, beacause dynamo has cached graph by guarded code.
    exector = torch_xmlir._XMLIRC.compile(ir_module_str)

    def exector_wrap(*args):
        return exector(list(args))

    return exector_wrap
```

MLIR module 以**字符串**形式交给扩展，之后的解析、优化、XPU codegen、返回 executor
全在 `XMLIR Python extension`（44MB）和 `XMLIR runtime component` / `JIT compiler component` 里。**[.so 内]**

### 流水线

`compile_fx`（`xmlir_builder.py:417-454`）是 AOTAutograd 的薄封装：
从 `get_aot_compilation_context()` 读模型名和 graph index（`:422`），
围绕 `FXGraph2IRModule().compile_to_fn` 建 `fw_compiler`/`bw_compiler`，
返回 `aot_autograd(...)` 且 `keep_inference_input_mutations=True`（`:453`）。
分解表和 partitioner 都用 AOTAutograd 默认值 —— `:449` 标了 TODO。

`GraphIRLowering`（`:80-356`）是一个 `torch.fx.Interpreter`，边遍历边发射 MLIR。
算子名映射是机械的：

```python
# $PKG/dynamo/xmlir_builder.py:219-231
if isinstance(node.target, torch._ops.OpOverload):
    schema = node.target._schema
    namespace, _, unqualified_name = schema.name.partition("::")
    mlir_op_name = f"xgraph.{namespace}.{unqualified_name}"
    if schema.overload_name != "":
        mlir_op_name += f".{schema.overload_name}"
    assert ir.Context.current.is_registered_operation(
        mlir_op_name
    ), f"Unregistered operation: {mlir_op_name}"
```

即 `aten::add.Tensor` → `xgraph.aten.add.Tensor`。

三类节点特殊处理：`operator.getitem` 靠 env 别名解决不发 op（`:207-209`）；
SymInt magic method 变成 `xgraph.builtin_op` 并把算子名做首个 `ConstantStrOp`（`:259-281`）；
其余落到 `lowerings` 表（`:282-299`，`dynamo/lowering.py` 只有 3 条：
`scalar_tensor`、`sym_size`、`sym_stride`）。
`call_module` / `call_method` 直接 assert —— 图必须先被 AOTAutograd 完全展平。

### 动态 shape 是半成品

```python
# $PKG/dynamo/ir.py:199-207
def _symbolic_shape_to_xmlir_ams(self, x) -> str:
    if self.is_symbolic:
        # TODO: replace "?" by affine_expr
        affine_expr = SymExprToAffineExpr(x).convert()
        return "?"
```

**算出 affine 表达式然后丢掉。** `affine.py` 整整 316 行 sympy→AffineExpr 机制
（`SymExprToAffineExpr`，`:276-316`）只为喂这个被丢弃的值，等于全部未使用。

### `dynamo/passes/`：唯一一个 pass

`ReplaceMisSemanticOp`（`passes/replace_missemantice_op.py:35`）。
对 16 个白名单二元算子（`converse_binary_patttern_list`，`:16-32`：
`add`/`add_`/`sub`/`sub_`/`mul`/`mul_`/`div` × 5 overload/`remainder`/`pow`/`pow_`，
全是 `.Tensor` 变体），逐位置参数对照 schema 声明类型。
schema 要 Tensor 而图给了 Python 标量时，插入节点：

```python
# $PKG/dynamo/passes/replace_missemantice_op.py:117-134
if is_tensor(expected_type) and is_scalar(arg):
    dtype = get_type(arg)
    device = torch.device("cpu")
    scalar_tensor: torch.fx.Node = self.module.graph.call_function(
        the_function=aten.scalar_tensor,
        args=(arg,),
        kwargs={"dtype": dtype, "device": device, "layout": None, "pin_memory": False},
    )
    counters["xacc"]["scalar_tensor_nodes"] += 1
    arguments[i] = scalar_tensor
    n.prepend(scalar_tensor)
```

dtype 推断 `int→int64` / `float→float32` / `bool→bool` / else 默认（`:99-108`），
device 硬编码 `cpu`（`:119`）。改写计数进 `torch._dynamo.utils.counters["xacc"]`。
docstring（`:36-46`）说明动机：`aten.add.Tensor(tensor, 1)` 转 xgraph 时会类型不匹配。

这个 pass 和 `dynamo/lowering.py` 里的 `aten.scalar_tensor` 条目是**同一个 workaround 的两半**。

⚠️ `__all__ = ["ReplaceMisSemanticeOp"]`（`:6`）多了个 `e`，与类名不符，`import *` 会失败。

两个本该有的 pass 明确缺席：`FakeTensorProp` 在 `xmlir_builder.py:397` 被注释掉
（shape inference 挪进 `GraphIRLowering`，因为它在动态 shape 上会炸），且没有 decomposition pass。

## `_dynamo_patched_functions.py` 的三个补丁

全部由 `_apply_patches_201_dynamo()`（`:130-137`）应用，**本构建不执行**。
文件头 `:66` 的注释写着「torch2.0.1 官方bug」，`:77` 的 TODO 说明整个文件都是 2.0.1 workaround。

| # | 目标 | 方式 | 内容 |
|---|---|---|---|
| 1 | `torch.fx.experimental.proxy_tensor.set_meta` | `_patch`（`:131-133`） | 对真实（非 fake、非 sparse）tensor 不再返回 `None`，而是起一个 `FakeTensorMode(allow_fallback_kernels=True)` 造 `torch.empty_strided(...)`（`:41-45`），让 FX 节点拿到正确的 `meta["val"]` —— 这正是 `GraphIRLowering.to_xmlir_ir_type` 需要的。作者在 `:35-40` 自己标注了 hacky 且不感知 storage |
| 2 | `aten.unsqueeze_.default` 的 Meta kernel | **直写 `py_kernels` dict**（`:135`），不可逆 | 用 `maybe_wrap_dim` + `inferUnsqueezeGeometry` + `as_strided` 三行重实现（`:70-74`）。注意 `:7` import 了 `register_meta` 却没用 |
| 3 | `torch.compile` | `_patch`（`:137`） | torch 2.0.1 版 `torch.compile` 的拷贝加一个 `elif backend == "xmlir"` 分支（`:117-127`）。签名钉死在 2.0.1 形状（`:78-87`，`dynamic: bool = False`、无 `guard_filter_fn`/`backend_options`），在 torch 2.9 上会静默丢掉新 kwargs |

补丁 3 即使执行也会被后来的 symbrewrite 覆盖 —— `plugins/torch/__init__.py:132`
再一次替换 `torch.compile`。

`_patch` 本身有签名校验（`_patched_functions.py:43-53`）：

```python
def _patch(fn, newfn):
    xfingerprint = inspect.signature(fn)
    fingerprint = inspect.signature(newfn)
    if xfingerprint != fingerprint:
        raise RuntimeError("Unable to patch {}, signature mismatch: {} vs {}".format(...))
    newfn._orig = fn
    return newfn
```

**这是个好设计** —— 签名漂移会立刻抛错而不是静默行为不一致。
（对比 symbrewrite 的 `setattr`，完全不校验。）

## 实际结论

想在昆仑 P800 + torch 2.9 上用图编译：

- `torch.compile` 默认无效，必须显式 `XMLIR_ENABLE_MOCK_TORCH_COMPILE=false` 才走官方 Inductor
- 昆仑自研的 `xmlir` 后端在本构建**不可用**（缺 `torch_xmlir.ir` / `dialects`）
- 真正生效的加速手段是 [06](06-custom-ops-nn.md) 里的 eager 融合算子
  （Linear/hydrax、RoPE、RMSNorm、FA）和 [04](04-dispatch-hijack.md) 的 op_select

---

下一页：[09-amp-optimizer.md](09-amp-optimizer.md)
