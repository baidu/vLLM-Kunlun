# 02 · 导入 hook：整条链的起点

[← 首页](README.md) · [← 01 架构](01-architecture.md) · [→ 03 符号改写](03-symbrewrite.md)

这一层回答一个问题：为什么用户代码里 **只写 `import torch`**，昆仑芯的补丁就已经生效了。

## 触发器：`.pth` 文件

`site-packages/torch_xmlir.pth` 只有一行，Python 解释器启动时由 `site` 模块自动执行
（`.pth` 中以 `import` 开头的行会被 exec）：

```python
import os, sys;exec('import xpytorch_import_hook\nxpytorch_import_hook.hook_torch()')
```

这意味着 **只要这个 conda 环境的解释器一启动**，hook 就装好了 —— 哪怕脚本完全不 import torch。**[源码]**

## 安装：替换 `builtins.__import__`

`site-packages/xpytorch_import_hook.py`（151 行）：

```python
# xpytorch_import_hook.py:13-14
if not hasattr(builtins, "__origin__import__"):
    builtins.__origin__import__ = builtins.__import__

# xpytorch_import_hook.py:147-151
def hook_torch():
    disable_xpytorch = int(os.environ.get("DISABLE_XPYTORCH", "0"))
    if disable_xpytorch:
        return
    builtins.__import__ = _custom_import
```

注意实现方式：**替换 `builtins.__import__` 函数**，而不是 `sys.meta_path` 上装
`MetaPathFinder`/`Loader`。这是全局的、粗粒度的，也是后面一系列怪异行为的根源。**[源码]**

原函数被存到 `builtins.__origin__import__`（一组下划线），用于逃生（见下）。

`DISABLE_XPYTORCH=1` 是整个适配层的总开关，在两处被检查：
`xpytorch_import_hook.py:148` 和 `$PKG/__init__.py:49`。设置后 hook 不装、
`_XMLIRC` 不导入、`init()` 不跑。

## 触发条件与一次性 bootstrap

```python
# xpytorch_import_hook.py:30-34
trigger_hook_list = ["torch", "torch_xmlir", "torchvision", "transformers"]
is_trigger = any(
    module_name == hook or module_name.startswith(hook + ".")
    for hook in trigger_hook_list
)
```

`transformers` 出现在这个列表里有历史原因，行 29 的注释：
「升级到 transformers 4.42.3 后遇到 import hook 问题，因此需要加入此列表」。**[源码]**

bootstrap（`:37-106`）只跑一次，由模块级 `START` 标志守卫。它做的事按顺序：

1. 注入环境变量（下一节）
2. `import torch_plugin; torch_plugin.initialize_runtime()` —— 预载 xcudart 垫片（`:75-77`）
3. `import torch` → `import torch_xmlir`（`:78-80`）
4. `try: import pyarrow except ImportError: pass`（`:83-86`）—— 注释里挂了一个内部 ku 文档链接，
   典型的「某个库的加载顺序会炸，先 import 一下压住」workaround
5. 取两个注册表单例，`enable_rewrite_for_module("torch")`（`:93-95`）
6. 对其它已注册且已在 `sys.modules` 里的顶层模块，**先 `del sys.modules[m]` 再重新 import**（`:96-106`）

第 6 步是最激进的一步：

```python
# xpytorch_import_hook.py:96-106
for i in SYMBOL_REWRITE_REGISTER.reg.keys():
    module_list = [m for m in sys.modules if (i == m or m.startswith(i + "."))]
    if i != "torch" and module_list:
        SKIP_LIST.append(i)
        for m in module_list:
            del sys.modules[m]
        for m in module_list:
            importlib.import_module(m)
        SYMBOL_REWRITE_REGISTER.enable_rewrite_for_module(i)
```

目的是让补丁落在「干净的模块对象」上。副作用是：如果用户在 `import torch` 之前已经
`import transformers` 并持有了里面的类引用，那些引用会指向被丢弃的旧模块对象。**[静态推断]**

## 强制注入的环境变量

bootstrap 里有 5 个 `if X not in os.environ` 的强制设置，全部带中文注释说明动机。
这些是**理解昆仑芯行为的关键**，因为它们改变了官方 CUDA runtime 的语义：

| 变量 | 强制值 | 行 | 注释给出的原因 |
|---|---|---|---|
| `CUDART_DUMMY_REGISTER` | `1` | `:48-49` | 强制 `__cudaRegister***` 返回成功；不设置的话官方 torch 在 KL3 上会触发底层 runtime 报错 |
| `CUDART_MODULE_LOADING` | `LAZY` | `:51-52` | 惰性加载 CUDA module |
| `CUDA_DEVICE_MAX_CONNECTIONS` | `8` | `:58-59` | 单进程单卡场景下把并行流上限拉满；注释点名「llama 70b 多流优化至少需要 5 个流，默认 4 发挥不出优势」 |
| `CUDA_DEVICE_ORDER` | `OAM_ID` | `:64-68` | 统一各机型拓扑表征。**仅当** `CUDA_VISIBLE_DEVICES` 恰好等于字符串 `"0,1,2,3,4,5,6,7"` 时才设置 |
| `XPU_FORCE_SHARED_DEVICE_CONTEXT` | `1` | `:71-72` | 共享 device context |

`CUDA_DEVICE_ORDER` 那个条件是精确字符串比较 —— 写成 `"0,1,2,3,4,5,6,7 "`（带空格）
或者只用 4 张卡，就不会设置，拓扑可能和 8 卡跑法不一致。**[源码]**

## 稳态：每次 import 做什么

bootstrap 之后，`_custom_import` 的每次调用（`:111-139`）逻辑很短：

- 若模块名落在符号改写注册表的某个 bucket，且顶层模块不在 `SKIP_LIST` 里
  → import 它，然后 `enable_rewrite_for_module`（`:116-122`）
- 若模块名落在模块替换注册表 → `replace_module`（`:127-139`）
- 最后一律转交 `builtins.__origin__import__`（`:141-143`）

`SKIP_LIST` 提供「只改一次」语义。所有异常都被 `try/except` 包住，
只打 `WARNING: hook error!` 到 stderr（`:123-125`, `:137-139`）——
**补丁失败不会让程序崩，只会让你静默地跑在未打补丁的 torch 上**。这是调试时值得警惕的一点。

## 逃生舱：临时卸掉 hook

因为 hook 会递归进 torch 内部造成麻烦，有一个装饰器专门用来临时还原：

```python
# $PKG/symbrewrite/plugins/torch/mock_torch.py:109-128
def with_import_hook_disabled(func):
    def wrapper(*args, **kwargs):
        import builtins
        import_backup = getattr(builtins, "__import__", None)
        if hasattr(builtins, "__origin__import__"):
            builtins.__import__ = builtins.__origin__import__
        try:
            result = func(*args, **kwargs)
            ...
        finally:
            if import_backup is not None:
                builtins.__import__ = import_backup
```

用在 `torch._export.aot_compile`（`plugins/torch/__init__.py:121-125`）和
「关闭 mock 时的真实 `torch.compile`」（`:136-139`）。**[源码]**

注意名字区分：hook 自己用 `__origin__import__`，symbrewrite 存原始符号用
`__origin_<name>`，两套约定互不相干。

---

下一页：[03-symbrewrite.md](03-symbrewrite.md) —— 符号改写引擎与插件系统
