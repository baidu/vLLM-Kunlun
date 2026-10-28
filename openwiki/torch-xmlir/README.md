# 昆仑芯 torch hook 层 Wiki（torch_xmlir / xmlir）

自动生成的代码 wiki，覆盖 **昆仑芯 XPU 的 PyTorch 适配层**：它如何 hook 住官方 PyTorch，
让写给 CUDA 的模型代码不改一行就跑在昆仑芯上。

风格参考 [langchain-ai/openwiki](https://github.com/langchain-ai/openwiki)：互链页面 +
每条事实带源码出处 + 明确区分「已验证 / 静态推断 / .so 内不可见」。

## 分析范围与快照边界

本文基于一个面向 Kunlun3 P800、使用 PyTorch 2.9 系列的 XMLIR 运行时快照进行分析。它是**独立分析快照**，不改变也不代表本仓库的默认安装组合。

截至本文编写时，`ci/scripts/env/install_env.sh` 安装的是 `xpytorch-cp310-torch251`，并向 PyTorch 2.5.1 的 Dynamo 路径写入补丁。因此，本文中标记为 PyTorch 2.9 行为的结论**不能直接用于**该安装流程；将两者组合前应单独完成兼容性验证。

文中 `$PKG` 指已安装的 `torch_xmlir` Python 包目录。包目录之外还涉及 `torch_xmlir.pth` 与 `xpytorch_import_hook.py`，它们位于 Python 的 site-packages 目录中。

本文记录的是特定运行时构建的行为快照，不应将未标注为源码事实的结论直接外推到其他 XMLIR、驱动或 PyTorch 版本。

## 一句话架构

昆仑芯**不注册新设备类型**。它把自己伪装成 CUDA：在 C++ 层把 `DispatchKey::CUDA` 上的
官方 aten kernel 全部注销再换成自己的，在 Python 层用 `builtins.__import__` hook
批量改写 `torch.*` 符号，在系统层通过 CUDA 驱动与运行时 ABI 兼容接口完成转接。
于是 `torch.cuda.is_available()` 返回 True，`.to("cuda")` 把张量搬到昆仑卡上。

四层劫持详见 [01-architecture.md](01-architecture.md)。

## 页面导航

| 页面 | 内容 |
|---|---|
| [01-architecture.md](01-architecture.md) | 四层劫持总览、`import torch` 的完整启动时序、版本门控 |
| [02-import-hook.md](02-import-hook.md) | `.pth` → `builtins.__import__` 替换、强制注入的 CUDA 环境变量 |
| [03-symbrewrite.md](03-symbrewrite.md) | 符号改写引擎、`__origin_` 约定、xflags 插件系统与 `XFLAGS` CLI |
| [04-dispatch-hijack.md](04-dispatch-hijack.md) | `op_deregister_C` 注销 CUDA kernel、CUDA ABI 垫片、op_select 策略 |
| [05-device-runtime.md](05-device-runtime.md) | `xpu/` 设备/流/事件/显存/Graph、`torch.cuda.*` 的真实映射关系 |
| [06-custom-ops-nn.md](06-custom-ops-nn.md) | `custom-op component` 与 `nn/` 融合算子（Linear、RoPE、RMSNorm、FA…） |
| [07-distributed.md](07-distributed.md) | kccl/bccl 通信后端、DTensor 自定义 handler、DDP |
| [08-compile-dynamo.md](08-compile-dynamo.md) | `torch.compile` 被空装饰器替换、`dynamo/` 死代码分析 |
| [09-amp-optimizer.md](09-amp-optimizer.md) | autocast 双套状态、GradScaler、融合优化器 |
| [10-env-vars.md](10-env-vars.md) | 环境变量总表（Python 可见 + `.so` 内字符串） |
| [11-debug-tools.md](11-debug-tools.md) | `validate_aten` 双跑对齐、profiler 工具链、cpp_extension |
| [12-pitfalls.md](12-pitfalls.md) | 已确认的死代码、latent bug、行为陷阱 |

## 建议阅读顺序

想搞懂「为什么我的 CUDA 代码能跑」→ 01 → 04 → 02 → 03。
想调性能/精度 → 10 → 06 → 09 → 11。
遇到诡异行为 → 12。

## 事实可信度标记

全文统一使用：

- **[已验证]**：在分析环境中实际执行确认过，例如 Python 探测、`objdump` 或 `strings`。
- **[源码]**：直接读到的 Python 源码，标注相对包路径和行号。
- **[静态推断]**：由源码或符号表推理得出，未在运行时验证。
- **[.so 内]**：逻辑位于编译产物中，Python 侧不可见，只能给出符号名。
