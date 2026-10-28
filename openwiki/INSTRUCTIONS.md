# OpenWiki Instructions

本文件由维护者手写，用于约束 wiki 的生成范围与优先级。**任何自动化流程都不应重写本文件。**

## 仓库定位

`baidu/vLLM-Kunlun` 是 vLLM 的**树外硬件插件**（out-of-tree hardware plugin），
按 vLLM Hardware-pluggable RFC（vllm-project/vllm#11162）把昆仑芯 Kunlun3 P800 XPU
接入 vLLM。它不 fork vLLM，而是通过 entry point + import hook + 模块重定向
+ 安装期文件覆盖四种手段，在运行时把 vLLM 的 CUDA 路径改写到 XPU 上。

## 本 wiki 的范围（按优先级）

1. **插件挂载机制** —— 插件如何被 vLLM 发现、bootstrap 的七个阶段、
   四种覆盖手段各自的可见性边界。这是读懂本仓库的唯一入口，优先级最高。
2. **KunlunPlatform 契约** —— `device_name="cuda"` 的伪装策略、`check_and_update_config`
   的五处强制改写、attention backend 选择逻辑。
3. **计算子系统** —— attention（4+1 个 backend）、Kunlun Graph（编译/图捕获）、
   量化（W8A8/W4A16/AWQ/GPTQ）、MoE 与专家并行、FLA/Mamba、投机解码、采样。
4. **模型支持面** —— 12 个注册模型各自为什么需要 OOT 实现，以及文档矩阵与代码的差异。
5. **构建、安装与版本对齐** —— 厂商 wheel、xpytorch、`setup_env.sh`、版本号的三处不一致。
6. **测试与 CI 的真实覆盖度** —— 哪些 workflow 是门禁，哪些只是装饰。

## 明确排除

+ 不针对某个特定压测/profiling 工作流做裁剪，本 wiki 是**通用全仓库文档**。
+ 不复制上游 vLLM 的通用概念解释（PagedAttention、continuous batching 等），
  只写 Kunlun 插件相对上游的**差异**。
+ 不记录厂商 wheel（`kunlun_ops` / `xspeedgate_ops` / `cocopod`）的内部实现，
  它们是闭源二进制，只记录调用契约。

## 约定

+ 每条实质性论断都必须绑定版本化证据，形式为 `repo://<path>#L<start>-L<end>`。
  结构化 Claim 存放在 `openwiki/.claims/<page>.json` 旁挂文件里，**不内联进正文**。
+ 证据版本固定为分支 `v0.25.1-dev`、commit `c53e090`。仓库的 `main` 分支内容陈旧
  （仍宣称 v0.15.1 / "Initial release"），**不要以 `main` 为准**。
+ "缺失"类论断（例如"不支持 PD 分离"）也需要证据：给出穷尽搜索的范围与否定结果。
+ 代码与文档冲突时，以**代码**为准，并把冲突本身作为一条 Claim 记录到
  [known-gaps.md](known-gaps.md)。
