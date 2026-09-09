# 10 · 环境变量总表

[← 首页](README.md) · [← 09 AMP 与优化器](09-amp-optimizer.md) · [→ 11 调试工具](11-debug-tools.md)

分三部分：**A. 最该知道的**、**B. Python 侧完整表**、**C. 只在 `.so` 里的**。

C 部分只有名字，**默认值和语义未验证** —— 是 `strings` 扫出来的字符串表证据。

---

## A · 最该知道的 12 个

| 变量 | 默认 | 一句话 |
|---|---|---|
| `DISABLE_XPYTORCH` | `0` | **总开关**。设 1 则 hook 不装、`_XMLIRC` 不导入、`init()` 不跑，退回纯官方 torch |
| `XMLIR_ENABLE_MOCK_TORCH_COMPILE` | `true` | **默认让 `torch.compile` 变空装饰器**。测编译性能前必须设成 `false`，见 [08](08-compile-dynamo.md) |
| `XMLIR_ENABLE_LINEAR_FC_FUSION` | `1` | 替换 `nn.Linear`/`F.linear`。`int()` 解析，非数字值会抛 `ValueError` |
| `XMLIR_USE_HYDRA_LINEAR` | `1` | Linear 走 hydrax 快 FC。**精确比较 `== "1"`**，写 `true` 无效 |
| `XMLIR_ENABLE_FAST_FC` | `0` | bf16 fast-FC 总开关。**开了就屏蔽三个分方向变量**，见 [06](06-custom-ops-nn.md) |
| `XPU_RUN_MODE` | 未设 | `TRAIN-BF16`/`TRAIN-FP16`/`INFER-BF16`/`INFER-FP16`。**会在 import 期写硬件寄存器，影响整机共享状态**；拼错静默忽略 |
| `XMLIR_ENABLE_NEW_PG` | `1` | 通信后端选 `kccl`（新）vs `bccl`（旧） |
| `LOG_SYMBOL_REPLACE` | `0` | 设 1 打印每次符号替换的调用点。**排查「走的是官方还是昆仑实现」的首选** |
| `XMLIR_D_FORCE_FALLBACK_STR` | — | 强制指定算子回落到 CPU。**精度对不上时的核心诊断工具** |
| `XMLIR_DISABLE_CUDA_ALLOCATOR` | 未设 | **不要设成 1** —— 会把 14 个 `torch.cuda.memory_*` 指到本构建已损坏的实现上，见 [05](05-device-runtime.md) |
| `WITHOUT_MLIR` | `0` | **解析有 bug，永远为真**。见 [12](12-pitfalls.md) |
| `XTORCHRUN_CLEAN` | `0` | 非 0 时启动前 `kill -9` 所有持有 `/dev/xpu` 的其它进程。**共享节点上不要开** |

---

## B · Python 侧完整表

### B1 · 总控与启动（`$PKG/__init__.py`、`xpytorch_import_hook.py`）

| 变量 | 位置 | 默认 | 含义 |
|---|---|---|---|
| `DISABLE_XPYTORCH` | `__init__.py:49`、`xpytorch_import_hook.py:148` | `0` | 总开关 |
| `WITHOUT_MLIR` | `__init__.py:47` | `"0"` | 本意禁用 MLIR/dynamo。`bool("0")` → 永真，导致整条 dynamo 注册路径与 `_xpu_set_xmlir_opt_path` 从不执行 |
| `XPU_RUN_MODE` | `__init__.py:48` → `_xpu3_run_mode.py` | `None` | 硬件精度模式，写 12 个寄存器 |
| `TORCH_NN_INIT_IS_GPU_ALIGNED` | `__init__.py:288`、`plugins/torch/__init__.py:179` | `"0"` | 启用 GPU 位对齐 RNG（14 处改写）并加载 RNG 兼容组件。默认关闭，因为该组件依赖的 OpenMP 运行时可能与 sklearn 的 GNU OpenMP 运行时冲突并导致 segfault（`:283-286`） |
| `XMLIR_PERSISTENT_CACHE_ENABLED` | `_utils.py:536` | `"false"` | 必须**恰好等于 `"true"`**（`"1"`/`"True"` 无效）。把 `<pkg>/builtin_modules/*` 拷进 `~/.xmlir`。本构建 `builtin_modules/` 不存在，函数在 `:547` 直接 return，是 no-op |
| `XMLIR_XTRANS_OP_CONFIG` | `_op_select.py:55` | `<pkg>/op_config.yml` | op-select YAML 路径 |
| `LOCAL_RANK` | `_xpu3_run_mode.py:58` | `None` | 限定寄存器只写这一张卡 |
| `CUDA_VISIBLE_DEVICES` | `_xpu3_run_mode.py:59`、`xpytorch_import_hook.py:66` | `None` | 卡数回落判断 / `CUDA_DEVICE_ORDER` 条件 |

**强制注入**（若未设置则由 hook 写入，见 [02](02-import-hook.md)）：

| 变量 | 强制值 | 位置 |
|---|---|---|
| `CUDART_DUMMY_REGISTER` | `1` | `xpytorch_import_hook.py:48-49` |
| `CUDART_MODULE_LOADING` | `LAZY` | `:51-52` |
| `CUDA_DEVICE_MAX_CONNECTIONS` | `8` | `:58-59` |
| `CUDA_DEVICE_ORDER` | `OAM_ID` | `:64-68`（仅当 `CUDA_VISIBLE_DEVICES == "0,1,2,3,4,5,6,7"`） |
| `XPU_FORCE_SHARED_DEVICE_CONTEXT` | `1` | `:71-72` |

### B2 · Linear / fast-FC / 量化

| 变量 | 位置 | 默认 | 含义 |
|---|---|---|---|
| `XMLIR_ENABLE_LINEAR_FC_FUSION` | `plugins/torch/__init__.py:141` | `1` | 装 `nn.Linear`/`F.linear` 替换 |
| `XMLIR_USE_HYDRA_LINEAR` | `nn/linear.py:14,112` | `"1"` | 1 → hydrax；else → 包内 `custom_ops.linear` |
| `FORCE_NN_LINEAR` | `nn/linear.py:24` | 未设 | 任意真值 → 一律用官方 `torch._C._nn.linear`。最高优先级 kill switch |
| `USE_L3` | `nn/linear.py:23` | 未设 | legacy 路径输出 buffer 分配到 L3 scratch（`:49-51`） |
| `XMLIR_DYNAMO_WORKAROUND` | `nn/linear.py:25-28` | `"0"` | 走 `torch.ops._dynamo_workaround.linear` 便于图捕获 |
| `XDNN_FC_GEMM_DTYPE` | `nn/bmm.py:9`、`nn/baddbmm.py:9` | 未设 | `"float32"` → 跳过 `findmax`；否则跑动态 scale。C++ 侧 `XDNN component` 也读 |
| `DISABLE_CAST_CACHE` | `config.py:89` | `"0"` | 硬否决两个 cast cache flag |
| `XMLIR_ENABLE_FAST_FC` | `config.py:94` | `"0"` | 总开关，见 [06](06-custom-ops-nn.md) 的覆盖关系 |
| `XMLIR_LINEAR_CACHE_PARAM` | `config.py:101` | `"1"` | 缓存 cast 后权重。**仅在总开关开时才读**，否则强制 False |
| `XMLIR_LINEAR_CACHE_ACTIV` | `config.py:109` | `"0"` | 缓存 cast 后激活，同上 |
| `XMLIR_ENABLE_FAST_FC_FWD_OUT` | `config.py:118` | `"0"` | 前向输出 bf16。**总开关开时被忽略** |
| `XMLIR_ENABLE_FAST_FC_BWD_DW` | `config.py:121` | `"0"` | 反向权重梯度，同上 |
| `XMLIR_ENABLE_FAST_FC_BWD_DX` | `config.py:124` | `"0"` | 反向输入梯度，同上 |
| `XMLIR_BATCH_PARALLEL` | `config.py:137` | `"0"` | batch 并行 |
| `XMLIR_PARALLEL_SAVE_MEMORY` | `config.py:142` | `"true"` | 并行模式省显存。`config.py` 里唯一默认开的 |

⚠️ `config = Config.from_env()` 在 `config.py:163` **模块导入期**执行，
`import torch_xmlir` 之后改这些变量无效。且 TE / hydrax 各自直接读、解析方式不一致，
详见 [06](06-custom-ops-nn.md) 末节。

### B3 · 分布式

| 变量 | 位置 | 默认 | 含义 |
|---|---|---|---|
| `XMLIR_ENABLE_NEW_PG` | `mock_torch.py:21` | `"1"` | `kccl`（新 `torch_xmlir::kccl` PG）vs `bccl`（旧 `c10d::ProcessGroupXCCL`） |
| `XMLIR_DIST_DISABLE_ASYNC_ISEND_IRECV` | `mock_batch_isend_irecv.py:74` | `"0"` | 把 `batch_isend_irecv` 换成串行非 coalesce 循环（正确性逃生阀） |
| `TORCH_C10D_USE_RANDOM_SLEEP` | `plugins/torch/__init__.py:166` | `"0"` | rendezvous 加 rank 错峰 sleep，缓解 TCPStore 创建风暴 |
| `TORCHELASTIC_USE_AGENT_STORE` | `mock_torch.py:183` | 未设 | 由上者选中；走 agent store + `sleep(RANK//30+1)`（`:197-199`） |
| `TORCHELASTIC_RESTART_COUNT` | `mock_torch.py:196` | — | 用作 `PrefixStore` 路径。**无默认值，该分支上缺失会 KeyError** |
| `RANK` | `mock_torch.py:197` | `"0"` | 错峰 sleep 计算 |
| `XPYTORCH_RUN_ENHANCE` | `plugins/torch/__init__.py:148` | `0` | 替换 `torch.distributed.run.main` 为 NUMA 绑核 + SIGKILL 版 |
| `XTORCHRUN_CLEAN` | `enhance_launch_torch2_5.py:142`、`…2_0.py:201` | `"0"` | 启动前 `kill -9` 占用 `/dev/xpu` 的其它进程 |
| `TORCH_NCCL_ASYNC_ERROR_HANDLING` | `enhance_launch_torch2_5.py:117` | `"1"` | 透传给 worker |
| `NCCL_ASYNC_ERROR_HANDLING` | `enhance_launch_torch2_0.py:181` | `"1"` | torch < 2.3 的旧名 |
| `OMP_NUM_THREADS` | `enhance_launch_torch2_5.py:121-122` | — | 已设置才透传 |
| `ALLREDUCE_FUSION` | `core/xpu_model.py:801` | `True` | all-reduce 前分桶 |
| `ALLREDUCE_BUCKET_SIZE` | `core/xpu_model.py:802-804` | 32MB | 桶大小 |
| `ALLREDUCE_CHECK` | `core/xpu_model.py:809,838` | `False` | 拿 gloo CPU all-reduce 对账（调试） |
| `XRT_SHARD_WORLD_SIZE` / `_ORDINAL` / `_LOCAL_ORDINAL` | `core/xpu_model.py:113,129,145` | `1`/`0`/`-1` | 复制组参数 |

`nn/parallel/distributed.py:50-92` 读约 38 个 `NCCL_*` 变量，**仅用于打日志**，不影响行为。

### B4 · RNG 位对齐（`TORCH_NN_INIT_IS_GPU_ALIGNED=1` 时才相关）

| 变量 | 位置 | 默认 | 含义 |
|---|---|---|---|
| `SET_PHILOX_VERSION` | `mock_torch_init.py:46` | `"xpu"` | `xpu`/`python`/`cpp`；非法值静默回落 `xpu`（`:47-48`） |
| `RANDPERM_KEY_BITS` | `mock_torch_init.py:54` | 见下 | `"32"` 或 `"64"`，用于选择不同的 key 生成实现。优先级：本变量 > `GPU_NAME` 配置 > `"64"` |
| `GPU_NAME` | `gpu_config_manager.py:231` | — | 选择内置 GPU 配置，大小写不敏感 |
| `GPU_SM_COUNT` | `gpu_config_manager.py:190` | — | 目标 GPU SM 数，须与下者成对给出 |
| `GPU_THREADS_PER_SM` | `gpu_config_manager.py:200` | — | 每 SM 最大线程数。成对必需（`:209`），优先级高于 `GPU_NAME`（`:255-263`） |
| `PHILOX_DEBUG` | `mock_torch_init.py:37` | `"0"` | Philox 调试打印 |

### B5 · 日志、调试、可复现

| 变量 | 位置 | 默认 | 含义 |
|---|---|---|---|
| `LOG_SYMBOL_REPLACE` | `symbol_rewrite.py:13`、`module_replace.py:14` | `0` | `=="1"` 时每次替换调用都追踪打印 |
| `LOG_LEVEL` | `symbrewrite/logger.py:11` | `"INFO"` | logger 级别，变量名可由调用方覆盖 |
| `FA_LOG_LEVEL` | `plugins/flash_attn/mock_flash_attn_interface.py:9` | `"INFO"` | flash-attn 专用 |
| `XMLIR_FA_ACCUM_TYPE` | 同上 `:52,194` | `"float"` | `float`/`float16`，softmax-LSE 累加 dtype，非法值 assert |
| `XPYTORCH_REPRODUCE_SEED` | `plugins/reproduce/__init__.py:13` | `42` | 确定性复现种子。**该插件 import 即生效**（`:38`） |
| `MODELCMP_SAVEDIR` | `debug/model_comparator.py:54` | 未设 | 未设时 `save()` 是 no-op 直通 |
| `SAVE_GRAPH_FMT` | `debug/graph_saver.py:22` | `"text"` | `text`/`dot`/`hlo`，其它值 raise |
| `XLA_METRICS_FILE` | `debug/metrics_saver.py:35` | 未设 | metrics 报告输出。`STDERR` 分支因缺 `import sys` 会 NameError（`:65`） |
| `XLA_EMIT_STEPLOG` | `core/xpu_model.py:680` | `False` | 每 `mark_step` 打日志 —— 不可达（`mark_step` 在 `:678` raise） |
| `XLA_SYNC_WAIT` | `core/xpu_model.py:696,716` | `False` | 阻塞同步（`:716` 在 `get_xpu_data_ptr` 里是活的） |
| `XLA_OP_PRINT_COMPUTATIONS` | `core/xpu_op_registry.py:52` | `False` | 首次编译时把 HLO 打到 stderr |
| `RATE_TRACKER_SMOOTHING` | `core/xpu_model.py:269` | `0.4` | `RateTracker` EMA 系数 |
| `DEBUG` | `utils/utils.py:330` | `"0"` | 非 0 时 `get_print_fn()` 返回 `eprint` |

### B6 · 显存 / Graph

| 变量 | 位置 | 默认 | 含义 |
|---|---|---|---|
| `XMLIR_DISABLE_CUDA_ALLOCATOR` | `mock_allocator.py:6` | 未设 | `=="1"` 重定向 14 个 `torch.cuda.memory_*`。**本构建下会导致 AttributeError** |
| `XMLIR_FORCE_USE_XPU_GRAPH` | `plugins/torch/__init__.py:157` | `0` | 装 `xpu/graphs.py` 覆盖 `torch.cuda.CUDAGraph`/`graph` |
| `PYTORCH_CUDA_ALLOC_CONF` | 官方 torch | — | **默认路径下按官方语义正常生效**（torch_xmlir 不引用它） |

### B7 · 构建扩展（`utils/cpp_extension.py`）

| 变量 | 位置 | 默认 | 含义 |
|---|---|---|---|
| `XPYTORCH_XTDK` | `:151,155,156` | 必需 | XTDK 工具链路径，`<xtdk>/bin/clang++` 作为编译器。缺失抛 `NotImplementedError` 并给下载链接 |
| `XTRANS_PATH` | `:171,172,771,772` | 未设 | 编 `.cu` 时必需，提供 `bin/nvcc` 和 include |
| `XPYTORCH_CPP_EXTENSION_CXX11_ABI` | `:723,727` | 未设 | 覆盖 `_GLIBCXX_USE_CXX11_ABI` |
| `CONDA_PREFIX` | `:389,437`；`mock_torch.py:441,468` | 未设或环境前缀 | `xcudart/{include,lib}` 与 NVML 兼容库的查找前缀 |
| `CC` | `:577` | 未设 | 作为 `-ccbin` 传入 |

---

## C · 只在 `.so` 里的（仅名字，语义未验证）

以下字符串存在于 `XMLIR Python extension`（及部分在
`XMLIR runtime component` / `PyTorch integration component`），Python 侧完全不出现。
**[.so 内]** 默认值、解析方式、语义均**未验证**，仅供搜索定位。

**回落与调试**：`XACC_DEBUG_FALLBACK_ALL`、`XACC_INSTALL_RUNTIME_HOOKER`、
`XMLIR_DEBUG_FALLBACK_ALL`、`XMLIR_D_FORCE_FALLBACK_STR`、
`XMLIR_ENABLE_FALLBACK_TO_CPU_BOOL`、`XMLIR_DUMP_FALLBACK_OP_LIST_BOOL`、
`XMLIR_FALLBACK_OP_LIST_FILE_PATH`、`XMLIR_XDNN_PYTORCH_CHECK_ENABLE_FALLBACK_BOOL`、
`XMLIR_D_ENABLE_BACKTRACE`、`XMLIR_D_FILE_LOG_BOOL`、`XMLIR_D_PT_XPU_DEBUG_BOOL`、
`XMLIR_D_PT_XPU_DEBUG_FILE_STR`、`XMLIR_D_OP_COMPARISON_ERROR_COUNT_THRESHOLD_INT`、
`XMLIR_D_OP_COMPARISON_SPECIFY_OP_STR`

**显存 / L3 / 缓存**：`XMLIR_CACHING_ALLOC_ENABLED`、`XMLIR_DISABLE_CUDA_ALLOCATOR`、
`XMLIR_D_XPU_L3_SIZE`、`XMLIR_D_XPU_ENABLE_L3_SHARE`、`XMLIR_F_XPU_RESERVED_L3_INT`、
`XMLIR_F_XPU_RESERVED_WORKSPACE_INT`、`XMLIR_LRU_CACHE_SIZE`、`XMLIR_MEMCPY_RETRY_SYNC`、
`XMLIR_EMPTY_OP_INIT_ZERO`、`XMLIR_DISK_CACHE_DIR`、`XMLIR_PERSISTENT_CACHE_ENABLED`、
`XMLIR_ENABLE_H2D_SSE_COPY`

**GEMM / FC 调优**：`XMLIR_FC_AUTOTUNE_BOOL`、`XMLIR_FC_AUTOTUNE_FILE_NAME`、
`XMLIR_FC_AUTOTUNE_FILE_PATH`、`XMLIR_FC_AUTOTUNE_REQUESTED_ALGO_COUNT`、
`XMLIR_FC_AUTOTUNE_TOPK_VALUE`、`XMLIR_FC_TRY_TIMES_PER_ALGO`、`XMLIR_FC_BIAS_FUSION`、
`XMLIR_FC_TINTER_RES_TYPE`、`XMLIR_F_XPU_FC_GEMM_MODE`、`XMLIR_F_XPU_CONV_GEMM_MODE`、
`XMLIR_CONV_GEMM_DTYPE`、`XMLIR_ENABLE_XBLAS_ADDMM`、`XMLIR_MATMUL_FAST_MODE`、
`XMLIR_MATMUL_FAST_TUNE`、`XMLIR_MATMUL_SATBLE_MODE`（源串拼写为 SATBLE）

**算子行为**：`XMLIR_BADDBMM_DISPATCH_VALUE`、`XMLIR_BMM_DISPATCH_VALUE`、`XMLIR_BMM_TO_MOE`、
`XMLIR_FA_ACCUM_TYPE`、`XMLIR_FA_GEMM_TYPE`、`XMLIR_FUSED_SDP_CHOICE`、
`XMLIR_F_FAST_INDEX_PUT`、`XMLIR_USE_LSTM_KERNEL`、`XMLIR_CUDNN_ENABLED`、
`XMLIR_BF16_FIND_MAX_SDNN`、`XMLIR_ENABLE_FAST_BF16_INITIAL_SDNN`、`XMLIR_FORCE_USE_CPU_INIT`、
`XMLIR_XTRANS_OP_PRIOR`、`XMLIR_XTRANS_REGISTER_OPSELECT`、`XMLIR_PASS_OPTIONS`、
`XMLIR_F_XLA_TENSOR_UPDATE_SYNC_BOOL`、`XMLIR_F_XPU_ENABLED_BOOL`、`XMLIR_OP_USE_DEFAULT_STREAM`

**通信**：`TORCH_XCCL_NAN_CHECK`、`TORCH_XCCL_AVOID_RECORD_STREAMS`、
`TORCH_XCCL_DEFAUTL_PG_TIMEOUT_MILSEC`（拼写错误在二进制里）、
`TORCH_XCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING`、
`TORCH_XCCL_ENV_STORE_KEY_WITH_PG_NAME`、`TORCH_XCCL_ENV_WAIT_GDB`、
`XMLIR_XCCL_USE_ASYNC_OP`、`XMLIR_DIST_ASYNC_ISEND_IRECV`、`XMLIR_DIST_SINGLETON_STREAM`、
`XMLIR_DIST_USE_DEFAULT_STREAM`、`XMLIR_DIST_CHECK`、`XMLIR_DIST_CHECK_INF_NAN`、
`XMLIR_DIST_CHECK_INF_NAN_ASYNC`、`XMLIR_DIST_CHECK_INF_NAN_SAVE_DIR`、`XMLIR_ENABLE_NEW_PG`

**硬件 / 模拟**：`XPU_NUM_CLUSTER`、`XPU_NUM_SDNN`、`XPU_SIMULATOR_MODE`、`XPU_SUPPORT_IPC_EVENT`

---

下一页：[11-debug-tools.md](11-debug-tools.md)
