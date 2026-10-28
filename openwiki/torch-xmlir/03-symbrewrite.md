# 03 · symbrewrite：符号改写引擎与插件系统

[← 首页](README.md) · [← 02 导入 hook](02-import-hook.md) · [→ 04 Dispatcher 劫持](04-dispatch-hijack.md)

`symbrewrite/` 是全包最大的模块（13259 行）。名字容易误导：

> **它不是符号表重写，也不是字节码改写。就是 monkeypatch。**

具体只有两种操作：**[源码]**

- **符号替换** — `setattr(父模块或类, 叶子名, 新对象)`，同时把原对象存到 `__origin_<叶子名>`
- **模块替换** — `sys.modules[源名] = 目标模块`，于是后续 `import apex` 拿到昆仑的替身

（唯一一处 `ctypes.CDLL` 出现在 NVML 垫片调用里，是叶子细节，不是机制本身。）

## 插件作者 API

`$PKG/symbrewrite/__init__.py`（28 行，全文即 API）：

```python
def SYMBOL_REPLACE(symbol, obj, match_module=None):
    return SymbolReplaceEntry("K_SYMBOL_REPLACE").register(symbol, obj, match_module)

def MODULE_REPLACE(src, dst):
    return SymbolReplaceEntry("K_MODULE_REPLACE").register(src, dst)

def MODULE_DPES(model, deps):
    return SymbolRewriteRegister().add_module_deps(model, deps)
```

- `SYMBOL_REPLACE("torch.nn.Linear", MyLinear)` — 点分路径字符串 → 对象
- `MODULE_REPLACE("apex", "torch_xmlir.symbrewrite.plugins.apex")`；`dst=""` 会造一个空的
  `types.ModuleType` 桩（`module_replace.py:73-78`）
- `MODULE_DPES("torch", ["torch", "torch.distributed.run"])` — 声明「打补丁前这些子模块必须先 import」

注意 `SymbolReplaceEntry.register` **返回 `None`**（`symbrewrite/__init__.py:27-28`）。
插件里那些 `symbol_replacements += [SYMBOL_REPLACE(...), ...]` 构造出来的其实是一串
`None`，注册完全靠副作用完成。**[源码]**

## 打补丁的核心代码

`SymbolRewriteRegister`（`symbrewrite/symbol_rewrite.py:22-171`，`@Singleton`）
按点分路径第一段分 bucket。实际的 setattr：

```python
# $PKG/symbrewrite/symbol_rewrite.py:96-110
symb_list = source_symbol.split(".")
obj_ = sys.modules[symb_list[0]]
skip = False
for symb in symb_list[1:-1]:
    try:
        obj_ = getattr(obj_, symb)
    except Exception as e:
        print(f"WARNING: replace {source_symbol} error: {e}, skip!", file=sys.stderr)
        skip = True
if not skip:
    setattr(obj_, symb_list[-1], target)
    setattr(obj_, f"__origin_{symb_list[-1]}", source)
```

三个要点：

1. **`__origin_` 是 mock 拿回原实现的唯一途径**。例如
   `torch.distributed.__origin_init_process_group(...)`（`plugins/torch/mock_torch.py:69`）、
   `ProcessGroup.__origin_barrier`（`:425`）、
   `torch.nn.functional.__origin_linear`（`nn/linear.py:114`）。
2. `setattr` 无条件执行，所以 symbrewrite **也能新增本来不存在的属性**。
   profiler 插件就靠这个给 `EventList` 加了个 `dump` 方法（见 [11](11-debug-tools.md)）。
3. 中间层 getattr 失败只 skip 加打警告 —— 又一条静默失败路径。

两个额外机制在 `enable_rewrite_for_module`（`:112-142`）：

- **延迟构造**：`if hasattr(target, "is_wrapper") and target.is_wrapper: target = target()`（`:132-133`）。
  插件用它把替换用的 `nn.Module` 类定义在函数体里，只在补丁真正生效时才构造。
  例：`wrapper_mock_RMSNorm.is_wrapper = True`（`plugins/transformers/mock_transformers.py:26`）。
- **调用追踪**：`LOG_SYMBOL_REPLACE=1` 时每次调用都打印调用方 `file:line` 和替换实现的
  `file:line`（`:134-135` → `symbrewrite/event.py:7-39`）。**排查「到底走的是官方实现还是昆仑实现」时最有用的开关。**

成功时两个注册表都会往 stderr 打彩色横幅 `SYMBOL_REWRITE <module> success` /
`MODULE_REPLACE <module> success`（本机启动可见 `SYMBOL_REWRITE torch success`）。**[已验证]**

## xflags 插件开关系统

18 个插件，每个是 `symbrewrite/plugins/` 下一个子目录，由 `.xflags` 文件控制开关。

### 文件格式

行式 `name=value`，解析在 `xflags/xflags_cli.py:85-103`：

```python
for line in lines:
    if "=" not in line or "#" in line:
        continue
    parts = line.strip().split("=")
    if len(parts) == 2:
        name, value = parts[0].strip(), parts[1].strip()
        flags[name] = value
```

两个坑：**含 `#` 的行整行被丢**（所以 `foo=True # 注释` 会静默失效），
`=` 超过一个的行被跳过。值以字符串存，`get_flag`（`:66-74`）做大小写不敏感的
`"true"`/`"false"` → bool 转换，未知名字返回 `False`。**[源码]**

### 默认值（本构建实际状态）

`symbrewrite/plugins/plugins.default.xflags` 全文：

```
# 列表从上到下插件优先级依次降低
# 如果插件中出现重复改写，低优先级插件中改写被忽略
torch=True
deepspeed_0_14_4=False
torchbenchmark=False
torch_vision=True
torchaudio=True
reproduce=False
flash_attn=False
flash_attn_2_4_2=False
profiler=False
apex=False
transformers_4_32_1=False
transformers_4_42_3=False
transformers_4_43_3=False
transformers_4_49_0=False
transformers_4_52_3=False
mmsegmentation_1_0_0=False
swift=False
transformers=True
```

`plugins.xflags`（用户态文件）与之逐字节相同（仅少两行注释）。所以**默认开启的只有
`torch`、`torch_vision`、`torchaudio`、`transformers` 四个**。**[已验证]**

注意：**flash_attn 默认是关的**。要用昆仑的 FA 实现必须 `XFLAGS --enable flash_attn`。

### 优先级机制

`initialize_plugin_with_xflags()`（`$PKG/__init__.py:95-143`）在导入前把注册表顺序
`reverse()`（`:121`），从后往前导入。配合注册表「越靠前优先级越高」的约定，
结果是 **`torch` 插件最后注册、覆盖一切**。**[源码]**

第三方插件走命名约定：任何顶层可导入模块名以 `xpytorch_plugin` 开头的都会被无条件
导入（`:134-143`），且在 `load_flags` 里被自动注册为 `True`（`xflags_cli.py:61-64`）。

整个调用被 try/except 包住（`:274-281`），插件加载失败只 warn。

### `XFLAGS` 命令行

console_script，装在 `<env>/bin/XFLAGS`，实现 `xflags/xflags_cli.py:173-205`。
无子命令，四个选项：

| 选项 | 效果 |
|---|---|
| `XFLAGS --list` | 打印 ASCII logo + `XFLAG-NAMES / STATUS / DEFAULT` 三列表格（绿=开，红=关） |
| `XFLAGS --enable a,b` | 逗号分隔批量开启；未知名字抛 `ValueError` |
| `XFLAGS --disable a,b` | 批量关闭 |
| `XFLAGS --reset` | 用 default 覆盖用户文件 |

两个注意点 **[源码]**：

- 用户文件 `plugins.xflags` **写在 site-packages 内部**（`xflags_cli.py:32-37` 用
  `importlib.resources.path` 定位），不是 `$HOME`。所以 `XFLAGS --enable` 需要对已安装包的写权限，
  且**会被重装/升级覆盖**。容器里改了要注意持久化。
- `--reset` 与 `--enable` 同时给会互相打架：`:198` 先把 default 写盘，
  `:201` 的 `update_flags()` 又用**加载 reset 之前**的内存状态覆盖回去，等于 reset 被丢弃。

新增但没写进 flags 文件的插件目录会被自动注册为 `"False"`（`xflags_cli.py:47-53`），即默认关。

## 18 个插件速查

| 插件 | 行数 | 默认 | 干什么 |
|---|---|---|---|
| `torch` | 2894 | **开** | 主战场，~35 处改写，详见下节 |
| `flash_attn` | 2143 | 关 | 整模块替换，FA API → `custom_ops.mha_varlen_fwd/bwd` |
| `flash_attn_2_4_2` | 2139 | 关 | 同上，少 `flash_attn_kvpacked_func` |
| `profiler` | 1626 | 关 | 给 `EventList` 加 `dump` + 3 个离线 CLI，见 [11](11-debug-tools.md) |
| `apex` | 604 | 关 | 整模块替换 `amp_C`/`apex`/`fused_weight_gradient_mlp_cuda` |
| `transformers` | 232 | **开** | Qwen2/2.5-VL/3/3-VL/3.5/3.5-MoE 的 RMSNorm + RoPE 融合，15 处 |
| `torchaudio` | 250 | **开** | `amplitude_to_DB` / `mask_along_axis(_iid)` |
| `torch_vision` | 187 | **开** | **实际什么都不改** —— 6 处替换全被注释掉（`plugins/torch_vision/__init__.py:13-36`） |
| `mmsegmentation_1_0_0` | 389 | 关 | `SyncBatchNorm` + `Dropout2d`（fp16 bug workaround） |
| `reproduce` | 336 | 关 | 确定性复现；**import 即生效**（`__init__.py:38`） |
| `torchbenchmark` | 336 | 关 | 伪造 14 个 `pynvml` 函数 |
| `transformers_4_42_3` | 204 | 关 | BERT attn、Llama RMSNorm/RoPE、DataLoader/DDP 包装 |
| `deepspeed_0_14_4` | 161 | 关 | FusedAdam 映射；故意让 `assert_no_cuda_mismatch` 抛异常 |
| `transformers_4_49_0` | 111 | 关 | Qwen2 RMSNorm+RoPE |
| `swift` | 61 | 关 | `SwiftSft._get_data_collator`，前 2 步定长 padding 预热 |
| `transformers_4_52_3` | 56 | 关 | `check_torch_load_is_safe` 等 |
| `transformers_4_32_1` | 143 | 关 | Baichuan2-13B attention |
| `transformers_4_43_3` | 41 | 关 | `is_torch_sdpa_available` 等桩 |

## `torch` 插件改了什么

`plugins/torch/__init__.py:21-125` 无条件注册的部分，按意图分组 **[源码]**：

**NCCL → 昆仑集合通信**（详见 [07](07-distributed.md)）：三处 `ProcessGroupNCCL` 别名、
`is_nccl_available → lambda: True`、`init_process_group`/`new_group`/`get_backend` 包装。

**关掉 GPU codegen 路径**：`torch.jit.script → empty_decorator`（`:22-25`）、
`torch.jit.fuser → empty_with_decorator`（`:34-37`）、
`torch.backends.cudnn.enabled → False`（`:78-82`，注释：「防止官方 torch dispatch 到 cudnn 算子」）。

**CUDA IPC**：`reduce_tensor` / `rebuild_cuda_tensor` / `init_reductions` 换成走
`_XMLIRC._share_cuda_` / `_new_shared_cuda` 的版本（`:98-109`）。协议与官方完全一致，
只有两个碰驱动 handle 的原语不同。对应 Python 侧实现在 `$PKG/multiprocessing.py:101-203`。

**NVML**：`torch.cuda._raw_device_count_nvml` 和 `torch.cuda.utilization` 通过 `ctypes.CDLL`
加载随运行时提供的 NVML 兼容库（`mock_torch.py:437-504`），`nvmlUtilization_t` 结构体在 Python 里重新声明了一遍。

**杂项**：`torch.cuda._sleep` / `torch._C._cuda_sleep` → `_XMLIRC._xpu_sleep`（`:26-33`）；
`Logger.set_runtime_stats_and_log → empty_func`（`:70-73`）。

环境变量门控的块（默认值见 [10](10-env-vars.md)）：

| 变量 | 默认 | 效果 |
|---|---|---|
| `XMLIR_ENABLE_MOCK_TORCH_COMPILE` | `true` | **`torch.compile` 变成空装饰器**（`:128-139`） |
| `XMLIR_ENABLE_LINEAR_FC_FUSION` | `1` | 替换 `nn.Linear` / `F.linear`（`:141-146`） |
| `XPYTORCH_RUN_ENHANCE` | `0` | 替换 `torch.distributed.run.main` |
| `XMLIR_FORCE_USE_XPU_GRAPH` | `0` | 替换 `torch.cuda.CUDAGraph` / `graph` |
| `TORCH_C10D_USE_RANDOM_SLEEP` | `0` | rendezvous 加 rank 错峰 sleep |
| `TORCH_NN_INIT_IS_GPU_ALIGNED` | `0` | 启用 14 处 GPU 位对齐 RNG 改写（下节） |

### 最特殊的一块：GPU 位对齐 RNG

`TORCH_NN_INIT_IS_GPU_ALIGNED=1` 时，`nn.init.{normal_,uniform_,xavier_*,kaiming_*}`、
`torch.{rand,randn,randperm,manual_seed}`、`torch.cuda.manual_seed*`、`F.dropout`、
`nn.Dropout` 全被换成纯 Python/XPU 的 Philox 实现（`mock_torch_init.py`，1207 行）。

目标写在 `mock_torch_init.py:3-20`：实现尝试与目标 CUDA GPU 的 RNG 输出逐位对齐到 `<1e-6`。
因为 CUDA 的 RNG 输出依赖线程几何，所以需要知道目标卡的 SM 数；可通过
`GPU_NAME` 或 `GPU_SM_COUNT`+`GPU_THREADS_PER_SM` 指定。

`RANDPERM_KEY_BITS` 用于选择两条 `randperm` key 生成路径：`"32"` 使用 `uint4` 与 unroll 4，`"64"` 使用 `ulonglong2` 与 unroll 2（`gpu_config_manager.py:22-25`，`mock_torch_init.py:50-65`）。

`SET_PHILOX_VERSION` 可选 `xpu`（默认）/`python`/`cpp`；`cpp` 走
`$PKG/RNG compatibility component`。该组件的加载被
`$PKG/__init__.py:283-299` 单独守卫，注释说明原因：**它链的是 LLVM `libomp`，
而 sklearn 链 GNU `libgomp`，同进程共存可能 segfault**，所以默认不加载。**[源码]**

### torchrun 增强（`XPYTORCH_RUN_ENHANCE=1`）

比官方 torchrun 多三件事 **[源码]**：

1. 按 `xpu-smi topo -mo` 解析出的 NUMA 节点给每个 worker 绑 CPU（`enhance_context.py:20-44`）
2. 用 `SIGKILL` 代替优雅退出（`:58-72`, `:89-98`）
3. `XTORCHRUN_CLEAN` 非 0 时，启动前清理占着 `/dev/xpu` 的残留进程
   （`enhance_launch_torch2_5.py:142-156`）

第 3 点值得警惕：实现是 `lsof | grep /dev/xpu | ... | xargs kill -9`，
**会杀掉本机上除自己以外任何持有 XPU 设备的进程**。多人共用节点时不要开。

---

下一页：[04-dispatch-hijack.md](04-dispatch-hijack.md)
