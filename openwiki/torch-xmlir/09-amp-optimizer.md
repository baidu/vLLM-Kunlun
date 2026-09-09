# 09 · AMP 与优化器

[← 首页](README.md) · [← 08 编译](08-compile-dynamo.md) · [→ 10 环境变量](10-env-vars.md)

## AMP：两套并行的 autocast 状态

关键结论先说：

> `torch_xmlir.amp.autocast` **只操作 `_XMLIRC` 自己的 autocast 状态，完全不碰 `torch.amp`**。
> 而且它 **fp16 only**、**无人 import**。实际生效的是官方 `torch.amp.autocast("cuda")`。

### `torch_xmlir.amp.autocast` 的实现

```python
# $PKG/amp/autocast_mode.py:64-89
def __enter__(self):
    if torch._jit_internal.is_scripting():
        assert self.fast_dtype is not None
        return self
    self.prev_cache_enabled = torch_xmlir._XMLIRC.is_autocast_cache_enabled()
    if self.device == "xla" or self.device == "xpu":
        self.prev = torch_xmlir._XMLIRC.is_autocast_enabled()
        self.prev_fastdtype = torch.float16
        torch_xmlir._XMLIRC.set_autocast_enabled(self._enabled)
        torch_xmlir._XMLIRC.autocast_increment_nesting()
    torch_xmlir._XMLIRC.set_autocast_cache_enabled(self._cache_enabled)

def __exit__(self, exc_type, exc_val, exc_tb):
    ...
    if self.device == "xla" or self.device == "xpu":
        if torch_xmlir._XMLIRC.autocast_decrement_nesting() == 0:
            torch_xmlir._XMLIRC.clear_autocast_cache()
        torch_xmlir._XMLIRC.set_autocast_enabled(self.prev)
    torch_xmlir._XMLIRC.set_autocast_cache_enabled(self.prev_cache_enabled)
    return False
```

它是官方 `autocast` 的结构性克隆，把每个 `torch._C.*` 换成了 `torch_xmlir._XMLIRC.*`。**[源码]**

8 个状态函数在 `XMLIR Python extension` 里，C++ namespace `torch_xmlir::amp`，
translation unit `autocast.cpp`（符号表确认，含 mangled
`_ZN11torch_xmlir3amp19is_autocast_enabledEv`）。**[已验证]** 逻辑不可见。

三个使用陷阱 **[源码]**：

1. `device_type` 默认 `"xpu"`（`:24`），**只接受 `"xla"`/`"xpu"`**，
   传 `"cuda"` 在 `:39-41` 直接 raise。**不能当 `torch.amp.autocast("cuda")` 的 drop-in。**
2. **只支持 fp16**：`:36-37` 硬编码 `self.fast_dtype = torch.float16`；
   `:53-61` 强制 `supported_dtype = [torch.float16]`，遇 bf16 或 fp32
   **只警告并静默关闭 autocast**。考虑到本构建整套 fast-FC 开关都是 bf16 导向的，
   这强烈提示该路径已经过时。
3. `cache_enabled` 默认取 `_XMLIRC` 当前值（`:42`），不是 `True`。

### 没有算子精度策略表

`amp/` 里**完全没有** `lower_precision_fp` / `fp32` / `promote` 之类的表，
也没有 `KERNEL(...)` 注册。

原因很直接：**XPU 偷了 CUDA dispatch key**（`$PKG/__init__.py:32`），
所以生效的 cast 策略就是 libtorch 自己的 CUDA autocast 策略。

旁证 **[源码]**：`nn/linear.py:122-123` 查的是
`torch.is_autocast_enabled()` / `torch.get_autocast_gpu_dtype()` —— **torch 的状态，
不是 `_XMLIRC` 的**；`hydrax/.../_linear.py:49-51` 用 `torch.get_autocast_dtype("cuda")`。

**[静态推断]** 「XPU 侧完全不存在 cast 策略表」是符号/字符串扫描的负面结果，
未反汇编确认，视为「很可能」而非「已证明」。

### GradScaler

`$PKG/amp/grad_scaler.py`（605 行）是官方 pre-2.0 `GradScaler` 的自包含拷贝
（`class GradScaler(object)` @ `:42`，不继承任何东西）。
默认值标准：`init_scale=2**16`、`growth_factor=2.0`、`backoff_factor=0.5`、
`growth_interval=2000`（`:110-117`）。

XPU 特有部分很小 **[源码]**：

- 可用性判断用 `torch_xmlir.xpu.is_available()`（`:118`）
- 类型断言期待 `"torch.xpu.FloatTensor"`（`:411`）
- 两个数值 kernel **是未修改的官方 ATen 调用**：
  `torch._amp_foreach_non_finite_check_and_unscale_`（`:242`）和
  `torch._amp_update_scale_`（`:431`）—— 靠 CUDA key dispatch 落到昆仑，不走 custom op
- 支持 `optimizer._step_supports_amp_scaling`（`:354-364`）

### `amp/` 是死路径

全 site-packages grep `torch_xmlir.amp` / `from torch_xmlir import amp`：
唯一命中是它自己的警告字符串（`grad_scaler.py:120`）。
`$PKG/__init__.py` 也不 import 它。**[已验证]**

**实践建议：用官方 `torch.amp.autocast("cuda", dtype=torch.bfloat16)` +
`torch.amp.GradScaler`，不要用 `torch_xmlir.amp`。**

（另：`symbrewrite/plugins/apex/amp/` 与此无关，只有 10 行 `_amp_state` 桩。）

## 优化器：11 个，分两档

`$PKG/optimizer/__init__.py:11-21` 的 `__all__` 只导出 9 个 ——
`fused_adam.FusedAdam` 和 `fused_lars.FusedLARS` **不在里面**，必须全路径 import。**[源码]**

### 第一档：真融合（每 step 一个 XPU kernel）

| 类 | 位置 | kernel |
|---|---|---|
| `FusedAdamW` | `fused_adamw.py:75` | `multi_tensor_adam`（快路径）/ `optimizer_AdamW`（回落） |
| `FusedAdam` | `fused_adam.py:41` | `multi_tensor_adam` |
| `ApexFusedAdamW` | `apex_fused_adamw.py:57` | `optimizer_AdamW`，逐参数 |
| `FusedLAMB` | `fused_lamb.py:7` | `optimizer_LAMB` |
| `FusedSGD` | `fused_sgd.py:6` | `optimizer_SGD` |
| `FusedLARS` | `fused_lars.py:5` | `optimizer_LARS` |

`FusedAdamW` 最完整，是唯一带真 multi-tensor-apply 快路径 + 守卫回落的：

```python
# $PKG/optimizer/fused_adamw.py:22-72（节选）
# Fast path: single XPU kernel for all params when step is uniform and dtype=float32.
# Only applicable on XPU/CUDA; multi_tensor_adam has no CPU implementation.
all_same_step = len(set(state_steps)) == 1
if (all_same_step
    and params[0].dtype == torch.float32
    and params[0].device.type != "cpu"):
    n_list = [p.numel() if p.shape != torch.Size([]) else 1 for p in params]
    torch.ops.custom_ops.multi_tensor_adam(
        grads, params, exp_avgs, exp_avg_sqs, n_list,
        lr, beta1, beta2, eps, weight_decay, state_steps[0],
        1,  # adam_w_mode=True  (decoupled weight decay)
        1,  # bias_correction=True
    )
    return
# Fallback: per-param scalar kernel (different steps or non-fp32 dtype).
```

**全档 fp32 only** **[源码]**：

- `FusedAdam` 对 fp16 抛 `"FusedAdam only support fp32 for now."`（`fused_adam.py:135`，
  `:136` 有 fp16 moment 的 `##TODO`）
- `ApexFusedAdamW` 抛 `"ApexFusedAdamW only support fp32."`（`apex_fused_adamw.py:256`），
  并拒绝 `amsgrad` / `capturable` / `master_weights` / `adam_w_mode=False`（`:135-146`）
- `FusedLAMB` 拒绝 `eager=False`（`fused_lamb.py:37-38`）

⚠️ `fused_adam.py` 两个 bug（见 [12](12-pitfalls.md)）：`state` 从
`for p in group["params"]` 循环泄漏出来（`:149`, `:166`），
`state["step"] += 1` 只递增最后一个参数的 step 却拿这个值喂整组；
且 fp16 在 `:135` 就 raise 了，`:148` 的 `if len(g_16) > 0` 分支不可达。

一个有意思的时间线证据：`FusedSGD.__init__`（`fused_sgd.py:73-74`）断言
`device.type == "cuda"` —— 写这个文件时 XPU 已经在冒充 CUDA 了。
而 `nn/parallel/` 还在假设 `"xla"`。**同一个包里能读出两个时代的痕迹。**

### 第二档：「XPU 友好」但不融合

`Adam`（`adam.py:65`）、`AdamW`（`adamw.py:66`）、`SGD`（`sgd.py:6`）、
`Lamb`（`lamb.py:24`）、`BertAdam`（`BertAdam.py:123`）**完全没有 custom op**。

它们的适配是另一回事：**把每个会变成 host 侧标量的 Python 数值提升成单元素设备张量，
避免 host↔device 同步**（lazy-tensor 时代的遗产）：

```python
# $PKG/optimizer/adam.py:34-37
beta1_pow = torch.tensor([beta1**step], dtype=torch.float32).to(grad.device)
beta2_pow = torch.tensor([beta2**step], dtype=torch.float32).to(grad.device)
bias_correction1 = torch.sub(1.0, beta1_pow)
bias_correction2 = torch.sub(1.0, beta2_pow)
```

`param.addcdiv_(exp_avg, denom, value=-step_size)` 也被拆成先算
`coef = exp_avg / denom * (-step_size)` 再 `param.add_(coef)`
（`adam.py:59-62`，同样模式见 `adamw.py:63`、`sgd.py:113,133`、`lamb.py:135-139`）。

⚠️ **精度妥协有明确记录**：`lamb.py:118-128` 把
`weight_norm == 0 or adam_norm == 0` 的守卫注释掉了，注释说
「Hardcode here to avoid the cpu sync problem in lazytensor」——
这让 `trust_ratio = weight_norm / adam_norm` **可能产出 inf/NaN**。
`BertAdam.py:59-120` 为同样原因自带 `local_norm` / `local_clip_grad_norm_`；
`nn/clip_grad.py:9` 的 `_decomposed_norm` 是同一思路的第三份拷贝。

### 未被使用的 kernel

`multi_tensor_l2norm`、`multi_tensor_scale`、`multi_tensor_cuda_lamb`、
`optimizer_LAMB_fused` 在 `custom-op component` 里存在，但 `optimizer/` 里**没有 Python 调用者**。
（其中 `multi_tensor_l2norm` / `multi_tensor_scale` 在 `distributed/tensor/` 里有 DTensor handler，
见 [07](07-distributed.md)；apex 插件的 `amp_c/amp_c.py:89` 也用到。）

### 各框架怎么接进来

| 框架 | 位置 | 做法 |
|---|---|---|
| apex | `plugins/apex/optimizers/fused_adam.py:5` | `class FusedAdam(Adam, ApexFusedAdamW)`；不支持的 apex kwarg 抛 `NotImplementedError` |
| apex | `plugins/apex/optimizers/fused_sgd.py:25` | 返回 `xmFusedSGD` |
| DeepSpeed | `plugins/deepspeed_0_14_4/mock_deepspeed.py:4,113-114` | 把 `FusedAdamW` 和 `ApexFusedAdamW` 加进 `ZERO_SUPPORTED_OPTIMIZERS`；`FusedLamb` 无条件 raise（`:57-58`）；故意让 `assert_no_cuda_mismatch` 抛 `MissingCUDAException` 逼 DeepSpeed 走 no-CUDA 分支（`:61-64`） |
| transformers | `plugins/transformers_4_42_3/__init__.py:8,12` | 替换成 `FusedAdamW`；另有 `MODULE_REPLACE("torch.optim.AdamW", "...FusedAdamW")`（`:10-13`，注意这是对一个**类路径**用了模块替换语义） |

**注意所有这些插件默认都是关的**（见 [03](03-symbrewrite.md) 的 xflags 表）。

---

下一页：[10-env-vars.md](10-env-vars.md)
