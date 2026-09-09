# 06 · custom_ops 与 nn/ 融合算子

[← 首页](README.md) · [← 05 设备与运行时](05-device-runtime.md) · [→ 07 分布式](07-distributed.md)

## 自定义算子组件

`$PKG/__init__.py:215` 会加载自定义算子组件。`utils/custom_op_loader.py:17` 是对 `torch.ops.load_library` 的断言包装。

加载后，算子挂在 `torch.ops.custom_ops.*` 命名空间下。分析环境从编译组件的字符串表中提取到 **108 个 `custom_ops::` schema**。**[已验证]**

该组件依赖多个计算、GEMM、DNN、Flash Attention、集合通信和运行时组件。**[.so 内]** 数值实现位于这些编译组件中，Python 侧只能看到 schema 名和调用点。

`$PKG/__init__.py:218` 紧接着 `from torch_xmlir.distributed import tensor`，
顺序必须如此 —— DTensor 注册代码在导入期就引用 `torch.ops.custom_ops.*`（见 [07](07-distributed.md)）。

## 唯一默认生效的 nn 替换：Linear

```python
# $PKG/symbrewrite/plugins/torch/__init__.py:141-146
linear_fc_fusion_enable = int(os.getenv("XMLIR_ENABLE_LINEAR_FC_FUSION", 1))
if linear_fc_fusion_enable:
    symbol_replacements += [
        SYMBOL_REPLACE("torch.nn.Linear", linear.Linear),
        SYMBOL_REPLACE("torch.nn.functional.linear", linear.linear),
    ]
```

默认 **开**，运行时确认 `torch.nn.Linear.__module__ == 'torch_xmlir.nn.linear'`。**[已验证]**

注意 `int(os.getenv(...))` —— 传非数字值（如 `XMLIR_ENABLE_LINEAR_FC_FUSION=false`）会抛 `ValueError`。

### 三条分派路径

`nn/linear.py:102-138` 的 `_linear` 是真正的决策点 **[源码]**：

```python
def _linear(input, weight, bias=None, *, out=None) -> Tensor:
    if out is not None or force_nn_linear:
        return torch._C._nn.linear(input, weight, bias, out=out)

    if os.getenv("XMLIR_USE_HYDRA_LINEAR", "1") == "1":
        with stateful_config.make_tensors_stateful(input, weight):
            return hydra_linear(input, weight, bias)
    else:
        if not input.is_cuda:
            return torch.nn.functional.__origin_linear(input, weight, bias)
        ...
        output = LinearFunction.apply(input.reshape(-1, input.shape[-1]), weight, bias)
        return output.view(input.shape[:-1] + (weight.shape[0],))
```

| 条件 | 走哪条 |
|---|---|
| `FORCE_NN_LINEAR` 非空，或给了 `out=` | 官方 `torch._C._nn.linear`（最高优先级 kill switch） |
| `XMLIR_USE_HYDRA_LINEAR=1`（**默认**） | `hydrax.xaccelerator.linear.linear` |
| `XMLIR_USE_HYDRA_LINEAR=0` | 包内 `LinearFunction` → `torch.ops.custom_ops.linear` |

**关键发现：默认路径不在 torch_xmlir 里。** 真正的 fast-FC GEMM 位于独立的 `hydrax` 包：`hydrax/xaccelerator/linear/_linear.py:118 LinearFunction` → `_utils.py:26 fc_fusion_wrapper` → `hydrax.Hydra.*fc_fusion`。反向拆成 `linear_bwd_dgrad` / `linear_bwd_wgrad` / `linear_bwd_bgrad`。**排查 Linear 性能或精度问题时还需要检查 hydrax。**

`XMLIR_USE_HYDRA_LINEAR` 是**精确字符串比较 `== "1"`**，写 `"true"` 无效。

legacy 路径（`XMLIR_USE_HYDRA_LINEAR=0`）在 autocast 需要 cast 任一操作数时会退回官方
`torch._C._nn.linear`（`nn/linear.py:118-131`）。

`XMLIR_DYNAMO_WORKAROUND=1` 且无 `out=` 时，走自定义 op
`torch.ops._dynamo_workaround.linear`（定义在 `nn/linear.py:141-147`，
注册了 `CUDA`/`AutogradCUDA`/CPU），让 dynamo 能 trace（`:162`）。

一个行为提示：`Linear.reset_parameters`（`nn/linear.py:233-241`）是官方实现的逐字拷贝，
所以 `config.weight_initializer_enable` 这个名字虽然叫「初始化器使能」，**对这里毫无影响**。

## `nn/` 融合算子全表

全部经 `torch.ops.custom_ops.*` 调用 **[源码]**。除 Linear 外**都需要用户显式 import**
或由某个 symbrewrite 插件接入。

### 归一化

| 类 | 位置 | custom_ops |
|---|---|---|
| `RMSNormFunction` / `RMSNormLayer` | `nn/rms_norm.py:6,40` | `rms_layer_norm` / `rms_layer_norm_backward` |
| `DropoutAddLayernormFunction` | `nn/dropout_add_layernorm.py:12` | `dropout_add_layernorm_forward` / `_backward` |
| `DropoutAddRMSLayernormFunction` | `nn/dropout_add_rms_layernorm.py:12` | `dropout_add_rms_layernorm_forward` / `_backward` |

`rms_norm.py:17` 里 `rstd` 恒按 `float32` 分配；`:12-16` 算出来的 `stats_dtype` 是死代码。

**普通 LayerNorm 在 `nn/` 里没有替换。** `custom_ops::te_layer_norm` 存在但没有 Python 包装。
apex 插件（`plugins/apex/normalization/fused_layer_norm.py:23-31`）明确回落到
`torch.ops.aten.native_layer_norm`，注释：「APEX's FusedLayerNormAffineFunction is not
supported in KL yet」，只有 RMS 变体复用了 `torch_xmlir.nn.rms_norm.RMSNormFunction`。

### RoPE 家族（5 个文件 8 个实现）

| 类 | 位置 | custom_ops |
|---|---|---|
| `FusedRoPEFunc` / `FusedRope` | `nn/rope.py:7,118` | `fused_rope_forward` / `_backward`（支持 BLHD/BHLD/LBHD/THD、interleaved） |
| `RotaryPosEmbFunction` | `nn/rotary_pos_emb.py:8` | `rotary_pos_emb` / `_backward` |
| `RotaryPosEmbQKFunction` | `nn/rotary_pos_emb.py:50` | `rotary_pos_emb_qk` / `_qk_backward` |
| `RotaryPosEmbIdxFunction` | `nn/rotary_pos_emb_index.py:6` | `rotary_pos_emb_index` / `_index_backward` |
| `RotaryNoFreqsPosEmbAABBFunction` | `nn/rotary_no_freqs_pos_emb.py:6` | `rotary_no_freqs_pos_emb_forward` / `_backward` |
| `RotaryNoFreqsPosEmbABABFunction` | `nn/rotary_no_freqs_pos_emb.py:45` | `rotary_no_freqs_pos_emb_abab` / `_abab_backward` |
| `TeRotaryPosEmbFunction` | `nn/te_rotary_pos_emb.py:8` | `te_rotary_pos_emb` / `_backward` |
| `PrecomputeRotaryEmbeddingFunction` | `nn/precompute_rotary_embedding.py:12` | `precompute_rotary_embedding_forward` / `_backward` |

⚠️ `nn/rotary_pos_emb_index.py:18,73` 用 `layout is "BLHD"` —— **字符串用 `is` 比较**，
依赖 CPython interning，不保证成立。见 [12](12-pitfalls.md)。

### Attention 相关

`nn/` 里 **没有 SDPA，也没有 flash attention**。有的是：

| 类 | 位置 | custom_ops |
|---|---|---|
| `ScaledSoftmaxFunction` | `nn/scaled_softmax.py:5` | `scaled_softmax_forward` / `_backward` |
| `SoftmaxWithMaskFunction` | `nn/softmax_with_mask.py:5` | `softmax_with_mask` / `_backward`（mask 按 `mask.to(dtype) * (-10000.0)` 处理，`:15`） |
| `BmmFunction` | `nn/bmm.py:12` | `findmax` + `bmm_with_max` |
| `BaddbmmFunction` | `nn/baddbmm.py:12` | `findmax` + `baddbmm_with_max` |

`bmm`/`baddbmm` 是量化实现：先跑 `custom_ops.findmax` 求 per-tensor scale 到一个 64 元素
float buffer，除非 `XDNN_FC_GEMM_DTYPE == "float32"` 才跳过（`nn/bmm.py:31`, `nn/baddbmm.py:22`）。

**Flash attention 在插件里**（默认关，需 `XFLAGS --enable flash_attn`）：
`plugins/flash_attn/mock_flash_attn_interface.py:63,135` 调
`torch.ops.custom_ops.mha_varlen_fwd` / `mha_varlen_bwd`，其 kernel 位于编译组件中。
`XMLIR_FA_ACCUM_TYPE`（`float`|`float16`，默认 `float`）选 softmax-LSE 累加 dtype。
定长入口会自己合成 `cu_seqlens`（`:34-39`）。

**SDPA 在 C++ 层截获** —— `wrapper_CUDA___fused_sdp_choice`，见 [04](04-dispatch-hijack.md)。

### 激活与逐元素

| 类 | 位置 | custom_ops | 备注 |
|---|---|---|---|
| `SwiGLUFunction` | `nn/swiglu.py:6` | `swiglu_forward` / `_backward` | 带 `axis` 和 `turn`（哪一半过 SiLU）；`SwiGLUCpu`（`:68`）是 CPU 参考实现 |
| `GeluWithBiasFunction` | `nn/gelu_with_bias.py:12` | `gelu_with_bias` | **只有前向**（`:21` 注明 backward 未实现） |
| `T2iModulateFunction` | `nn/t2i_modulate.py:12` | `t2i_modulate` / `_backward` | DiT/PixArt 的 `x*(1+scale)+shift` |
| `DropoutAddFunction` | `nn/dropout_add.py:12` | `dropout_add_forward` / `_backward` | |
| `BiasDropoutAddFunction` | `nn/bias_dropout_add.py:12` | `bias_dropout_add_forward` | backward 复用 `dropout_add_backward` |

### 其它

- **ResNet unit 融合**：`nn/resnet_unit_fusion.py`（642 行，3 个 Function）和
  `nn/resnet_unit_fusion_frozen_bn.py`（650 行，3 个）。Conv+BN+ReLU(+shortcut) 融合。
  两个设计约束写在代码里：custom-op 框架**没有 C++ shape inference，所以形状在 Python 里算**
  （`:91-92` → `conv2d_shape_infer:10`），BN 参数必须 fp32（`check_bn_dtype_fp32:40`）。
  上一个 unit 返回的 `reserve_space` 会作为下一个的 `x_maxptr` 传入（`:82`），是量化 scale 串联方案。
- **Paged KV cache**：`nn/alloc_extend.py:60` → `custom_ops.alloc_extend_forward`，仅前向。
- **checkpoint**：`nn/checkpoint.py:112` 带 XPU RNG 状态保存/恢复的激活重算。
- **clip_grad**：`nn/clip_grad.py:69` 把 `clip_grad_norm` 拆成 pow/sum/pow，**避开 `torch.norm`**。
  同样思路的第二、三份拷贝在 `optimizer/BertAdam.py:59-120`。
- **mmcv**：`nn/mmcv_ops/mmcv_scatter_points.py`（96 行）只覆盖 `dynamic_scatter`；
  `mean`/`sum` 的 backward **直接 raise**（`:83-86`），只有 `max` 实现了。
  更广的 mmcv 算子（deform_conv、ms_deform_attn、roi_align、nms_rotated、ball_query…）
  在 `.so` 里有 C++ 实现但没有 Python 包装。

### Loss 算子

`nn/` 里**没有**。`.so` 里有 `custom_ops::hard_softmax_with_cross_entropy`、
`te_cross_entropy`、`sigmoid_focal_loss_forward/backward`，但没有 Python 调用点。**[已验证]**

## `config.py`：13 个 flag，只有 2 个有人用

`Config`（`config.py:4`）13 个布尔属性，`config = Config.from_env()` 在
`config.py:163` **模块导入期实例化** —— `import torch_xmlir` 之后再改环境变量无效。**[源码]**

### 直接来自环境变量

| 属性 | 环境变量 | 默认 |
|---|---|---|
| `disable_cast_cache` | `DISABLE_CAST_CACHE` | `0` |
| `use_fast_bf16_fc` | `XMLIR_ENABLE_FAST_FC` | `0` |
| `enable_param_state` | `XMLIR_LINEAR_CACHE_PARAM` | `1` |
| `enable_activ_state` | `XMLIR_LINEAR_CACHE_ACTIV` | `0` |
| `use_fast_bf16_fc_fwd_out` | `XMLIR_ENABLE_FAST_FC_FWD_OUT` | `0` |
| `use_fast_bf16_fc_bwd_dw` | `XMLIR_ENABLE_FAST_FC_BWD_DW` | `0` |
| `use_fast_bf16_fc_bwd_dx` | `XMLIR_ENABLE_FAST_FC_BWD_DX` | `0` |
| `batch_parallel_enabled` | `XMLIR_BATCH_PARALLEL` | `0` |
| `parallel_save_memory_enabled` | `XMLIR_PARALLEL_SAVE_MEMORY` | **`true`** |

### 派生（无独立环境变量）

| 属性 | 推导 | 行 |
|---|---|---|
| `weight_mix_precision` | `fwd_out != bwd_dx`（XOR） | `:129` |
| `weight_initializer_enable` | `fwd_out or bwd_dx` | `:130` |
| `use_fast_bf16` | `fwd_out or bwd_dw or bwd_dx` | `:131-135` |
| `use_cast_fc_fusion` | **硬编码 `0`**，与环境无关 | `:136` |

### 覆盖关系（两条，都是坑）

**(a) `XMLIR_ENABLE_FAST_FC` 是总开关，会屏蔽三个分方向变量**（`config.py:94-127`）：

```python
if use_fast_bf16_fc:
    use_fast_bf16_fc_fwd_out = True
    use_fast_bf16_fc_bwd_dw  = True
    use_fast_bf16_fc_bwd_dx  = True
    enable_param_state = os.getenv("XMLIR_LINEAR_CACHE_PARAM", "1").lower() in (...) and not disable_cast_cache
    enable_activ_state = os.getenv("XMLIR_LINEAR_CACHE_ACTIV", "0").lower() in (...) and not disable_cast_cache
else:
    use_fast_bf16_fc_fwd_out = ...  # 分别读三个变量
    enable_param_state = False
    enable_activ_state = False
```

两个后果：
1. 开了总开关就**不能再单独关掉某一个方向** —— 三个分变量根本不读。而且三者全 True 让
   `:129` 的 XOR 恒为 False，`weight_mix_precision` 在总开关模式下永远拿不到。
   混精度只能靠「总开关关闭 + 单独设分方向变量」达到。
2. 总开关关闭时，**两个 cast cache flag 被强制 False**（`:126-127`），
   无视 `XMLIR_LINEAR_CACHE_PARAM`。由于后者默认 `"1"`，参数 cast cache **看起来默认开、
   实际在原生环境里是关的**。

**(b) `DISABLE_CAST_CACHE` 是硬否决**（`:106`, `:114`），压过 `XMLIR_LINEAR_CACHE_*`。

### 谁在真正读这些 flag

对整个 site-packages grep 每个属性名：**只有 2 个有消费者** ——
`enable_param_state` 和 `enable_activ_state`，都在 `nn/linear.py:19-20` 喂给 hydrax 的
`StatefulConfig`。另外 11 个属性**零消费者**。**[已验证]**

但环境变量本身不是死的，只是别人绕过 `Config` 直接读 **[已验证]**：

- `transformer_engine/pytorch/module/linear.py:294,299,503,509`、`layernorm_linear.py:322,565`、
  `grouped_linear.py:916`、`cpp_extensions/gemm.py:135` 读 `XMLIR_ENABLE_FAST_FC` 和
  `XMLIR_BATCH_PARALLEL`，而且**解析方式不一致** —— `gemm.py:135` 用
  `in ("true","1","True")` 且无默认值，大小写敏感。
- `hydrax/xaccelerator/linear/_mixed_precision_linear.py:169-173` 读全部四个 `XMLIR_ENABLE_FAST_FC*`
  加 `DISABLE_CAST_CACHE`；`hydrax/xaccelerator/cache.py:156` 读 `DISABLE_CAST_CACHE`。
- `DISABLE_CAST_CACHE` 与 `XDNN_FC_GEMM_DTYPE` 都出现在编译组件的字符串表中，说明 C++ 侧也会读取这些变量。

**结论：`config.py` 是一个被生态绕过的历史遗留注册表，唯一还在履行的职责是 linear cast-cache 策略。
调 fast-FC 时应该直接设环境变量，并且注意 TE / hydrax 各自的解析差异。**

---

下一页：[07-distributed.md](07-distributed.md)
