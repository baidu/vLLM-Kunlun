#ifndef OPS_H
#define OPS_H
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
void rms_norm_xpu(torch::Tensor &output,
                  torch::Tensor &input,
                  torch::Tensor &weight,
                  double eps);
// inplace
void fused_add_rms_norm_xpu(torch::Tensor& input,     // [..., hidden_size]
                        torch::Tensor& residual,  // [..., hidden_size]
                        torch::Tensor& weight,    // [hidden_size]
                        double epsilon);

void silu_and_mul_xpu(torch::Tensor &output,
                      torch::Tensor &input);


void quick_gelu_xpu(torch::Tensor &output,
                      torch::Tensor &input);

// neox && gptj
void rotary_embedding(torch::Tensor &positions,
                      torch::Tensor& query,
                      torch::Tensor& key,
                      int64_t head_size,
                      torch::Tensor& cos_sin_cache,
                      bool is_neox);

void batched_rotary_embedding(torch::Tensor &positions,
                      torch::Tensor& query,
                      torch::Tensor& key,
                      int64_t head_size,
                      torch::Tensor& cos_sin_cache,
                      bool is_neox,
                      int64_t rot_dim,
                      torch::Tensor& offsets);

// x = 16 // sizeof(cache dtype)
void paged_attention_v1_xpu(
    torch::Tensor& out,    // [num_seqs, num_heads, head_size]
    torch::Tensor& query,  // [num_seqs, num_heads, head_size]
    torch::Tensor& key_cache,  // [num_blocks, num_kv_heads, block_size, head_size]
    torch::Tensor& value_cache,  // [num_blocks, num_kv_heads, block_size, head_size]
    int64_t num_kv_heads,
    double scale,
    torch::Tensor& block_tables,  // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor& seq_lens,      // [num_seqs]
    torch::Tensor& seq_lens_host, // [num_seqs]
    int64_t block_size,
    int64_t max_seq_len,
    const c10::optional<torch::Tensor>& alibi_slopes, // [num_heads]
    const std::string& kv_cache_dtype,
    double k_scale,
    double v_scale,
    int64_t tp_rank, int64_t blocksparse_local_blocks, // no used but to keep same with vllm-offficial
    int64_t blocksparse_vert_stride, int64_t blocksparse_block_size, // no used but to keep same with vllm-offficial
    int64_t blocksparse_head_sliding_step // no used but to keep same with vllm-offficial
    );

void reshape_and_cache(
    torch::Tensor& key,    // [num_tokens, num_heads, head_size]
    torch::Tensor& value,  // [num_tokens, num_heads, head_size]
    torch::Tensor&
        key_cache,  // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::Tensor&
        value_cache,  // [num_blocks, num_heads, head_size, block_size]
    torch::Tensor& slot_mapping,  // [num_tokens]
    const std::string& kv_cache_dtype,
    const double k_scale,
    const double v_scale);

void flash_attention_context_vllm_xpu(
    torch::Tensor& query,    // [num_tokens, num_heads, head_size]
    torch::Tensor& key,  // [num_tokens, num_kv_heads, head_size]
    torch::Tensor& value,  // [num_tokens, num_kv_heads, head_size]
    torch::Tensor& out, //    [num_tokens, num_heads, head_size]
    torch::Tensor& seq_lod, // [batch_size + 1]
    torch::Tensor& seq_lod_host, // [batch_size + 1]
    int64_t max_seq_len,
    int64_t max_kv_len,
    double scale,
    const c10::optional<torch::Tensor>& alibi_slopes, // [num_heads],
    const c10::optional<torch::Tensor>& key_cache, // [num_blocks, num_kv_heads, block_size, head_size]
    const c10::optional<torch::Tensor>& value_cache, // [num_blocks, num_kv_heads, block_size, head_size]
    const c10::optional<torch::Tensor>& block_tables, // [num_seqs, max_num_blocks_per_seq]
    const c10::optional<torch::Tensor>& kv_prefix_start_loc, // [lod of prefix]
    const c10::optional<torch::Tensor>& kv_prefix_start_loc_host, // [lod of prefix]
    const c10::optional<bool> is_causal // use causal mask or not, default true
);

void paged_attention_v2_xpu(
    torch::Tensor &out,
    torch::Tensor &exp_sums,
    torch::Tensor &max_logits,
    torch::Tensor &tmp_out,
    torch::Tensor &query, // [num_seqs, num_heads, head_size]
    torch::Tensor &
        key_cache, // [num_blocks, num_kv_heads, block_size, head_size]
    torch::Tensor &
        value_cache, // [num_blocks, num_kv_heads, block_size, head_size]
    int64_t num_kv_heads,
    double scale,
    torch::Tensor &block_tables, // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor &seq_lens,     // [num_seqs]
    torch::Tensor& seq_lens_host, // [num_seqs]
    int64_t block_size, int64_t max_seq_len,
    const c10::optional<torch::Tensor> &alibi_slopes, // [num_heads]
    const std::string &kv_cache_dtype, double k_scale, double v_scale,
    int64_t tp_rank, int64_t blocksparse_local_blocks, // no used but to keep same with vllm-offficial
    int64_t blocksparse_vert_stride, int64_t blocksparse_block_size, // no used but to keep same with vllm-offficial
    int64_t blocksparse_head_sliding_step // no used but to keep same with vllm-offficial
    );

void weight_only_quant_matmul_xpu(
    torch::Tensor &x,
    torch::Tensor &out,
    torch::Tensor &qweight,
    torch::Tensor &qscale
);



void multi_latent_attention_xpu(
    torch::Tensor q,
    torch::Tensor kv_rope_cache,
    torch::Tensor out,
    torch::Tensor block_tables,
    torch::Tensor seq_lens,
    double scale,
    int64_t max_seq_len
);

void outplace_fused_experts_xpu(
    torch::Tensor &hidden_states,
    torch::Tensor &output,
    torch::Tensor &w1,
    torch::Tensor &w2,
    torch::Tensor &topk_weights,
    torch::Tensor &topk_ids
);

void outplace_fused_experts_sorted_xpu(
    torch::Tensor &hidden_states,
    torch::Tensor &output,
    torch::Tensor &w1,
    torch::Tensor &w2,
    torch::Tensor &topk_weights,
    torch::Tensor &topk_ids
);


void grouped_topk_xpu(torch::Tensor &router_logits,
    torch::Tensor& score_bias,
    torch::Tensor& topk_weight,
    torch::Tensor& topk_ids,
    double scale,
    int64_t expert_group_num,
    int64_t moe_topk_group,
    int64_t moe_top_k);

void topk_softmax_xpu(torch::Tensor &topk_weights, /* [m, topk] */
                      torch::Tensor& topk_indices, /* [m, topk] */
                      torch::Tensor& token_expert_indices, /* no used in xpu */
                      torch::Tensor& gating_output /* [m, n] */
                      );
torch::Tensor weak_ref_tensor(torch::Tensor& tensor);

void dynamic_scaled_int8_quant_xpu(torch::Tensor &out,
                  torch::Tensor &x,
                  torch::Tensor &input_scale,
                  const c10::optional<torch::Tensor>& input_azp
);
void cutlass_scaled_mm_xpu(torch::Tensor& out, torch::Tensor const& a,
                       torch::Tensor const& b, torch::Tensor const& a_scales,
                       torch::Tensor const& b_scales,
                       std::optional<torch::Tensor> const& bias);

void castte_xpu(
    torch::Tensor& input,    // [num_tokens, hidden_dim]
    torch::Tensor& ouput,  // [num_tokens, hidden_dim]
    torch::Tensor& scale // [1]
);

void castte_per_token_xpu(
    torch::Tensor& input,       // [num_tokens, hidden_dim]
    torch::Tensor& ouput,       // [num_tokens, hidden_dim]
    torch::Tensor& scale        // [num_tokens]
);

void fc_fusion_castte_xpu(
    torch::Tensor& x,    // [num_tokens, in_dim]
    torch::Tensor& ouput,  // [num_tokens, out_dim]
    torch::Tensor& x_scale, // [1]
    torch::Tensor& qweight, // [out_dim, in_dim]
    torch::Tensor& qscale, // [1]
    const c10::optional<torch::Tensor>& bias // [out_dim]
);

void fc_fusion_castte_per_token_xpu(
    torch::Tensor& x,       // [num_tokens, in_dim]
    torch::Tensor& ouput,   // [num_tokens, out_dim]
    torch::Tensor& x_scale, // [num_tokens]
    torch::Tensor& qweight, // [out_dim, in_dim]
    torch::Tensor& qscale,  // [1]
    const c10::optional<torch::Tensor>& bias // [out_dim]
);

// trival cutlass
bool cutlass_scaled_mm_supports_fp8_xpu(int64_t cuda_device_capability);
bool cutlass_scaled_mm_supports_block_fp8_xpu(int64_t cuda_device_capability);

void outplace_split_norm_rope_xpu(
    torch::Tensor &qkv,
    torch::Tensor &cos_sin_cache,
    torch::Tensor &q_weight,
    torch::Tensor &k_weight,
    torch::Tensor &positions,
    torch::Tensor &q_emb_out,
    torch::Tensor &k_emb_out,
    torch::Tensor &v_out,
    const int64_t emb_batch_size,
    const int64_t max_seqlen,
    const int64_t head_num,
    const int64_t kv_head_num,
    const int64_t head_dim,
    const int64_t rotary_dim
);

void moe_fc_int8(
    torch::Tensor &hidden_states, // dtype : bfloat16
    torch::Tensor &output,
    torch::Tensor &w1,
    torch::Tensor &w1_scale,
    torch::Tensor &w2,
    torch::Tensor &w2_scale,
    torch::Tensor &topk_weights,
    torch::Tensor &topk_ids
);

#endif // OPS_H
