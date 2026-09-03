# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun torch-native override for ``vllm.v1.worker.gpu.sample.logit_bias``.

Leaves the upstream ``LogitBiasState`` / constants alone and reimplements
``apply_logit_bias`` (allowed-token masking, logit-bias addition, and min-token
stop-token masking) with per-request torch ops. ``num_tokens`` here equals the
number of sampled positions (~ request count), so the per-request loop is cheap.
"""

import torch
import vllm.v1.worker.gpu.sample.logit_bias as _up

_NEG_INF = float("-inf")


def apply_logit_bias(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    pos: torch.Tensor,
    num_allowed_token_ids: torch.Tensor,
    allowed_token_ids: torch.Tensor,
    num_logit_bias: torch.Tensor,
    logit_bias_token_ids: torch.Tensor,
    logit_bias: torch.Tensor,
    min_lens: torch.Tensor,
    num_stop_token_ids: torch.Tensor,
    stop_token_ids: torch.Tensor,
) -> None:
    num_tokens = logits.shape[0]
    idx = expanded_idx_mapping.tolist()
    na = num_allowed_token_ids.tolist()
    nb = num_logit_bias.tolist()
    ns = num_stop_token_ids.tolist()
    ml = min_lens.tolist()
    pos_l = pos.tolist()

    for t in range(num_tokens):
        rs = idx[t]
        row = logits[t]

        # Allowed token IDs: keep only those logits, mask the rest to -inf.
        n_allow = na[rs]
        if n_allow > 0:
            allowed = allowed_token_ids[rs, :n_allow].to(torch.long)
            saved = row[allowed].clone()
            row.fill_(_NEG_INF)
            row[allowed] = saved

        # Logit bias: add bias to the specified token IDs.
        n_bias = nb[rs]
        if n_bias > 0:
            tids = logit_bias_token_ids[rs, :n_bias].to(torch.long)
            row[tids] += logit_bias[rs, :n_bias].to(row.dtype)

        # Min tokens: block stop tokens until min length is reached.
        n_stop = ns[rs]
        if n_stop > 0 and pos_l[t] + 1 < ml[rs]:
            stids = stop_token_ids[rs, :n_stop].to(torch.long)
            row[stids] = _NEG_INF


# ``LogitBiasState`` resolves ``apply_logit_bias`` from the upstream module's
# globals; install the torch-native version there.
_up.apply_logit_bias = apply_logit_bias
