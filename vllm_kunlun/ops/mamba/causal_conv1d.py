# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao.
# Adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py

from typing import Optional, Union

import numpy as np
import torch

from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.triton_utils import tl, triton


@triton.jit()
def _causal_conv1d_fwd_kernel(  # continuous batching
    # Pointers to matrices
    x_ptr,  # (dim, cu_seqlen) holding `batch` of actual sequences + padded sequences
    w_ptr,  # (dim, width)
    bias_ptr,
    initial_states_ptr,  # conv_states_ptr
    cache_indices_ptr,  # conv_state_indices_ptr
    has_initial_states_ptr,
    query_start_loc_ptr,
    batch_ptr,
    token_chunk_offset_ptr,
    o_ptr,  # (dim, seqlen) - actually pointing to x_ptr
    # Matrix dimensions
    batch: tl.int32,  # actually padded_batch
    dim: tl.constexpr,
    seqlen: tl.int32,  # cu_seqlen
    num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    stride_x_seq: tl.constexpr,  # stride to get to next sequence,
    stride_x_dim: tl.constexpr,  # stride to get to next feature-value,
    stride_x_token: tl.
    constexpr,  # stride to get to next token (same feature-index, same sequence-index)
    stride_w_dim: tl.constexpr,  # stride to get to next dim-axis value
    stride_w_width: tl.constexpr,  # stride to get to next width-axis value
    stride_istate_seq: tl.constexpr,
    stride_istate_dim: tl.constexpr,
    stride_istate_token: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    HAS_INITIAL_STATES: tl.constexpr,
    HAS_CACHE: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,

):
    conv_states_ptr = initial_states_ptr
    conv_state_indices_ptr = cache_indices_ptr
    stride_conv_state_seq = stride_istate_seq
    stride_conv_state_dim = stride_istate_dim
    stride_conv_state_tok = stride_istate_token
    state_len = KERNEL_WIDTH - 1  # can be passed via argument if it's not the same as this value

    # one program handles one chunk in a single sequence
    # rather than mixing sequences - to make updating initial_states across sequences efficiently

    # single-sequence id
    idx_seq = tl.load(batch_ptr + tl.program_id(0))
    chunk_offset = tl.load(token_chunk_offset_ptr + tl.program_id(0))

    # BLOCK_N elements along the feature-dimension (channel)
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if idx_seq == pad_slot_id:
        return

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    # find the actual sequence length
    seqlen = sequence_end_index - sequence_start_index

    token_offset = BLOCK_M * chunk_offset
    segment_len = min(BLOCK_M, seqlen - token_offset)

    # base of the sequence
    x_base = x_ptr + sequence_start_index * stride_x_token + idx_feats * stride_x_dim  # [BLOCK_N,]

    if IS_CONTINUOUS_BATCHING:
        # cache_idx
        conv_state_batch_coord = tl.load(conv_state_indices_ptr + idx_seq).to(
            tl.int64)
    else:
        # cache_idx
        conv_state_batch_coord = idx_seq
    if USE_PAD_SLOT:  # noqa
        if conv_state_batch_coord == pad_slot_id:
            # not processing as this is not the actual sequence
            return
    conv_states_base = (conv_states_ptr +
                        (conv_state_batch_coord * stride_conv_state_seq) +
                        (idx_feats * stride_conv_state_dim))  # [BLOCK_N,]

    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]

    # Does 2 things:
    # 1. READ prior-block init-state data - [done by every Triton programs]
    # 2. update conv_state with new data [only by the Triton program handles chunk_offset=0]
    if chunk_offset == 0:
        # read from conv_states
        load_init_state = False
        if HAS_INITIAL_STATES:  # the new HAS_INITIAL_STATES
            load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(
                tl.int1)
        if load_init_state:
            # load from conv_states
            prior_tokens = conv_states_base + (state_len -
                                               1) * stride_conv_state_tok
            mask_w = idx_feats < dim
            if KERNEL_WIDTH == 2:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 3:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 4:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 5:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col3 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 3 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
        else:
            # prior-tokens are zeros
            if KERNEL_WIDTH >= 2:  # STRATEGY1
                # first chunk and does not have prior-token, so just set to 0
                col0 = tl.zeros((BLOCK_N, ), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 3:  # STRATEGY1
                col1 = tl.zeros((BLOCK_N, ), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 4:  # STRATEGY1
                col2 = tl.zeros((BLOCK_N, ), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 5:  # STRATEGY1
                col3 = tl.zeros((BLOCK_N, ), dtype=x_ptr.dtype.element_ty)

        # STEP 2:
        # here prepare data for updating conv_state
        if state_len <= seqlen:  # SMALL_CACHE=True (only move part of 'x' into conv_state cache)
            # just read from 'x'
            # copy 'x' data to conv_state
            # load only 'x' data (and set 0 before 'x' if seqlen < state_len)
            idx_tokens_last = (seqlen - state_len) + tl.arange(
                0, NP2_STATELEN)  # [BLOCK_M]
            x_ptrs = x_ptr + (
                (sequence_start_index + idx_tokens_last) *
                stride_x_token)[:, None] + (
                    idx_feats * stride_x_dim)[None, :]  # [BLOCK_M,BLOCK_N,]
            mask_x = ((idx_tokens_last >= 0)[:, None] &
                      (idx_tokens_last < seqlen)[:, None] &
                      (idx_feats < dim)[None, :]
                      )  # token-index  # token-index  # feature-index
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            new_conv_state = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]
            conv_states_ptrs_target = conv_states_base[None, :] + (
                idx_tokens_conv * stride_conv_state_tok)[:, None]

            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats
                                                             < dim)[None, :]
            # tl.debug_barrier()  #  NOTE: use this due to bug in Triton compiler
            tl.store(conv_states_ptrs_target, new_conv_state, mask)

        else:
            if load_init_state:
                # update conv_state by shifting left, i.e. take last few cols from conv_state + cols from 'x'
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

                conv_states_ptrs_source = (
                    conv_states_ptr +
                    (conv_state_batch_coord * stride_conv_state_seq) +
                    (idx_feats * stride_conv_state_dim)[None, :] +
                    ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:,
                                                                         None]
                )  # [BLOCK_M, BLOCK_N]
                mask = ((conv_state_batch_coord < num_cache_lines)
                        & ((idx_tokens_conv + seqlen) < state_len)[:, None]
                        & (idx_feats < dim)[None, :])
                conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)

                VAL = state_len - seqlen

                x_ptrs = x_base[None, :] + (
                    (idx_tokens_conv - VAL) *
                    stride_x_token)[:, None]  # [BLOCK_M, BLOCK_N]

                mask_x = ((idx_tokens_conv - VAL >= 0)[:, None] &
                          (idx_tokens_conv - VAL < seqlen)[:, None] &
                          (idx_feats < dim)[None, :]
                          )  # token-index  # token-index  # feature-index
                loaded_x = tl.load(x_ptrs, mask_x, 0.0)

                # tl.debug_barrier(
                # )  # need this due to the bug in tl.where not enforcing this when data is the result of another tl.load
                new_conv_state = tl.where(
                    mask, conv_state, loaded_x
                )  # BUG in 'tl.where'  which requires a barrier before this
                conv_states_ptrs_target = conv_states_base + (
                    idx_tokens_conv *
                    stride_conv_state_tok)[:, None]  # [BLOCK_M, BLOCK_N]
                mask = (idx_tokens_conv
                        < state_len)[:, None] & (idx_feats < dim)[None, :]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)
            else:  # load_init_state == False
                # update conv_state by shifting left, BUT
                # set cols prior to 'x' as zeros + cols from 'x'
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

                VAL = state_len - seqlen

                x_ptrs = x_base[None, :] + (
                    (idx_tokens_conv - VAL) *
                    stride_x_token)[:, None]  # [BLOCK_M, BLOCK_N]

                mask_x = ((idx_tokens_conv - VAL >= 0)[:, None] &
                          (idx_tokens_conv - VAL < seqlen)[:, None] &
                          (idx_feats < dim)[None, :]
                          )  # token-index  # token-index  # feature-index
                new_conv_state = tl.load(x_ptrs, mask_x, 0.0)

                conv_states_ptrs_target = conv_states_base + (
                    idx_tokens_conv *
                    stride_conv_state_tok)[:, None]  # [BLOCK_M, BLOCK_N]
                mask = (idx_tokens_conv
                        < state_len)[:, None] & (idx_feats < dim)[None, :]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)

    else:  # chunk_offset > 0
        # read prior-token data from `x`
        load_init_state = True
        prior_tokens = x_base + (token_offset - 1) * stride_x_token
        mask_w = idx_feats < dim
        if KERNEL_WIDTH == 2:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')
        if KERNEL_WIDTH == 3:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')
        if KERNEL_WIDTH == 4:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')
            conv_states_ptrs = prior_tokens - 2 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')
        if KERNEL_WIDTH == 5:
            # ruff: noqa: F841
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col3 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')
            conv_states_ptrs = prior_tokens - 2 * stride_x_token  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')
            conv_states_ptrs = prior_tokens - 3 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')

    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias,
                              other=0.0).to(tl.float32)  # [BLOCK_N]
    else:
        acc_preload = tl.zeros((BLOCK_N, ), dtype=tl.float32)

    x_base_1d = x_base + token_offset * stride_x_token  # starting of chunk

    # PRE-LOAD WEIGHTS
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)
    mask_x_1d = idx_feats < dim
    for idx_token in range(segment_len):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):

            if KERNEL_WIDTH == 2:
                if j == 1:  # KERNEL_WIDTH-1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

            acc += matrix_x * matrix_w  # [BLOCK_N]

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < segment_len) & (
            idx_feats < dim)  # token-index  # feature-index
        o_ptrs = o_ptr + (sequence_start_index + token_offset + idx_token
                          ) * stride_o_token + (idx_feats * stride_o_dim)

        tl.store(o_ptrs, acc, mask=mask_1d)


def causal_conv1d_fn_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Union[torch.Tensor, None],
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    metadata=None,
    validate_data=False,
):
    """support varlen + continuous batching when x is 2D tensor

    x: (dim,cu_seq_len)
        cu_seq_len = total tokens of all seqs in that batch
        sequences are concatenated from left to right for varlen
    weight: (dim, width)
    conv_states: (...,dim,width - 1) itype
        updated inplace if provided
        [it use `cache_indices` to get the index to the cache of conv_state for that sequence

        conv_state[cache_indices[i]] for seq-i - to be used as initial_state when has_initial_state[i] = True
             and after that conv_state[cache_indices[i]] need to be shift-left and updated with values from 'x'
        ]
    query_start_loc: (batch + 1) int32
        The cumulative sequence lengths of the sequences in
        the batch, used to index into sequence. prepended by 0.
        if
        x = [5, 1, 1, 1] <- continuous batching (batch=4)
        then
        query_start_loc = [0, 5, 6, 7, 8] <- the starting index of the next sequence; while the last value is
           the ending index of the last sequence
        [length(query_start_loc)-1 == batch]
        for example: query_start_loc = torch.Tensor([0,10,16,17]),
        x.shape=(dim,17)
    cache_indices: (batch)  int32
        indicates the corresponding state index,
        like so: conv_state = conv_states[cache_indices[batch_id]]
    has_initial_state: (batch) bool
        indicates whether should the kernel take the current state as initial
        state for the calculations
        [single boolean for each sequence in the batch: True or False]
    bias: (dim,)
    activation: either None or "silu" or "swish" or True
    pad_slot_id: int
        if cache_indices is passed, lets the kernel identify padded
        entries that will not be processed,
        for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
        in this case, the kernel will not process entries at
        indices 0 and 3

    out: same shape as `x`
    """
    if isinstance(activation, bool) and activation:
        activation = "silu"

    args = None
    # Store original dtype to cast back at the end
    original_x_dtype = x.dtype
    x = x.to(conv_states.dtype)
    out = torch.empty_like(x)
    if metadata is not None:
        nums_dict = metadata.nums_dict
        args = nums_dict
        batch_ptr = metadata.batch_ptr
        token_chunk_offset_ptr = metadata.token_chunk_offset_ptr
    else:
        seqlens = query_start_loc.diff().to('cpu')
        args = seqlens
        MAX_NUM_PROGRAMS = 1024

        batch_ptr = torch.full(
            (MAX_NUM_PROGRAMS, ),
            PAD_SLOT_ID,
            dtype=torch.int32,
            device=x.device
        )  # tracking which seq-idx the Triton program is handling
        token_chunk_offset_ptr = torch.full(
            (MAX_NUM_PROGRAMS, ),
            PAD_SLOT_ID,
            dtype=torch.int32,
            device=x.device
        )  # tracking BLOCK_M-based index in the sequence the Triton program is handling

    is_channel_last = (x.stride(0) == 1) & (x.stride(1) > 1)
    dim, cu_seqlen = x.shape
    _, width = weight.shape
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    padded_batch = query_start_loc.size(0) - 1
    stride_x_seq = 0
    stride_x_dim = x.stride(0)
    stride_x_token = x.stride(1)
    stride_w_dim = weight.stride(0)
    stride_w_width = weight.stride(1)
    stride_istate_seq = 0
    stride_istate_dim = 0
    stride_istate_token = 0
    num_cache_lines = 0
    if conv_states is not None:
        # extensions to support vLLM:
        # 1. conv_states is used to replaced initial_states
        # 2. conv_states serve as a cache with num cache lines can be larger than batch size
        # 3. mapping from sequence x[idx] to a cache line at index as specified via cache_indices[idx]
        # 4. computation can be skipped if cache_indices[idx] == pad_slot_id
        num_cache_lines = conv_states.size(0)
        assert (num_cache_lines == conv_states.shape[0]
                and dim == conv_states.shape[1]
                and width - 1 <= conv_states.shape[2])
        stride_istate_seq = conv_states.stride(0)
        stride_istate_dim = conv_states.stride(1)
        stride_istate_token = conv_states.stride(2)
        assert stride_istate_dim == 1
    if out.dim() == 2:
        stride_o_seq = 0
        stride_o_dim = out.stride(0)
        stride_o_token = out.stride(1)
    else:
        stride_o_seq = out.stride(0)
        stride_o_dim = out.stride(1)
        stride_o_token = out.stride(2)
    stride_cache_indices = cache_indices.stride(
        0) if cache_indices is not None else 0

    if validate_data:
        assert x.dim() == 2
        assert query_start_loc is not None
        assert query_start_loc.dim() == 1
        assert x.stride(0) == 1 or x.stride(1) == 1
        if bias is not None:
            assert bias.dim() == 1
            assert dim == bias.size(0)
        if cache_indices is not None:
            assert cache_indices.dim() == 1
            assert padded_batch == cache_indices.size(0)
        if has_initial_state is not None:
            assert has_initial_state.size() == (padded_batch, )
            assert conv_states is not None, "ERROR: `has_initial_state` is used, which needs also `conv_states`"
        assert weight.stride(1) == 1
        assert (dim, width) == weight.shape
        assert is_channel_last, "Need to run in channel-last layout"

    if metadata is None:

        def num_program(META, seqlens):
            tot = 0

            mlist = []
            offsetlist = []  # type: ignore

            nums = -(-seqlens // META["BLOCK_M"])

            tot = nums.sum().item()
            mlist = np.repeat(np.arange(len(nums)), nums)
            for idx, num in enumerate(nums):
                offsetlist.extend(
                    range(num)
                )  # chunk-idx if a sequence is split into multiple chunks

            if META["batch_ptr"].nelement() < len(mlist):
                newlen = len(mlist) + 1
                META["batch_ptr"].resize_(newlen).fill_(PAD_SLOT_ID)
                META["token_chunk_offset_ptr"].resize_(newlen).fill_(
                    PAD_SLOT_ID)

            if META["batch_ptr"].nelement() >= len(mlist):
                META["batch_ptr"][0:len(mlist)].copy_(
                    torch.from_numpy(np.array(mlist)))
                META["token_chunk_offset_ptr"][0:len(mlist)].copy_(
                    torch.from_numpy(np.array(offsetlist)))

            META["batch_ptr"] = META["batch_ptr"].to(META["x_ptr"].device)
            META["token_chunk_offset_ptr"] = META["token_chunk_offset_ptr"].to(
                META["x_ptr"].device)
            return tot
    else:

        def num_program(META, nums_dict):
            tot = nums_dict[META["BLOCK_M"]]['tot']

            mlist = nums_dict[META["BLOCK_M"]]['mlist']
            mlist_len = nums_dict[META["BLOCK_M"]]['mlist_len']

            offsetlist = nums_dict[META["BLOCK_M"]]['offsetlist']

            if nums_dict[META["BLOCK_M"]]["batch_ptr"] is not None:
                META["batch_ptr"] = nums_dict[META["BLOCK_M"]]["batch_ptr"]
                META["token_chunk_offset_ptr"] = nums_dict[
                    META["BLOCK_M"]]["token_chunk_offset_ptr"]
            else:
                if META["batch_ptr"].nelement() < mlist_len:
                    newlen = mlist_len + 1
                    META["batch_ptr"].resize_(newlen).fill_(PAD_SLOT_ID)
                    META["token_chunk_offset_ptr"].resize_(newlen).fill_(
                        PAD_SLOT_ID)

                if META["batch_ptr"].nelement() >= mlist_len:
                    META["batch_ptr"][0:mlist_len].copy_(mlist)
                    META["token_chunk_offset_ptr"][0:mlist_len].copy_(
                        offsetlist)
            return tot

    def grid(META):
        return (
            num_program(META, args),
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    if batch_ptr.device != x.device:
        batch_ptr = batch_ptr.to(x.device)
        token_chunk_offset_ptr = token_chunk_offset_ptr.to(x.device)

    _causal_conv1d_fwd_kernel[grid](
        # Pointers to matrices
        x,
        weight,
        bias,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        batch_ptr,
        token_chunk_offset_ptr,
        out,
        # Matrix dimensions
        padded_batch,
        dim,
        cu_seqlen,
        num_cache_lines,
        # stride
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        # others
        pad_slot_id,
        # META
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        HAS_INITIAL_STATES=has_initial_state is not None,
        HAS_CACHE=conv_states is not None,
        IS_CONTINUOUS_BATCHING=cache_indices is not None,
        USE_PAD_SLOT=pad_slot_id is not None,
        NP2_STATELEN=np2_statelen,
        #launch_cooperative_grid=True
        BLOCK_M=8,
        BLOCK_N=256,
        num_stages=2,
        groups_per_cluster = np2_statelen,
        isCloseUnrollControl = True,
        isCloseVectorization = True,
        is_use_mask_zero = True
    )
    return out

import torch.nn.functional as F

def causal_conv1d_single(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    return_final_states: bool = False,
    final_states_out: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1)
    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape

    if initial_states is None:
        out = F.conv1d(x,
                       weight.unsqueeze(1),
                       bias,
                       padding=width - 1,
                       groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]

    if return_final_states:
        if True or x.shape[-1] < width - 1:
            final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(dtype_in)  # (batch, dim, width - 1)
        elif x.shape[-1] > width - 1:
            final_states = x[..., -width + 1:].to(dtype_in)
        else:
            final_states = x.to(dtype_in)
        if final_states_out is not None:
            final_states_out[..., :(width - 1)].copy_(final_states)
        else:
            final_states_out = final_states
    # out = out.to(dtype=dtype_in)
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states_out)



def causal_conv1d_fn9(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    conv_states: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    metadata=None,
    validate_data=False,
):
    """
    x: (batch, dim, seqlen) or (dim,cu_seq_len) for varlen
        sequences are concatenated from left to right for varlen
    weight: (dim, width)
    bias: (dim,)
    query_start_loc: (batch + 1) int32
        The cumulative sequence lengths of the sequences in
        the batch, used to index into sequence. prepended by 0.
        for example: query_start_loc = torch.Tensor([0,10,16,17]),
        x.shape=(dim,17)
    cache_indices: (batch)  int32
        indicates the corresponding state index,
        like so: conv_state = conv_states[cache_indices[batch_id]]
    has_initial_state: (batch) bool
        indicates whether should the kernel take the current state as initial
        state for the calculations
    conv_states: (...,dim,width - 1) itype
        updated inplace if provided
    activation: either None or "silu" or "swish"
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    if x.stride(-1) != 1:
        x = x.contiguous()
    bias = bias.contiguous() if bias is not None else None

    out_ref = []
    out_ref_b = []
    seqlens = query_start_loc[1:] - query_start_loc[:-1]
    seqlens = seqlens.tolist()
    splits = torch.split(x, seqlens, dim=-1)

    for i in range(len(seqlens)):
        x_s = splits[i]
        if cache_indices[i] == PAD_SLOT_ID:
            continue
        out_ref_b.append(
            causal_conv1d_single(
                x_s,
                weight,
                bias,
                activation=activation,
                return_final_states=True,
                final_states_out=conv_states[cache_indices[i]].unsqueeze(0),
                initial_states=conv_states[cache_indices[i]]
                if has_initial_state[i] else None))
    out_ref.append(torch.cat([t[0] for t in out_ref_b], dim=-1))
    out_ref_tensor = torch.cat(out_ref, dim=0)
    # out_ref_tensor = torch.cat([t[0] for t in out_ref_b], dim=-1)
    # out_ref_tensor = F.silu(out_ref_tensor)
    from vllm.distributed import get_tensor_model_parallel_rank
    if get_tensor_model_parallel_rank() == 0 and len(seqlens) > 1:
        print("x", x)
        print("out",out_ref_tensor.shape, out_ref_tensor)
        print("cache", conv_states[cache_indices].shape, conv_states[cache_indices])
    if len(seqlens) > 1:
        raise
    return out_ref_tensor

def causal_conv1d_single3(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    return_final_states: bool = False,
    final_states_out: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
):
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape

    if initial_states is None:
        out = F.conv1d(x,
                       weight.unsqueeze(1),
                       bias,
                       padding=width - 1,
                       groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]

    if return_final_states:
        if x.shape[-1] < width - 1:
            final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(dtype_in)  # (batch, dim, width - 1)
        elif x.shape[-1] > width - 1:
            final_states = x[..., -width + 1:].to(dtype_in)
        else:
            final_states = x.to(dtype_in)
        if final_states_out is not None:
            final_states_out[..., :(width - 1)].copy_(final_states)
        else:
            final_states_out = final_states
    out = out.to(dtype=dtype_in)
    # out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states_out)

def causal_conv1d_fn3(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    conv_states: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    metadata=None,
    validate_data=False,
):
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    if x.stride(-1) != 1:
        x = x.contiguous()
    bias = bias.contiguous() if bias is not None else None

    out_ref = []
    out_ref_b = []
    seqlens = query_start_loc[1:] - query_start_loc[:-1]
    seqlens = seqlens.tolist()
    splits = torch.split(x, seqlens, dim=-1)

    if metadata.num_decodes > 0:
        decode_x = x[..., :metadata.num_decodes].transpose(0, 1).unsqueeze(-1)
        decode_out = torch_causal_conv1d_update(
            decode_x,
            conv_states,
            weight,
            bias,
            activation,
            conv_state_indices=cache_indices[:metadata.num_decodes]
        ).transpose(0, 1)

    for i in range(metadata.num_decodes, len(seqlens)):
        x_s = splits[i]
        if cache_indices[i] == PAD_SLOT_ID:
            continue
        out_ref_b.append(
            causal_conv1d_single(
                x_s,
                weight,
                bias,
                activation=activation,
                return_final_states=True,
                final_states_out=conv_states[cache_indices[i]].unsqueeze(0),
                initial_states=conv_states[cache_indices[i]]
                if has_initial_state[i] else None))
    # out_ref.append(torch.cat([t[0] for t in out_ref_b], dim=-1))
    # out_ref_tensor = torch.cat(out_ref, dim=0)
    out_ref_tensor = torch.cat([t[0] for t in out_ref_b], dim=-1)
    out_ref_tensor = F.silu(out_ref_tensor)
    if metadata.num_decodes > 0:
        out_ref_tensor = torch.cat([decode_out, out_ref_tensor], dim=-1)
    return out_ref_tensor

def causal_conv1d_fn8(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    conv_states: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    metadata=None,
    validate_data=False,
):
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    bias = bias.contiguous() if bias is not None else None

    out_ref = []
    out_ref_b = []
    seqlens = query_start_loc[1:] - query_start_loc[:-1]
    seqlens = seqlens.tolist()
    splits = torch.split(x, seqlens, dim=-1)

    import kunlun_ops
    if metadata.num_decodes > 0:
        decode_x = x[..., :metadata.num_decodes].transpose(0, 1).unsqueeze(1).contiguous()
        decode_out = torch.empty_like(decode_x)
        decode_cache_indices = cache_indices[:metadata.num_decodes]
        kunlun_ops.causal_conv1d_update(
                    decode_x,
                    weight,
                    decode_out,
                    conv_states,
                    None,
                    bias,
                    conv_state_indices_cpu=decode_cache_indices.cpu(),
                    conv_state_indices_xpu=decode_cache_indices,
                    act=activation,
                    is_ncw=False
        )
        decode_out = decode_out.squeeze(1).transpose(0, 1)

    prefill_x = x[..., metadata.num_decodes:].contiguous()
    out = torch.empty_like(prefill_x)
    prefill_query_start_loc = query_start_loc[metadata.num_decodes:]
    prefill_query_start_loc -= prefill_query_start_loc[0]
    batch_size = prefill_query_start_loc.shape[0] - 1
    prefill_initial_state = torch.zeros(batch_size, dtype=torch.int32, device=has_initial_state.device)
    dim = prefill_x.shape[-2]
    cu_seqlen = prefill_x.shape[-1]
    width = weight.shape[-1]
    tmp_cache = torch.empty(batch_size, conv_states.shape[2], conv_states.shape[1], dtype=conv_states.dtype, device=conv_states.device)
    num_cache_lines = batch_size
    state_width = conv_states.shape[-1]
    tmp_indices = torch.arange(batch_size, dtype=torch.int32, device=cache_indices.device)
    tmp_indices_cpu = tmp_indices.cpu()

    kunlun_ops.causal_conv1d_fn(prefill_x, out, dim, cu_seqlen, weight, width, tmp_cache, num_cache_lines,
                                state_width, prefill_query_start_loc.cpu(), prefill_query_start_loc, batch_size, bias,
                                cache_indices_cpu=tmp_indices_cpu, cache_indices_xpu=tmp_indices,
                                has_initial_state_cpu=prefill_initial_state.cpu(), has_initial_state_xpu=prefill_initial_state,
                                act=activation)
    tmp_cache = tmp_cache.transpose(-1, -2).contiguous()
    prefill_indices = cache_indices[metadata.num_decodes:]
    conv_states[prefill_indices] = tmp_cache
    if metadata.num_decodes > 0:
        out = torch.cat([decode_out, out], dim=-1)
    print("out", out.shape, out)
    print("conv_states", conv_states[prefill_indices])
    raise
    return out



    for i in range(metadata.num_decodes, len(seqlens)):
        x_s = splits[i]
        if cache_indices[i] == PAD_SLOT_ID:
            continue
        out_ref_b.append(
            causal_conv1d_single(
                x_s,
                weight,
                bias,
                activation=activation,
                return_final_states=True,
                final_states_out=conv_states[cache_indices[i]].unsqueeze(0),
                initial_states=conv_states[cache_indices[i]]
                if has_initial_state[i] else None))
    # out_ref.append(torch.cat([t[0] for t in out_ref_b], dim=-1))
    # out_ref_tensor = torch.cat(out_ref, dim=0)
    out_ref_tensor = torch.cat([t[0] for t in out_ref_b], dim=-1)
    out_ref_tensor = F.silu(out_ref_tensor)
    if metadata.num_decodes > 0:
        out_ref_tensor = torch.cat([decode_out, out_ref_tensor], dim=-1)
    return out_ref_tensor

def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    conv_states: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    metadata=None,
    validate_data=False,
):

    x = x.contiguous()
    out = torch.empty_like(x)
    has_initial_state = has_initial_state.to(torch.int32)
    dim = x.shape[-1]
    cu_seqlen = x.shape[-2]
    width = weight.shape[-1]
    num_cache_lines = conv_states.shape[0]
    state_width = conv_states.shape[-2]
    batch_size = query_start_loc.shape[0] - 1
    stride = conv_states.stride()[0]

    import kunlun_ops
    kunlun_ops.causal_conv1d_fn(x, out, dim, cu_seqlen, weight, width, conv_states, num_cache_lines,
                                state_width, query_start_loc.cpu(), query_start_loc, batch_size, bias,
                                cache_indices_cpu=cache_indices.cpu(), cache_indices_xpu=cache_indices,
                                has_initial_state_cpu=has_initial_state.cpu(), has_initial_state_xpu=has_initial_state,
                                act="SWISH", state_seq_stride=stride)
    # out = torch.nn.functional.silu(out)
    return out

def torch_causal_conv1d_update(
    hidden_states,
    conv_state,
    weight,
    bias=None,
    activation=None,
    conv_state_indices=None
):
    _, hidden_size, seq_len = hidden_states.shape
    tmp_conv_state = conv_state[conv_state_indices]
    state_len = tmp_conv_state.shape[-1]

    hidden_states_new = torch.cat([tmp_conv_state, hidden_states], dim=-1).to(weight.dtype)

    cast_conv_state = conv_state.unsqueeze(0)
    import kunlun_ops
    tmp_hidden_states = hidden_states_new[:, :, -state_len:].contiguous()
    # ori_shape = tmp_hidden_states.shape
    # tmp_hidden_states = tmp_hidden_states.transpose(1, 2).reshape(ori_shape)
    kunlun_ops.reshape_and_cache_flash(
                        tmp_hidden_states,
                        tmp_hidden_states,
                        cast_conv_state,
                        cast_conv_state,
                        conv_state_indices)

    # conv_state[conv_state_indices] = hidden_states_new[:, :, -state_len:]
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    out = out.to(hidden_states.dtype).squeeze(-1)
    return out


@triton.jit()
def _causal_conv1d_update_kernel_xpu(
    # Pointers to matrices
    x_ptr,  # (batch, dim, seqlen)
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,
    cache_seqlens_ptr,  # circular buffer
    conv_state_indices_ptr,
    num_accepted_tokens_ptr,
    o_ptr,  # (batch, dim, seqlen)
    # Matrix dimensions
    batch_id,
    batch: int,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # ruff: noqa: E501
    idx_seq = batch_id
    # idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        return

    # [BLOCK_N,] elements along the feature-dimension (channel)
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if IS_CONTINUOUS_BATCHING:
        # mask = idx_seq < batch
        conv_state_batch_coord = tl.load(conv_state_indices_ptr +
                                         idx_seq * stride_state_indices).to(
                                             tl.int64)
    else:
        conv_state_batch_coord = idx_seq
    if USE_PAD_SLOT:  # noqa
        if conv_state_batch_coord == pad_slot_id:
            # not processing as this is not the actual sequence
            return

    if IS_SPEC_DECODING:
        # The rolling of conv state:
        #
        # Before forward, the conv_state is:
        # [history1, history2, ..., historyM].
        #
        # After forward, the conv_state becomes:
        # [history2, ..., historyM, draft1, draft2, ..., draftN].
        #
        # After acceptance, it becomes:
        #
        # - accept 1 tokens: [history2, ..., historyM, draft1]
        # - accept 2 tokens: [history3, ..., historyM, draft1, draft2]
        # - and so on.
        conv_state_token_offset = (tl.load(num_accepted_tokens_ptr + idx_seq) -
                                   1)
    else:
        conv_state_token_offset = 0

    # STEP 1: READ init_state data
    conv_states_base = (conv_state_ptr +
                        (conv_state_batch_coord * stride_conv_state_seq) +
                        (idx_feats * stride_conv_state_dim))
    mask_w = idx_feats < dim

    prior_tokens = conv_states_base + conv_state_token_offset * stride_conv_state_tok
    if KERNEL_WIDTH >= 2:
        conv_states_ptrs = prior_tokens  # [BLOCK_N]
        col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 3:
        conv_states_ptrs = prior_tokens + 1 * stride_conv_state_tok  # [BLOCK_N]
        col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 4:
        conv_states_ptrs = prior_tokens + 2 * stride_conv_state_tok  # [BLOCK_N]
        col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH == 5:
        conv_states_ptrs = prior_tokens + 3 * stride_conv_state_tok  # [BLOCK_N]
        col3 = tl.load(conv_states_ptrs, mask_w, 0.0)

    # STEP 2: assume state_len > seqlen
    idx_tokens = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

    # With speculative decoding, the conv_state updates works in a sliding
    # window manner, at each forward pass, the tokens are shift by 1, so we
    # load since idx_tokens + 1.
    conv_state_ptrs_source = (
        conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) +
        conv_state_token_offset * stride_conv_state_tok +
        (idx_feats * stride_conv_state_dim)[None, :] +
        ((idx_tokens + (1 if IS_SPEC_DECODING else seqlen)) *
         stride_conv_state_tok)[:, None])  # [BLOCK_M, BLOCK_N]
    mask = ((conv_state_batch_coord < num_cache_lines)
            & ((idx_tokens + seqlen) < state_len)[:, None]
            & (idx_feats < dim)[None, :])
    conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)

    VAL = state_len - seqlen
    x_base = x_ptr + (idx_seq * stride_x_seq) + (idx_feats * stride_x_dim
                                                 )  # [BLOCK_N]

    x_ptrs = x_base[None, :] + (
        (idx_tokens - VAL) * stride_x_token)[:, None]  # [BLOCK_M, BLOCK_N]

    mask_x = ((idx_tokens - VAL >= 0)[:, None] &
              (idx_tokens - VAL < seqlen)[:, None] & (idx_feats < dim)[None, :]
              )  # token-index  # token-index  # feature-index
    loaded_x = tl.load(x_ptrs, mask_x, 0.0)

    new_conv_state = tl.where(mask, conv_state, loaded_x)

    conv_state_base = (conv_state_ptr +
                       (conv_state_batch_coord * stride_conv_state_seq) +
                       (idx_feats * stride_conv_state_dim))  # [BLOCK_N,]
    conv_state_ptrs_target = conv_state_base + (
        idx_tokens * stride_conv_state_tok)[:, None]  # [BLOCK_M, BLOCK_N]
    mask = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
    tl.store(conv_state_ptrs_target, new_conv_state, mask)

    # STEP 3: init accumulator
    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias,
                              other=0.0).to(tl.float32)  # [BLOCK_N]
    else:
        acc_preload = tl.zeros((BLOCK_N, ), dtype=tl.float32)

    # STEP 4:
    # PRE-LOAD WEIGHTS
    # first kernel column, configured for weights to handle BLOCK_N features in range
    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)

    x_base_1d = x_base  # starting of chunk [BLOCK_N]
    mask_x_1d = idx_feats < dim

    # STEP 5: compute each token
    for idx_token in tl.static_range(seqlen):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:  # KERNEL_WIDTH-1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

            acc += matrix_x * matrix_w  # [BLOCK_N]

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < seqlen) & (idx_feats < dim
                                          )  # token-index  # feature-index
        o_ptrs = o_ptr + (
            idx_seq) * stride_o_seq + idx_token * stride_o_token + (
                idx_feats * stride_o_dim)

        tl.store(o_ptrs, acc, mask=mask_1d)

@triton.jit()
def _causal_conv1d_update_kernel(
    # Pointers to matrices
    x_ptr,  # (batch, dim, seqlen)
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,
    cache_seqlens_ptr,  # circular buffer
    conv_state_indices_ptr,
    num_accepted_tokens_ptr,
    o_ptr,  # (batch, dim, seqlen)
    # Matrix dimensions
    batch: int,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # ruff: noqa: E501
    # idx_seq = tl.program_id(0)

    idx_seq = batch_id
    if idx_seq >= batch:
        return

    # [BLOCK_N,] elements along the feature-dimension (channel)
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if IS_CONTINUOUS_BATCHING:
        # mask = idx_seq < batch
        conv_state_batch_coord = tl.load(conv_state_indices_ptr +
                                         idx_seq * stride_state_indices).to(
                                             tl.int64)
    else:
        conv_state_batch_coord = idx_seq
    if USE_PAD_SLOT:  # noqa
        if conv_state_batch_coord == pad_slot_id:
            # not processing as this is not the actual sequence
            return

    if IS_SPEC_DECODING:
        # The rolling of conv state:
        #
        # Before forward, the conv_state is:
        # [history1, history2, ..., historyM].
        #
        # After forward, the conv_state becomes:
        # [history2, ..., historyM, draft1, draft2, ..., draftN].
        #
        # After acceptance, it becomes:
        #
        # - accept 1 tokens: [history2, ..., historyM, draft1]
        # - accept 2 tokens: [history3, ..., historyM, draft1, draft2]
        # - and so on.
        conv_state_token_offset = (tl.load(num_accepted_tokens_ptr + idx_seq) -
                                   1)
    else:
        conv_state_token_offset = 0

    # STEP 1: READ init_state data
    conv_states_base = (conv_state_ptr +
                        (conv_state_batch_coord * stride_conv_state_seq) +
                        (idx_feats * stride_conv_state_dim))
    mask_w = idx_feats < dim

    prior_tokens = conv_states_base + conv_state_token_offset * stride_conv_state_tok
    if KERNEL_WIDTH >= 2:
        conv_states_ptrs = prior_tokens  # [BLOCK_N]
        col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 3:
        conv_states_ptrs = prior_tokens + 1 * stride_conv_state_tok  # [BLOCK_N]
        col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 4:
        conv_states_ptrs = prior_tokens + 2 * stride_conv_state_tok  # [BLOCK_N]
        col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH == 5:
        conv_states_ptrs = prior_tokens + 3 * stride_conv_state_tok  # [BLOCK_N]
        col3 = tl.load(conv_states_ptrs, mask_w, 0.0)

    # STEP 2: assume state_len > seqlen
    idx_tokens = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

    # With speculative decoding, the conv_state updates works in a sliding
    # window manner, at each forward pass, the tokens are shift by 1, so we
    # load since idx_tokens + 1.
    conv_state_ptrs_source = (
        conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) +
        conv_state_token_offset * stride_conv_state_tok +
        (idx_feats * stride_conv_state_dim)[None, :] +
        ((idx_tokens + (1 if IS_SPEC_DECODING else seqlen)) *
         stride_conv_state_tok)[:, None])  # [BLOCK_M, BLOCK_N]
    mask = ((conv_state_batch_coord < num_cache_lines)
            & ((idx_tokens + seqlen) < state_len)[:, None]
            & (idx_feats < dim)[None, :])
    conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)

    VAL = state_len - seqlen
    x_base = x_ptr + (idx_seq * stride_x_seq) + (idx_feats * stride_x_dim
                                                 )  # [BLOCK_N]

    x_ptrs = x_base[None, :] + (
        (idx_tokens - VAL) * stride_x_token)[:, None]  # [BLOCK_M, BLOCK_N]

    mask_x = ((idx_tokens - VAL >= 0)[:, None] &
              (idx_tokens - VAL < seqlen)[:, None] & (idx_feats < dim)[None, :]
              )  # token-index  # token-index  # feature-index
    loaded_x = tl.load(x_ptrs, mask_x, 0.0)
    # tl.debug_barrier()

    new_conv_state = tl.where(mask, conv_state, loaded_x)

    conv_state_base = (conv_state_ptr +
                       (conv_state_batch_coord * stride_conv_state_seq) +
                       (idx_feats * stride_conv_state_dim))  # [BLOCK_N,]
    conv_state_ptrs_target = conv_state_base + (
        idx_tokens * stride_conv_state_tok)[:, None]  # [BLOCK_M, BLOCK_N]
    mask = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
    tl.store(conv_state_ptrs_target, new_conv_state, mask)

    # STEP 3: init accumulator
    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias,
                              other=0.0).to(tl.float32)  # [BLOCK_N]
    else:
        acc_preload = tl.zeros((BLOCK_N, ), dtype=tl.float32)

    # STEP 4:
    # PRE-LOAD WEIGHTS
    # first kernel column, configured for weights to handle BLOCK_N features in range
    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)

    x_base_1d = x_base  # starting of chunk [BLOCK_N]
    mask_x_1d = idx_feats < dim

    # STEP 5: compute each token
    for idx_token in tl.static_range(seqlen):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:  # KERNEL_WIDTH-1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

            acc += matrix_x * matrix_w  # [BLOCK_N]

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < seqlen) & (idx_feats < dim
                                          )  # token-index  # feature-index
        o_ptrs = o_ptr + (
            idx_seq) * stride_o_seq + idx_token * stride_o_token + (
                idx_feats * stride_o_dim)

        tl.store(o_ptrs, acc, mask=mask_1d)


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    conv_state_indices_cpu: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
    metadata=None,
    validate_data=False,
):
    """
    x: (batch, dim) or (batch, dim, seqlen)
        [shape=2: single token prediction]
        [shape=3: single or multiple tokens prediction]
    conv_state: (..., dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state
        starting at the index
        @cache_seqlens % state_len.
    conv_state_indices: (batch,), dtype int32
        If not None, the conv_state is a larger tensor along the batch dim,
        and we are selecting the batch coords specified by conv_state_indices.
        Useful for a continuous batching scenario.
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1 ,20 ,pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim) or (batch, dim, seqlen)
    """
    if validate_data:
        assert cache_seqlens is None  # not implemented yet - ok for vLLM
        assert pad_slot_id is not None
        assert x.stride(1) == 1
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]
    unsqueeze = x.dim() == 2
    if unsqueeze:
        # make it (batch, dim, seqlen) with seqlen == 1
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    _, width = weight.shape
    # conv_state: (..., dim, state_len), where state_len >= width - 1
    num_cache_lines, _, state_len = conv_state.size()

    if False and validate_data:
        assert dim == weight.size(0)
        assert conv_state.stride(
            -2
        ) == 1, f"ERROR: expect contiguous along feat-dim of conv_state (currently stride={conv_state.stride()})"
        assert state_len >= width - 1
        # when above happens, we don't shift-left to keep any records in conv_state
        assert dim == conv_state.size(1)
        if conv_state_indices is None:
            assert conv_state.size(0) >= batch
        else:
            assert (batch, ) == conv_state_indices.shape

        assert num_cache_lines >= batch
        assert weight.stride(1) == 1  # Need this
        assert cache_seqlens is None  # not needed for vLLM - circular buffer

    if True or batch > 1:
        x = x.squeeze(-1).unsqueeze(1)
        out = torch.empty_like(x)
        import kunlun_ops
        stride = conv_state.stride()[0]
        kunlun_ops.causal_conv1d_update(
                    x,
                    weight,
                    out,
                    conv_state,
                    None,
                    bias,
                    conv_state_indices_cpu=conv_state_indices_cpu,
                    conv_state_indices_xpu=conv_state_indices,
                    act="SWISH",
                    state_seq_stride=stride,
                    is_ncw=False
        )
        # out = F.silu(out)
        out = out.squeeze(1)
        return out
        return torch_causal_conv1d_update(
                x,
                conv_state,
                weight,
                bias,
                activation,
                conv_state_indices=conv_state_indices
            )

    # adopt the strategy in vLLM that overwrite on 'x' directly, rather than creating a new tensor 'o'
    out = x
    stride_w_dim, stride_w_width = weight.stride()

    stride_x_seq, stride_x_dim, stride_x_token = x.stride(
    )  # X (batch, dim, seqlen)

    stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride(
    )
    stride_state_indices = conv_state_indices.stride(
        0) if conv_state_indices is not None else 0
    if num_accepted_tokens is not None:
        state_len = width - 1 + (seqlen - 1)  # effective state_len needed
    else:
        state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    def grid(META):
        return (
            1,
            triton.cdiv(dim, META["BLOCK_N"]),
        )
    for batch_id in range(batch):
        _causal_conv1d_update_kernel_xpu[grid](
            x,
            weight,
            bias,
            conv_state,
            cache_seqlens,
            conv_state_indices,
            num_accepted_tokens,
            out,
            batch_id=batch_id,
            batch=batch,
            dim=dim,
            seqlen=seqlen,
            state_len=state_len,
            num_cache_lines=num_cache_lines,
            stride_x_seq=stride_x_seq,
            stride_x_dim=stride_x_dim,
            stride_x_token=stride_x_token,
            stride_w_dim=stride_w_dim,
            stride_w_width=stride_w_width,
            stride_conv_state_seq=stride_istate_seq,
            stride_conv_state_dim=stride_istate_dim,
            stride_conv_state_tok=stride_istate_token,
            stride_state_indices=stride_state_indices,
            stride_o_seq=stride_o_seq,
            stride_o_dim=stride_o_dim,
            stride_o_token=stride_o_token,
            pad_slot_id=pad_slot_id,
            HAS_BIAS=bias is not None,
            KERNEL_WIDTH=width,
            SILU_ACTIVATION=activation in ["silu", "swish"],
            IS_CONTINUOUS_BATCHING=conv_state_indices is not None,
            IS_SPEC_DECODING=num_accepted_tokens is not None,
            NP2_STATELEN=np2_statelen,
            USE_PAD_SLOT=pad_slot_id is not None,
            BLOCK_N=256,
            groups_per_cluster=np2_statelen,
            isCloseUnrollControl=True,
            isCloseVectorization=True,
            isCloseOffsetAnalysis=True,
            is_use_mask_zero = True
        )
    if unsqueeze:
        out = out.squeeze(-1)
    return out
