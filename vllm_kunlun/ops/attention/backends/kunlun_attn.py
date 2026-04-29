"""kunlun attention wrapper for context and decode"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

import torch

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUBuilder

from itertools import accumulate

from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionType,
)
from vllm.attention.backends.utils import (
    compute_slot_mapping,
    compute_slot_mapping_start_idx,
    is_block_tables_empty,
)
from vllm.logger import init_logger

from vllm_kunlun.ops._kunlun_ops import KunlunOps
from vllm_kunlun.ops.paged_attn import PagedAttention, PagedAttentionMetadata

from .utils import CommonAttentionState, CommonMetadataBuilder

logger = init_logger(__name__)


class KunlunAttentionBackend(AttentionBackend):
    """KunlunAttentionBackend"""

    accept_output_buffer = False

    @staticmethod
    def get_name() -> str:
        return "KUNLUN_ATTENTION"

    @staticmethod
    def get_impl_cls() -> Type["KunlunAttentionImpl"]:
        """get_impl_cls"""
        return KunlunAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["KunlunMetadata"]:
        """get_metadata_cls"""
        return KunlunMetadata

    @staticmethod
    def get_builder_cls() -> Type["KunlunMetadataBuilder"]:
        """get_builder_cls"""
        return KunlunMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return PagedAttention.get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size
        )

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: Dict[int, int],
    ) -> None:
        PagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        PagedAttention.copy_blocks(kv_caches, src_to_dists)


@dataclass
class KunlunMetadata(AttentionMetadata, PagedAttentionMetadata):
    """KunlunMetadata"""

    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ----------------------|
    #                                   |-- query_len ---|

    # seq_lens stored as a tensor.
    seq_lens_tensor: Optional[torch.Tensor]

    # FIXME: It is for flash attn.
    # Maximum sequence length among prefill batch. 0 if there are decoding
    # requests only.
    max_prefill_seq_len: int
    # Maximum sequence length among decode batch. 0 if there are prefill
    # requests only.
    max_decode_seq_len: int

    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.
    use_cuda_graph: bool

    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[List[int]] = None

    # FIXME: It is for flash attn.
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor] = None

    # (batch_size,) A tensor of context lengths (tokens that are computed
    # so far).
    context_lens_tensor: Optional[torch.Tensor] = None

    # Maximum query length in the batch. None for decoding.
    max_query_len: Optional[int] = None

    # Max number of key/value length in the batch, especially for prefix cache
    max_kv_len: Optional[int] = None

    # Max number of query tokens among request in the batch.
    max_decode_query_len: Optional[int] = None

    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    query_start_loc: Optional[torch.Tensor] = None
    query_start_loc_host: Optional[torch.Tensor] = None
    # serve only for prefix cache
    kv_prefix_start_loc_host: Optional[torch.Tensor] = None
    kv_prefix_start_loc: Optional[torch.Tensor] = None

    # Self-attention prefill/decode metadata cache
    _cached_prefill_metadata: Optional["KunlunMetadata"] = None
    _cached_decode_metadata: Optional["KunlunMetadata"] = None

    # Begin encoder attn & enc/dec cross-attn fields...

    # Encoder sequence lengths representation
    encoder_seq_lens: Optional[List[int]] = None
    encoder_seq_lens_tensor: Optional[torch.Tensor] = None

    # Maximum sequence length among encoder sequences
    max_encoder_seq_len: Optional[int] = None

    # Number of tokens input to encoder
    num_encoder_tokens: Optional[int] = None

    # Cross-attention memory-mapping data structures: slot mapping
    # and block tables
    cross_slot_mapping: Optional[torch.Tensor] = None
    cross_block_tables: Optional[torch.Tensor] = None

    seq_lens_tensor_cpu: Optional[torch.Tensor] = None

    def __post_init__(self):
        # Set during the execution of the first attention op.
        # It is a list because it is needed to set per prompt
        # when alibi slopes is used. It is because of the limitation
        # from xformer API.
        # will not appear in the __repr__ and __init__
        self.attn_bias: Optional[List["AttentionBias"]] = None  # noqa: F821
        self.encoder_attn_bias: Optional[List["AttentionBias"]] = None  # noqa: F821
        self.cross_attn_bias: Optional[List["AttentionBias"]] = None  # noqa: F821

    @property
    def is_all_encoder_attn_metadata_set(self):
        """
        All attention metadata required for encoder attention is set.
        """
        return (
            (self.encoder_seq_lens is not None)
            and (self.encoder_seq_lens_tensor is not None)
            and (self.max_encoder_seq_len is not None)
        )

    @property
    def is_all_cross_attn_metadata_set(self):
        """
        All attention metadata required for enc/dec cross-attention is set.

        Superset of encoder attention required metadata.
        """
        return (
            self.is_all_encoder_attn_metadata_set
            and (self.cross_slot_mapping is not None)
            and (self.cross_block_tables is not None)
        )

    @property
    def prefill_metadata(self) -> Optional["KunlunMetadata"]:
        """prefill_metadata"""
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            # Recover cached prefill-phase attention
            # metadata structure
            return self._cached_prefill_metadata

        assert (self.seq_lens is not None) or (self.encoder_seq_lens is not None)
        assert (self.seq_lens_tensor is not None) or (
            self.encoder_seq_lens_tensor is not None
        )

        # Compute some attn_metadata fields which default to None
        query_start_loc = (
            None
            if self.query_start_loc is None
            else self.query_start_loc[: self.num_prefills + 1]
        )
        # flash attention needs both lod information on host and device
        query_start_loc_host = (
            None
            if self.query_start_loc_host is None
            else self.query_start_loc_host[: self.num_prefills + 1]
        )
        kv_prefix_start_loc_host = (
            None
            if self.kv_prefix_start_loc_host is None
            else self.kv_prefix_start_loc_host[: self.num_prefills + 1]
            + query_start_loc_host
        )
        kv_prefix_start_loc = (
            None
            if kv_prefix_start_loc_host is None
            else kv_prefix_start_loc_host.cuda()
        )
        slot_mapping = (
            None
            if self.slot_mapping is None
            else self.slot_mapping[: self.num_prefill_tokens]
        )
        seq_lens = None if self.seq_lens is None else self.seq_lens[: self.num_prefills]
        seq_lens_tensor = (
            None
            if self.seq_lens_tensor is None
            else self.seq_lens_tensor[: self.num_prefills]
        )
        context_lens_tensor = (
            None
            if self.context_lens_tensor is None
            else self.context_lens_tensor[: self.num_prefills]
        )
        # for prefix cache, block table only contains blocks that hit
        # if self.block_tables is None:
        #     block_tables = None
        # elif self.block_tables.shape[1] == 0:
        #     block_tables = self.block_tables[:self.num_prefills]
        # else:
        #     block_tables = self.block_tables[:self.num_prefills][:, -1].clone()

        block_tables = (
            None
            if self.block_tables is None
            else self.block_tables[: self.num_prefills]
        )

        # Construct & cache prefill-phase attention metadata structure
        self._cached_prefill_metadata = KunlunMetadata(
            multi_modal_placeholder_index_maps=self.multi_modal_placeholder_index_maps,
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=self.max_query_len,
            max_kv_len=self.max_kv_len,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_seq_len=0,
            query_start_loc=query_start_loc,
            query_start_loc_host=query_start_loc_host,
            kv_prefix_start_loc=kv_prefix_start_loc,
            kv_prefix_start_loc_host=kv_prefix_start_loc_host,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=False,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            max_encoder_seq_len=self.max_encoder_seq_len,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables,
            enable_kv_scales_calculation=False,
            seq_start_loc=self.seq_start_loc,
        )
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["KunlunMetadata"]:
        """decode_metadata"""
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            # Recover cached decode-phase attention
            # metadata structure
            return self._cached_decode_metadata
        assert (self.seq_lens_tensor is not None) or (
            self.encoder_seq_lens_tensor is not None
        )

        # Compute some attn_metadata fields which default to None
        slot_mapping = (
            None
            if self.slot_mapping is None
            else self.slot_mapping[self.num_prefill_tokens :]
        )
        seq_lens_tensor = (
            None
            if self.seq_lens_tensor is None
            else self.seq_lens_tensor[self.num_prefills :]
        )
        seq_lens_tensor_cpu = (
            None
            if self.seq_lens_tensor_cpu is None
            else self.seq_lens_tensor_cpu[self.num_prefills :]
        )
        block_tables = (
            None
            if self.block_tables is None
            else self.block_tables[self.num_prefills :]
        )

        # Construct & cache decode-phase attention metadata structure
        self._cached_decode_metadata = KunlunMetadata(
            multi_modal_placeholder_index_maps=self.multi_modal_placeholder_index_maps,
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=slot_mapping,
            seq_lens_tensor=seq_lens_tensor,
            seq_lens_tensor_cpu=seq_lens_tensor_cpu,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
            block_tables=block_tables,
            use_cuda_graph=self.use_cuda_graph,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            max_encoder_seq_len=self.max_encoder_seq_len,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables,
            enable_kv_scales_calculation=False,
        )
        return self._cached_decode_metadata


class KunlunMetadataBuilder(CommonMetadataBuilder[KunlunMetadata]):
    """KunlunMetadataBuilder"""

    _metadata_cls = KunlunMetadata

    def __init__(self, input_builder: "ModelInputForGPUBuilder"):
        super().__init__(input_builder)
        self.prefix_cache_kv_lens: List[int] = []

    def prepare(self):
        """prepare"""
        super().prepare()
        self.prefix_cache_kv_lens = list()

    def _add_seq_group(
        self,
        inter_data: "ModelInputForGPUBuilder.InterDataForSeqGroup",
        chunked_prefill_enabled: bool,
    ):
        is_prompt = inter_data.is_prompt
        block_tables = inter_data.block_tables

        for (
            seq_id,
            token_len,
            seq_len,
            curr_seq_len,
            query_len,
            context_len,
            curr_sliding_window_block,
        ) in zip(
            inter_data.seq_ids,
            [len(t) for t in inter_data.input_tokens],
            inter_data.orig_seq_lens,
            inter_data.seq_lens,
            inter_data.query_lens,
            inter_data.context_lens,
            inter_data.curr_sliding_window_blocks,
        ):
            self.context_lens.append(context_len)
            if is_prompt:
                mm_maps = inter_data.multi_modal_placeholder_maps
                if mm_maps:
                    for modality, placeholders in mm_maps.items():
                        self.multimodal_placeholder_maps[modality].extend(placeholders)

                self.num_prefills += 1
                self.num_prefill_tokens += token_len
                self.prefill_seq_lens.append(seq_len)
            else:
                assert (
                    query_len == 1
                ), "seq_len: {}, context_len: {}, query_len: {}".format(
                    seq_len, context_len, query_len
                )
                self.num_decode_tokens += query_len
                self.curr_seq_lens.append(curr_seq_len)

            # Compute block table.
            block_table = []
            assert (
                not chunked_prefill_enabled
            ), "chunk prefill not supported for kunlun attention"
            if inter_data.prefix_cache_hit:
                assert context_len != 0
                assert context_len % self.block_size == 0
                # block_table = block_tables[seq_id]
                block_table = block_tables[seq_id][: context_len // self.block_size]
            elif (not is_prompt) and block_tables is not None:
                if curr_sliding_window_block == 0:
                    block_table = block_tables[seq_id]
                else:
                    block_table = block_tables[seq_id][-curr_sliding_window_block:]
            self.block_tables.append(block_table)
            if is_prompt:
                self.prefix_cache_kv_lens.append(context_len)

            # Compute slot mapping.
            is_profile_run = is_block_tables_empty(block_tables)
            start_idx = compute_slot_mapping_start_idx(
                is_prompt, query_len, context_len, self.sliding_window
            )
            compute_slot_mapping(
                is_profile_run,
                self.slot_mapping,
                seq_id,
                seq_len,
                context_len,
                start_idx,
                self.block_size,
                inter_data.block_tables,
            )

    def build(
        self,
        seq_lens: List[int],
        query_lens: List[int],
        cuda_graph_pad_size: int,
        batch_size: int,
    ):
        """build"""
        attn_meta = super().build(seq_lens, query_lens, cuda_graph_pad_size, batch_size)
        query_start_loc = list(accumulate(query_lens, initial=0))
        query_start_loc_host = torch.tensor(
            query_start_loc, dtype=torch.int32, device="cpu"
        )
        attn_meta.query_start_loc_host = query_start_loc_host
        # max_kv_len = max(query_lens + prefix_cache_kv_lens)
        attn_meta.max_kv_len = max(self.prefix_cache_kv_lens + attn_meta.seq_lens)
        # 包含kv cache ，且存在命中的情况
        if len(self.prefix_cache_kv_lens) != 0 and max(self.prefix_cache_kv_lens) != 0:
            self.prefix_cache_kv_lens = list(
                accumulate(self.prefix_cache_kv_lens, initial=0)
            )
            prefix_cache_kv_lens_tensor = torch.tensor(
                self.prefix_cache_kv_lens, dtype=torch.int32, device="cpu"
            )
            attn_meta.kv_prefix_start_loc_host = prefix_cache_kv_lens_tensor
        attn_meta.seq_lens_tensor_cpu = attn_meta.seq_lens_tensor.to("cpu")
        return attn_meta


def _get_seq_len_block_table_args(
    attn_metadata: KunlunMetadata,
    is_prompt: bool,
    attn_type: AttentionType,
) -> tuple:
    """
    The particular choice of sequence-length- and block-table-related
    attributes which should be extracted from attn_metadata is dependent
    on the type of attention operation.

    Decoder attn -> select entirely decoder self-attention-related fields
    Encoder/decoder cross-attn -> select encoder sequence lengths &
                                  cross-attn block-tables fields
    Encoder attn -> select encoder sequence lengths fields & no block tables

    Arguments:

    * attn_metadata: Attention metadata structure associated with attention op
    * is_prompt: True if prefill, False otherwise
    * attn_type: encoder attention, decoder self-attention,
                 encoder/decoder cross-attention

    Returns:

    * Appropriate sequence-lengths tensor
    * Appropriate max sequence-length scalar
    * Appropriate block tables (or None)
    """

    if attn_type == AttentionType.DECODER:
        # Decoder self-attention
        # Choose max_seq_len based on whether we are in prompt_run
        if is_prompt:
            max_seq_len = attn_metadata.max_prefill_seq_len
        else:
            max_seq_len = attn_metadata.max_decode_seq_len
        return (attn_metadata.seq_lens_tensor, max_seq_len, attn_metadata.block_tables)
    elif attn_type == AttentionType.ENCODER_DECODER:
        # Enc/dec cross-attention KVs match encoder sequence length;
        # cross-attention utilizes special "cross" block tables
        return (
            attn_metadata.encoder_seq_lens_tensor,
            attn_metadata.max_encoder_seq_len,
            attn_metadata.cross_block_tables,
        )
    elif attn_type == AttentionType.ENCODER:
        # No block tables associated with encoder attention
        return (
            attn_metadata.encoder_seq_lens_tensor,
            attn_metadata.max_encoder_seq_len,
            None,
        )
    else:
        raise AttributeError(f"Invalid attention type {str(attn_type)}")


class KunlunAttentionImpl(AttentionImpl[KunlunMetadata]):
    """KunlunAttentionImpl"""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError("kunlunAttention does not support block-sparse attention.")
        # if logits_soft_cap is not None:
        #     raise ValueError(
        #         "kunlunAttention does not support attention logits soft capping.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        suppored_head_sizes = PagedAttention.get_supported_head_sizes()
        if head_size not in suppored_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {suppored_head_sizes}."
            )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        kv_cache: torch.Tensor,
        attn_metadata: KunlunMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> torch.Tensor:
        """Forward pass with KunlunAttn and PagedAttention.

        For decoder-only models: query, key and value must be non-None.

        For encoder/decoder models:
        * KunlunAttnImpl.forward() may be invoked for both self- and cross-
          attention layers.
        * For self-attention: query, key and value must be non-None.
        * For cross-attention:
            * Query must be non-None
            * During prefill, key and value must be non-None; key and value
              get cached for use during decode.
            * During decode, key and value may be None, since:
              (1) key and value tensors were cached during prefill, and
              (2) cross-attention key and value tensors do not grow during
                  decode

        A note on how the attn_type (attention type enum) argument impacts
        attention forward() behavior:

            * DECODER: normal decoder-only behavior;
                use decoder self-attention block table
            * ENCODER: no KV caching; pass encoder sequence
                attributes (encoder_seq_lens/encoder_seq_lens_tensor/
                max_encoder_seq_len) to kernel, in lieu of decoder
                sequence attributes (seq_lens/seq_lens_tensor/max_seq_len).
                Used for encoder branch of encoder-decoder models.
            * ENCODER_ONLY: no kv_caching, uses the normal attention
                attributes (seq_lens/seq_lens_tensor/max_seq_len).
            * ENCODER_DECODER: cross-attention behavior;
                use cross-attention block table for caching KVs derived
                from encoder hidden states; since KV sequence lengths
                will match encoder sequence lengths, pass encoder sequence
                attributes to kernel (encoder_seq_lens/encoder_seq_lens_tensor/
                max_encoder_seq_len)

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
                NOTE: kv_cache will be an empty tensor with shape [0]
                for profiling run.
            attn_metadata: Metadata for attention.
            attn_type: Select attention type, between encoder attention,
                       decoder self-attention, or encoder/decoder cross-
                       attention. Defaults to decoder self-attention,
                       which is the vLLM default generally
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """

        # Check that appropriate attention metadata attributes are
        # selected for the desired attention type
        if attn_type == AttentionType.ENCODER and (
            not attn_metadata.is_all_encoder_attn_metadata_set
        ):
            raise AttributeError(
                "Encoder attention requires setting " "encoder metadata attributes."
            )

        elif attn_type == AttentionType.ENCODER_DECODER and (
            not attn_metadata.is_all_cross_attn_metadata_set
        ):
            raise AttributeError(
                "Encoder/decoder cross-attention "
                "requires setting cross-attention "
                "metadata attributes."
            )

        query = query.view(-1, self.num_heads, self.head_size)
        if key is not None:
            assert value is not None
            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.view(-1, self.num_kv_heads, self.head_size)
        else:
            assert value is None

        # Self-attention vs. cross-attention will impact
        # which KV cache memory-mapping & which
        # seqlen datastructures we utilize

        if attn_type != AttentionType.ENCODER and kv_cache.numel() > 0:
            # KV-cache during decoder-self- or
            # encoder-decoder-cross-attention, but not
            # during encoder attention.
            #
            # Even if there are no new key/value pairs to cache,
            # we still need to break out key_cache and value_cache
            # i.e. for later use by paged attention
            key_cache, value_cache = PagedAttention.split_kv_cache(kv_cache=kv_cache)

            if (key is not None) and (value is not None):

                if attn_type == AttentionType.ENCODER_DECODER:
                    updated_slot_mapping = attn_metadata.cross_slot_mapping
                else:
                    updated_slot_mapping = attn_metadata.slot_mapping
                value = value.contiguous()
                KunlunOps.reshape_and_cache(
                    key,
                    value,
                    key_cache,
                    value_cache,
                    updated_slot_mapping,
                    self.kv_cache_dtype,
                )

        if attn_type == AttentionType.ENCODER:
            # Encoder attention - chunked prefill is not applicable;
            # derive token-count from query shape & and treat them
            # as 100% prefill tokens
            assert attn_metadata.num_encoder_tokens is not None
            num_prefill_tokens = attn_metadata.num_encoder_tokens
            num_encoder_tokens = attn_metadata.num_encoder_tokens
            num_decode_tokens = 0
        elif attn_type == AttentionType.DECODER:
            # Decoder self-attention supports chunked prefill.
            num_prefill_tokens = attn_metadata.num_prefill_tokens
            num_encoder_tokens = attn_metadata.num_prefill_tokens
            num_decode_tokens = attn_metadata.num_decode_tokens
            # Only enforce this shape-constraint for decoder
            # self-attention
            assert key.shape[0] == num_prefill_tokens + num_decode_tokens
            assert value.shape[0] == num_prefill_tokens + num_decode_tokens
        else:  # attn_type == AttentionType.ENCODER_DECODER
            # Encoder/decoder cross-attention requires no chunked
            # prefill (100% prefill or 100% decode tokens, no mix)
            num_prefill_tokens = attn_metadata.num_prefill_tokens
            if attn_metadata.num_encoder_tokens is not None:
                num_encoder_tokens = attn_metadata.num_encoder_tokens
            else:
                num_encoder_tokens = attn_metadata.num_prefill_tokens
            num_decode_tokens = attn_metadata.num_decode_tokens

        output = torch.empty_like(query)
        # Query for decode. KV is not needed because it is already cached.
        decode_query = query[num_prefill_tokens:]
        # QKV for prefill.
        query = query[:num_prefill_tokens]
        if key is not None and value is not None:
            key = key[:num_encoder_tokens]
            value = value[:num_encoder_tokens]

        assert query.shape[0] == num_prefill_tokens
        assert decode_query.shape[0] == num_decode_tokens

        if prefill_meta := attn_metadata.prefill_metadata:
            # Prompt run.
            if kv_cache.numel() == 0 or prefill_meta.block_tables.numel() == 0:
                out = KunlunOps.multi_query_kv_attention(
                    prefill_meta.query_start_loc,
                    prefill_meta.query_start_loc_host,
                    query,
                    key,
                    value,
                    alibi_slopes=self.alibi_slopes,
                ).view_as(query)
                assert output[:num_prefill_tokens].shape == out.shape
                output[:num_prefill_tokens] = out

        if decode_meta := attn_metadata.decode_metadata:
            assert (
                attn_type != AttentionType.ENCODER_ONLY
            ), "Encoder-only models should not have decode metadata."
            (
                seq_lens_arg,
                max_seq_len_arg,
                block_tables_arg,
            ) = _get_seq_len_block_table_args(decode_meta, False, attn_type)

            output[num_prefill_tokens:] = PagedAttention.forward_decode(
                decode_query,
                key_cache,
                value_cache,
                block_tables_arg,
                seq_lens_arg,
                decode_meta.seq_lens_tensor_cpu,
                False,
                max_seq_len_arg,
                self.kv_cache_dtype,
                self.num_kv_heads,
                self.scale,
                self.alibi_slopes,
                k_scale,
                v_scale,
            )

        # Reshape the output tensor.
        return output.view(-1, self.num_heads * self.head_size)
