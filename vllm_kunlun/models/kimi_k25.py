# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""
Kimi-K2.5 model implementation for vLLM-Kunlun.

Kimi-K2.5 extends Kimi-K2 (DeepseekV3) with vision support using
a MoonViT 3D vision tower and temporal pooling patch merger.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Optional, Union

import torch
from torch import nn
from transformers import BatchFeature
from transformers.processing_utils import ProcessorMixin
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from vllm.model_executor.models.interfaces import (
    SupportsMultiModal,
    SupportsPP,
    SupportsQuant,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
    merge_multimodal_embeddings,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargs,
    NestedTensors,
)
from vllm.multimodal.parse import ImageProcessorItems, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.processor import cached_get_image_processor
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from vllm_kunlun.transformer_utils.kimi_k25 import KimiK25Config

from .kimi_k25_vit import (
    KimiK25MultiModalProjector,
    MoonViT3dPretrainedModel,
    vision_tower_forward,
)

logger = init_logger(__name__)


@dataclass
class MaxImageTokenMeta:
    width: int = 3000
    height: int = 3000


class KimiK25MediaPixelInputs(TensorSchema):
    type: Literal["pixel_values"] = "pixel_values"
    pixel_values: Annotated[
        Union[torch.Tensor, list],
        TensorShape("np", 3, "ps", "ps"),
    ]
    grid_thws: Annotated[torch.Tensor, TensorShape("nm", 3)]


class MoonshotKimiVAutoProcessor(ProcessorMixin):
    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(self, media_processor=None, tokenizer=None, media_token_id=None):
        super().__init__(tokenizer)
        self.media_processor = media_processor
        self.media_token_id = media_token_id
        assert self.media_token_id is not None

    def __call__(
        self,
        text=None,
        images=None,
        videos=None,
        vision_chunks=None,
        **kwargs,
    ) -> BatchFeature:
        # Build vision_chunks from vLLM standard `images`/`videos` kwargs
        # if vision_chunks is not provided directly.
        if vision_chunks is None:
            vision_chunks = []
            if images is not None:
                if not isinstance(images, (list, tuple)):
                    images = [images]
                for img in images:
                    vision_chunks.append({"type": "image", "image": img})
            if videos is not None:
                if not isinstance(videos, (list, tuple)):
                    videos = [videos]
                for video in videos:
                    if isinstance(video, (list, tuple)):
                        # Already a list of frames
                        vision_chunks.append(
                            {
                                "type": "video_chunk",
                                "video_chunk": list(video),
                            }
                        )
                    else:
                        vision_chunks.append(
                            {
                                "type": "image",
                                "image": video,
                            }
                        )

        mm_inputs = {}
        input_ids = self.tokenizer.encode(text) if isinstance(text, str) else text
        if vision_chunks:
            mm_inputs = self.media_processor.preprocess(vision_chunks)

            num_tokens_per_chunk = [
                self.media_processor.media_tokens_calculator(chunk)
                for chunk in vision_chunks
            ]

            new_input_ids = []
            for token in input_ids:
                if token == self.media_token_id:
                    new_input_ids.extend(
                        [self.media_token_id] * num_tokens_per_chunk.pop(0)
                    )
                else:
                    new_input_ids.append(token)
            input_ids = new_input_ids

        return BatchFeature(
            data={
                "input_ids": torch.tensor([input_ids]),
                **mm_inputs,
            }
        )


class KimiK25ProcessingInfo(BaseProcessingInfo):
    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self.hf_config = self.get_hf_config()
        self.media_token_id = self.hf_config.media_placeholder_token_id
        self.media_processor = cached_get_image_processor(
            self.ctx.model_config.model,
            trust_remote_code=self.ctx.model_config.trust_remote_code,
        )
        self.hf_processor = MoonshotKimiVAutoProcessor(
            media_processor=self.media_processor,
            tokenizer=self.get_tokenizer(),
            media_token_id=self.media_token_id,
        )
        self.media_tokens_calculator = self.media_processor.media_tokens_calculator

    def get_hf_processor(self):
        return self.hf_processor

    def get_hf_config(self):
        return self.ctx.get_hf_config(KimiK25Config)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}


class KimiK25DummyInputsBuilder(BaseDummyInputsBuilder):
    def __init__(self, info: KimiK25ProcessingInfo) -> None:
        super().__init__(info)
        self.media_token_id = self.info.media_token_id
        self.frame_per_chunk = getattr(
            self.info.media_processor, "num_frames_per_chunk", 4
        )

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_media = mm_counts.get("image", 0)
        return "<|media_pad|>" * num_media

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        mm_data: MultiModalDataDict = {}
        if num_images > 0:
            mm_data["image"] = self._get_dummy_images(
                width=MaxImageTokenMeta.width,
                height=MaxImageTokenMeta.height,
                num_images=num_images,
            )
        return mm_data


class KimiK25MultiModalProcessor(BaseMultiModalProcessor):
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        grid_thws = hf_inputs.get("grid_thws", torch.empty((0, 3)))
        grid_sizes = grid_thws.prod(-1)

        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes("image", grid_sizes),
            grid_thws=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        media_token_id = self.info.get_hf_config().media_placeholder_token_id

        def get_replacement(item_idx: int):
            media = mm_items.get_items("image", ImageProcessorItems)
            media_item = {"type": "image", "image": media[item_idx]}
            try:
                num_media_token = self.info.media_tokens_calculator(media_item)
            except Exception:
                logger.warning(
                    "media_tokens_calculator failed for image item %d, "
                    "using fallback estimation",
                    item_idx,
                )
                num_media_token = 1
            return [media_token_id] * num_media_token

        updates = []
        if mm_items.get_count("image", strict=False) > 0:
            updates.append(
                PromptReplacement(
                    modality="image",
                    target=[media_token_id],
                    replacement=get_replacement,
                )
            )
        return updates

    def split_video_chunks(self, video):
        return self.info.media_processor.split_video_chunks(video)


@MULTIMODAL_REGISTRY.register_processor(
    KimiK25MultiModalProcessor,
    info=KimiK25ProcessingInfo,
    dummy_inputs=KimiK25DummyInputsBuilder,
)
class KimiK25ForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsQuant,
):
    supports_encoder_tp_data = True

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language_model.layers.": "language_model.model.layers.",
            "mm_projector.proj.0": "mm_projector.linear_1",
            "mm_projector.proj.2": "mm_projector.linear_2",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality == "image":
            return "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>"
        if modality == "video":
            return "<|kimi_k25_video_placeholder|>"
        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        model_config = vllm_config.model_config
        config: KimiK25Config = model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config

        self.use_data_parallel = (
            model_config.multimodal_config.mm_encoder_tp_mode == "data"
        )
        self.hidden_size = config.text_config.hidden_size
        self.device = current_platform.current_device()

        # Vision tower
        self.vision_tower = MoonViT3dPretrainedModel(
            config.vision_config,
            quant_config=self._maybe_ignore_quant_config(quant_config),
            prefix=maybe_prefix(prefix, "vision_tower"),
            use_data_parallel=self.use_data_parallel,
        ).to(device=self.device, dtype=model_config.dtype)

        # MM projector
        self.mm_projector = KimiK25MultiModalProjector(
            config=config.vision_config,
            use_data_parallel=self.use_data_parallel,
            quant_config=self._maybe_ignore_quant_config(quant_config),
            prefix=maybe_prefix(prefix, "mm_projector"),
        ).to(device=self.device, dtype=model_config.dtype)

        # Language model (DeepseekV3)
        self.quant_config = quant_config
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            architectures=["DeepseekV3ForCausalLM"],
            prefix=maybe_prefix(prefix, "language_model"),
        )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        self.media_placeholder: int = self.config.media_placeholder_token_id

    def _maybe_ignore_quant_config(self, quant_config):
        if isinstance(quant_config, CompressedTensorsConfig):
            return None
        return quant_config

    def _parse_and_validate_media_input(
        self, **kwargs: object
    ) -> Optional[KimiK25MediaPixelInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        grid_thws = kwargs.pop("grid_thws", None)
        if pixel_values is None:
            return None

        if isinstance(pixel_values, list):
            pixel_values = torch.cat(pixel_values, dim=0)

        if len(pixel_values.shape) in (3, 5):
            pixel_values = pixel_values.reshape(
                pixel_values.shape[0] * pixel_values.shape[1], *pixel_values.shape[2:]
            )

        target_dtype = next(self.vision_tower.parameters()).dtype
        pixel_values = pixel_values.to(target_dtype)

        assert isinstance(
            grid_thws, torch.Tensor
        ), f"expect grid_thws to be a tensor, get {type(grid_thws)}"
        grid_thws = grid_thws.reshape(-1, grid_thws.shape[-1])
        assert (
            grid_thws.ndim == 2 and grid_thws.size(1) == 3
        ), f"unexpected shape for grid_thws: {grid_thws.shape}"

        return KimiK25MediaPixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            grid_thws=grid_thws,
        )

    def _process_media_input(self, media_input: KimiK25MediaPixelInputs) -> list:
        return vision_tower_forward(
            self.vision_tower,
            media_input["pixel_values"],
            media_input["grid_thws"],
            mm_projector=self.mm_projector,
            use_data_parallel=self.use_data_parallel,
        )

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(self, **kwargs: object) -> NestedTensors:
        media_input = self._parse_and_validate_media_input(**kwargs)
        if media_input is None:
            return []
        return self._process_media_input(media_input)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None and len(multimodal_embeddings) != 0:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                placeholder_token_id=self.media_placeholder,
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if intermediate_tensors is not None:
            input_ids = None
            inputs_embeds = None
        elif inputs_embeds is None:
            multimodal_embeddings = self.get_multimodal_embeddings(**kwargs)
            if multimodal_embeddings:
                inputs_embeds = self.get_input_embeddings(
                    input_ids, multimodal_embeddings
                )
                input_ids = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple]) -> set:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
