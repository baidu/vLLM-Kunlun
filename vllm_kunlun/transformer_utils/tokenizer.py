#
# Copyright (c) 2026 Baidu, Inc. All Rights Reserved.

# This file is a part of the vllm-kunlun project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import tempfile
import shutil
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import huggingface_hub
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from vllm import envs
from vllm.logger import init_logger
from vllm.transformers_utils import tokenizer
from vllm.transformers_utils.config import get_sentence_transformer_tokenizer_config
from vllm.transformers_utils.tokenizer import get_cached_tokenizer
from vllm.transformers_utils.tokenizers import MistralTokenizer
from vllm.transformers_utils.utils import check_gguf_file

if TYPE_CHECKING:
    from vllm.transformers_utils.tokenizer_base import TokenizerBase
else:
    TokenizerBase = Any

logger = init_logger(__name__)

AnyTokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast, TokenizerBase]


def kunlun_get_tokenizer(
    tokenizer_name: Union[str, Path],
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    revision: Optional[str] = None,
    download_dir: Optional[str] = None,
    **kwargs,
) -> AnyTokenizer:
    """Gets a tokenizer for the given model name via HuggingFace or ModelScope."""
    if envs.VLLM_USE_MODELSCOPE:
        # download model from ModelScope hub,
        # lazy import so that modelscope is not required for normal use.
        # pylint: disable=C.
        from modelscope.hub.snapshot_download import snapshot_download

        # avoid circuit import
        from vllm.model_executor.model_loader.weight_utils import get_lock

        # Only set the tokenizer here, model will be downloaded on the workers.
        if not os.path.exists(tokenizer_name):
            # Use file lock to prevent multiple processes from
            # downloading the same file at the same time.
            with get_lock(tokenizer_name, download_dir):
                tokenizer_path = snapshot_download(
                    model_id=tokenizer_name,
                    cache_dir=download_dir,
                    revision=revision,
                    local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                    # Ignore weights - we only need the tokenizer.
                    ignore_file_pattern=[".*.pt", ".*.safetensors", ".*.bin"],
                )
                tokenizer_name = tokenizer_path

    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False

    if "truncation_side" not in kwargs:
        kwargs["truncation_side"] = "left"

    # Separate model folder from file path for GGUF models
    is_gguf = check_gguf_file(tokenizer_name)
    if is_gguf:
        kwargs["gguf_file"] = Path(tokenizer_name).name
        tokenizer_name = Path(tokenizer_name).parent

    # if tokenizer is from official mistral org
    is_from_mistral_org = str(tokenizer_name).split("/")[0] == "mistralai"
    if is_from_mistral_org and tokenizer_mode != "mistral":
        warnings.warn(
            "It is strongly recommended to run mistral models with "
            '`--tokenizer-mode "mistral"` to ensure correct '
            "encoding and decoding.",
            FutureWarning,
            stacklevel=2,
        )

    tokenizer: AnyTokenizer
    if tokenizer_mode == "mistral":
        tokenizer = MistralTokenizer.from_pretrained(
            str(tokenizer_name), revision=revision
        )
    elif tokenizer_mode == "custom":
        from vllm.transformers_utils.tokenizer_base import TokenizerRegistry

        tokenizer = TokenizerRegistry.get_tokenizer(
            str(tokenizer_name),
            *args,
            revision=revision,
            download_dir=download_dir,
            **kwargs,
        )
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=tokenizer_name,
                *args,
                trust_remote_code=trust_remote_code,
                revision=revision,
                **kwargs,
            )
        except ValueError as e:
            # If the error pertains to the tokenizer class not existing or not
            # currently being imported,
            # suggest using the --trust-remote-code flag.

            if not trust_remote_code and (
                "does not exist or is not currently imported." in str(e)
                or "requires you to execute the tokenizer file" in str(e)
            ):
                err_msg = (
                    "Failed to load the tokenizer. If the tokenizer "
                    "is a custom tokenizer not yet available in the "
                    "HuggingFace transformers library, consider "
                    "setting `trust_remote_code=True` in LLM or using "
                    "the `--trust-remote-code` flag in the CLI."
                )
                raise RuntimeError(err_msg) from e

            # FIXME: Temporary compatibility code for new config format. Remove after vLLM upgrade.
            if "TokenizersBackend" in str(e):
                logger.warning(
                    "TokenizerBackend not supported, patching tokenizer_config.json "
                    "and loading with PreTrainedTokenizerFast."
                )
                tmp_dir = tempfile.mkdtemp(prefix="vllm_tokenizer_patch_")
                try:
                    TOKENIZER_FILES = [
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "special_tokens_map.json",
                        "added_tokens.json",
                        "chat_template.jinja",
                        "generation_config.json",
                    ]

                    for fname in TOKENIZER_FILES:
                        src = os.path.join(tokenizer_name, fname)
                        if os.path.exists(src):
                            shutil.copy(src, tmp_dir)

                    config_path = os.path.join(tmp_dir, "tokenizer_config.json")
                    with open(config_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    if cfg.get("tokenizer_class") in ("TokenizersBackend",):
                        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
                    if "extra_special_tokens" in cfg:
                        cfg["additional_special_tokens"] = cfg.pop(
                            "extra_special_tokens"
                        )

                    with open(config_path, "w", encoding="utf-8") as f:
                        json.dump(cfg, f, indent=2)

                    tokenizer = AutoTokenizer.from_pretrained(
                        tmp_dir,
                        trust_remote_code=trust_remote_code,
                        revision=revision,
                        **kwargs,
                    )
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

            else:
                raise e

        # The special_tokens in tokenizer should also be
        # controlled by do_lower_case in encoder_config
        encoder_config = get_sentence_transformer_tokenizer_config(
            tokenizer_name, revision
        )
        if isinstance(encoder_config, dict) and encoder_config.get(
            "do_lower_case", False
        ):
            special_tokens_map = {
                k: v.lower() for k, v in tokenizer.special_tokens_map.items()
            }
            tokenizer.add_special_tokens(special_tokens_map)

        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            logger.warning(
                "Using a slow tokenizer. This might cause a significant "
                "slowdown. Consider using a fast tokenizer instead."
            )
        tokenizer = get_cached_tokenizer(tokenizer)

    return tokenizer


tokenizer.get_tokenizer = kunlun_get_tokenizer

logger.info_once(
    "[Monkey Patch Applied] >>> vllm.transformer_utils.tokenizer.get_tokenizer \
      --> vllm_kunlun.transformer_utils.tokenizer.kunlun_get_tokenizer"
)
