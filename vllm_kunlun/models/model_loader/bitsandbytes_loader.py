from typing import Callable


class BitsAndBytesModelLoader:
    """Model loader to load model weights with BitAndBytes quantization."""

    possible_config_file_names = ["adapter_config.json"]

    def __init__(self):

        # Save the module names without sharding.
        self.unsharded_weights_modules: list[str] = []
        # Save the module names that are sharded by column.
        self.column_sharded_weights_modules: list[str] = []
        # Modules whose weights might have fused on disk
        # we need their output_sizes to make shard in flight correctly with TP
        self.maybe_fused_weights_modules: dict[str, list[int]] = {}
        # Store all module names (from transformers) that support
        # BNB quantization.
        self.target_modules: list[str] = []
        # Store the mapping of expert parameters for MoE models.
        self.expert_params_mapping: list[tuple[str, str, int, str]] = []
        # mapping weight names from transformers to vllm.
        self.weight_mapper: Callable = lambda name: name
        self.pre_quant: bool = False
        self.load_8bit: bool = False
        self.is_pool_model: bool = False
