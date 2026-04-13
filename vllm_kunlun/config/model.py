def is_deepseek_mla(self) -> bool:
    if not hasattr(self.hf_text_config, "model_type"):
        return False
    elif self.hf_text_config.model_type in (
        "deepseek_v2",
        "deepseek_v3",
        "deepseek_v32",
        "deepseek_mtp",
        "kimi_k2",
        "longcat_flash",
        "glm_moe_dsa",
        "glm4_moe",
        "glm4_moe_mtp",
        "glm4_moe_lite",
        "glm4_moe_lite_mtp",
    ):
        return self.hf_text_config.kv_lora_rank is not None
    elif self.hf_text_config.model_type == "eagle":
        # if the model is an EAGLE module, check for the
        # underlying architecture
        _nested = self.hf_text_config.model.model_type
        _mla_backing = (
            "deepseek_v2",
            "deepseek_v3",
            "deepseek_v32",
            "glm4_moe",
            "glm4_moe_mtp",
            "glm4_moe_lite",
            "glm4_moe_lite_mtp",
        )
        return (
            _nested in _mla_backing
            and self.hf_text_config.kv_lora_rank is not None
        )
    return False
