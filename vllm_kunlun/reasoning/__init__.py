import importlib

from vllm.reasoning import ReasoningParserManager

"""
Reasoning parser registration module for vLLM Kunlun.
"""


REASONING_PARSERS = {
    "minimax_m2": (".minimax_m2_reasoning_parser", "MiniMaxM2ReasoningParser"),
    "kimi_k2": (".kimi_k2_reasoning_parser", "KimiK2ReasoningParser"),
}


def register_reasoning_parser():
    """
    Register all reasoning parsers with the ReasoningParserManager.
    """
    for name, (module_path, class_name) in REASONING_PARSERS.items():
        module = importlib.import_module(module_path, package=__name__)
        cls = getattr(module, class_name)
        ReasoningParserManager.register_module(name=name, module=cls)
