import importlib

from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import ToolParserManager

"""
Tool parser registration module for vLLM Kunlun.
"""

TOOL_PARSERS = {
    "minimax_m2": (".minimax_m2_tool_parser", "MinimaxM2ToolParser"),
    "glm47": (".glm47_moe_tool_parser", "Glm47MoeModelToolParser"),
    "kimi_k2": (".kimi_k2_tool_parser", "KimiK2ToolParser"),
}


def register_tool_parser():
    """
    Register all tool parsers with the ToolParserManager.
    """
    for name, (module_path, class_name) in TOOL_PARSERS.items():
        module = importlib.import_module(module_path, package=__name__)
        cls = getattr(module, class_name)
        ToolParserManager.register_module(name=name, module=cls)
