"""Tool parser registration module for vLLM Kunlun."""

import importlib

from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

# Map parser name to (relative module path, class name). Add new parsers here.
TOOL_PARSERS = {}


def register_tool_parser():
    """Register all tool parsers with the ToolParserManager."""
    for name, (module_path, class_name) in TOOL_PARSERS.items():
        module = importlib.import_module(module_path, package=__name__)
        cls = getattr(module, class_name)
        ToolParserManager.register_module(name=name, module=cls)
