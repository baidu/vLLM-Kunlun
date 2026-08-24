"""Reasoning parser registration module for vLLM Kunlun."""

import importlib

from vllm.reasoning import ReasoningParserManager

# Map parser name to (relative module path, class name). Add new parsers here.
REASONING_PARSERS = {}


def register_reasoning_parser():
    """Register all reasoning parsers with the ReasoningParserManager."""
    for name, (module_path, class_name) in REASONING_PARSERS.items():
        module = importlib.import_module(module_path, package=__name__)
        cls = getattr(module, class_name)
        ReasoningParserManager.register_module(name=name, module=cls)
