import os
import re

from setuptools import find_packages, setup

ROOT_DIR = os.path.dirname(__file__)


def get_version():
    version_file = os.path.join(ROOT_DIR, "vllm_kunlun", "platforms", "version.py")
    with open(version_file, encoding="utf-8") as f:
        content = f.read()
    match = re.search(
        r"^__version__\s*=\s*['\"]([^'\"]+)['\"]\s*$",
        content,
        flags=re.MULTILINE,
    )
    if not match:
        raise RuntimeError(f"Unable to find __version__ in {version_file}")
    return match.group(1)


if __name__ == "__main__":

    setup(
        name="vllm_kunlun",
        version=get_version(),
        author="vLLM-Kunlun team",
        license="Apache 2.0",
        description="vLLM Kunlun3 backend plugin",
        packages=find_packages(exclude=("docs", "examples", "tests*")),
        python_requires=">=3.10",
        entry_points={
            "vllm.platform_plugins": ["kunlun = vllm_kunlun:register"],
            "vllm.general_plugins": [
                "kunlun_model = vllm_kunlun:register_model",
                "kunlun_reasoning_parser = vllm_kunlun:register_reasoning_parser",
                "kunlun_tool_parser = vllm_kunlun:register_tool_parser",
            ],
        },
    )
