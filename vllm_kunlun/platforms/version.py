"""vllm_kunlun version.py"""

vllm_version = "0.19.0"

xvllm_version_tuple = (0, 19, 0)


def get_xvllm_version():
    major, minor, patch = xvllm_version_tuple
    return f"{major}.{minor}.{patch}"
