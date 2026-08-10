"""Pre-load XPU shared library dependencies before kunlun_ops import.
This must be called before vllm_kunlun's register() loads _custom_ops.py.

The dependency chain:
  libxops_blocks.so → libapiinfer.so → libxpuapi.so (needs libcudart.so.12)
                   → libxpu_blas.so, libxpu_dnn.so, libxpu_flash_attention.so, libbkcl.so

Key: libxpuapi.so is linked against libcudart.so.12 via NEEDED but the symlink
chain /usr/local/xcudart/lib/libcuda.so.1 → libxpucuda.so provides cuda entry
points at dlopen time. Pre-load in the correct order with RTLD_GLOBAL so that
later ctypes.CDLL calls in kunlun_ops/__init__.py resolve successfully.
"""
import ctypes
import logging
import os

logger = logging.getLogger("vllm_kunlun")

# Where the XPU shared libraries live
_XPU_DIR = "/opt/vllm_kunlun/lib/python3.10/site-packages/torch_xmlir"
_XCUDART_DIR = "/usr/local/xcudart/lib"
_KUNLUN_OPS_DIR = "/opt/vllm_kunlun/lib/python3.10/site-packages/kunlun_ops"

# Libraries (in dependency order) to pre-load with RTLD_GLOBAL
_PRELOAD_SOS = [
    # 1. CUDA runtime shim (libcudart.so.12 → libcudart.so.12.9.1.kunlun)
    #    Needed by libxpuapi.so (NEEDED libcudart.so.12)
    ("xcudart", os.path.join(_XCUDART_DIR, "libcudart.so.12")),

    # 2. XPU API layer (libxpuapi.so → libxpu_blas.so → ...)
    ("xpuapi", os.path.join(_XPU_DIR, "libxpuapi.so")),
    ("xpu_blas", os.path.join(_XPU_DIR, "libxpu_blas.so")),
    ("xpu_dnn", os.path.join(_XPU_DIR, "libxpu_dnn.so")),
    ("xpu_flash_attention", os.path.join(_XPU_DIR, "libxpu_flash_attention.so")),
    ("bkcl", os.path.join(_XPU_DIR, "libbkcl.so")),
]


def preload_xpu_libraries():
    """Pre-load all XPU shared libraries with RTLD_GLOBAL.

    Returns True if all libraries loaded successfully.
    Any failure triggers a warning (not fatal) — the caller should still attempt
    the normal registration path.
    """
    all_ok = True
    for name, path in _PRELOAD_SOS:
        if not os.path.exists(path):
            logger.warning("[KunlunPlugin] preload %s not found at %s", name, path)
            all_ok = False
            continue
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            logger.info("[KunlunPlugin] preload %s OK (%s)", name, path)
        except Exception as exc:
            logger.warning(
                "[KunlunPlugin] preload %s failed (%s): %s", name, path, exc
            )
            all_ok = False
    return all_ok
