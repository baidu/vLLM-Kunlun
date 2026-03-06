#
# setup.py for vllm_kunlun
#

import os
import shutil

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ROOT_DIR = os.path.dirname(__file__)

ext_modules = [
    CppExtension(
        name="vllm_kunlun._kunlun",
        sources=["vllm_kunlun/csrc/utils.cpp"],
        include_dirs=[
            "vllm_kunlun/csrc",
            "/usr/local/cuda/include",
        ],
        library_dirs=["/usr/local/cuda/lib64"],
        extra_compile_args=["-O3"],
    )
]


class CustomBuildExt(BuildExtension):
    def run(self):
        super().run()
        for ext in self.extensions:
            ext_path = self.get_ext_fullpath(ext.name)
            file_name = os.path.basename(ext_path)
            target_path = os.path.join("vllm_kunlun", file_name)

            if os.path.exists(target_path):
                os.remove(target_path)
            shutil.copyfile(ext_path, target_path)
            print(f"[BuildExt] Copied {ext_path} -> {target_path}")


if __name__ == "__main__":

    setup(
        name="vllm_kunlun",
        version="v1.0",
        author="vLLM-Kunlun team",
        license="Apache 2.0",
        description="vLLM Kunlun3 backend plugin",
        packages=find_packages(exclude=("docs", "examples", "tests*")),
        package_data={"vllm_kunlun": ["_kunlun.so", "so/*.so", "include/*.h"]},
        python_requires=">=3.10",
        ext_modules=ext_modules,
        cmdclass={
            "build_ext": CustomBuildExt,
        },
        entry_points={
            "vllm.platform_plugins": ["kunlun = vllm_kunlun:register"],
            "vllm.general_plugins": [
                "kunlun_model = vllm_kunlun:register_model",
                "kunlun_quant = vllm_kunlun:register_quant_method",
            ],
            # FusedMoE CustomOp OOT
            "vllm.plugins": [
                "kunlun_fused_moe = vllm_kunlun.ops.fused_moe:register_kunlun_fused_moe_ops"
            ],
            "console_scripts": ["vllm_kunlun = vllm_kunlun.entrypoints.main:main"],
        },
    )
