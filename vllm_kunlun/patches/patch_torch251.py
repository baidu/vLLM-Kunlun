"""Patch vLLM 0.19.x so it can run on PyTorch 2.5.1."""

from __future__ import annotations

import os
import shutil
import site
import sys

_site_packages = []
try:
    _site_packages = site.getsitepackages()
except Exception:
    _site_packages = []

if _site_packages:
    SITE_PACKAGES = _site_packages[0]
else:
    SITE_PACKAGES = sys.prefix

PATCHES = [
    {
        "file": "vllm/compilation/passes/inductor_pass.py",
        "lines": "22",
        "desc": (
            "Backfill torch._inductor.custom_graph_pass with a fallback base "
            "class when PyTorch 2.5.1 does not ship this module."
        ),
        "old": "from torch._inductor.custom_graph_pass import CustomGraphPass",
        "new": """\
try:
    from torch._inductor.custom_graph_pass import CustomGraphPass
except ModuleNotFoundError:
    class CustomGraphPass:
        def __call__(self, graph):
            return None

        def uuid(self):
            return self.__class__.__name__""",
    },
    {
        "file": "vllm/compilation/decorators.py",
        "lines": "547-550",
        "desc": "Adapt patched_inline_call to the PyTorch 2.5.1 callback signature.",
        "old": """\
        def patched_inline_call(self_: Any) -> Any:
            code = self_.f_code
            self.compilation_config.traced_files.add(code.co_filename)
            return inline_call(self_)""",
        "new": """\
        def patched_inline_call(parent, func, args, kwargs):
            code = func.get_code()
            self.compilation_config.traced_files.add(code.co_filename)
            return inline_call(parent, func, args, kwargs)""",
    },
    {
        "file": "vllm/model_executor/layers/attention/attention.py",
        "lines": "437-440",
        "desc": "Avoid torch.Size tracing issues on PyTorch 2.5.1.",
        "old": """\
                output_shape = torch.Size(
                    (num_tokens, self.num_heads * self.head_size_v)
                )""",
        "new": """\
                output_shape = torch.empty(
                    (num_tokens, self.num_heads * self.head_size_v)
                ).size()""",
    },
    {
        "file": "vllm/model_executor/models/openpangu.py",
        "lines": "779-783",
        "desc": "Avoid torch.Size tracing issues in OpenPangu attention output.",
        "old": """\
            output_shape=torch.Size(
                [q.shape[0], q.shape[1] // self.head_dim * self.v_channels]
            ),""",
        "new": """\
            output_shape=torch.empty(
                [q.shape[0], q.shape[1] // self.head_dim * self.v_channels]
            ).size(),""",
        "required": False,
    },
    {
        "file": "vllm/compilation/piecewise_backend.py",
        "lines": "221-227",
        "desc": (
            "Remove bundled_autograd_cache patching during serialize because "
            "that config flag does not exist on PyTorch 2.5.1."
        ),
        "old": """\
            with torch._functorch.config.patch("bundled_autograd_cache", True):
                entry = fn.serialize()

                f = io.BytesIO()
                StandaloneCompiledArtifactsPickler(f).dump(entry)
                result = f.getvalue()
            return result""",
        "new": """\
            entry = fn.serialize()

            f = io.BytesIO()
            StandaloneCompiledArtifactsPickler(f).dump(entry)
            result = f.getvalue()
            return result""",
    },
    {
        "file": "vllm/compilation/compiler_interface.py",
        "lines": "778-780",
        "desc": (
            "Skip bundled_autograd_cache assignment because the attribute is "
            "absent on PyTorch 2.5.1."
        ),
        "old": """\
def set_functorch_config() -> None:
    if not envs.VLLM_USE_MEGA_AOT_ARTIFACT:
        torch._functorch.config.bundled_autograd_cache = False""",
        "new": """\
def set_functorch_config() -> None:
    if not envs.VLLM_USE_MEGA_AOT_ARTIFACT:
        pass""",
    },
    {
        "file": "vllm/distributed/parallel_state.py",
        "lines": "32",
        "desc": "Import List so PyTorch 2.5 tracing does not trip on list[int].",
        "old": "from typing import TYPE_CHECKING, Any, Protocol",
        "new": "from typing import TYPE_CHECKING, Any, List, Protocol",
    },
    {
        "file": "vllm/distributed/parallel_state.py",
        "lines": "187,239",
        "desc": "Replace list[int] annotations with typing.List[int].",
        "old": "    output_shape: list[int],",
        "new": "    output_shape: List[int],",
        "count": 0,
    },
    {
        "file": "vllm/v1/structured_output/utils.py",
        "lines": "121",
        "desc": (
            "Force xgrammar torch-native backend to avoid backend selection "
            "mismatches on Kunlun."
        ),
        "old": "        xgr.apply_token_bitmask_inplace(logits, grammar_bitmask, indices=index_tensor)",
        "new": (
            "        xgr.apply_token_bitmask_inplace("
            'logits, grammar_bitmask, indices=index_tensor, backend="torch_native")'
        ),
        "required": False,
    },
    {
        "file": "vllm/compilation/wrapper.py",
        "lines": "199-203",
        "desc": "Fall back to nullcontext() when torch.compiler.set_stance is unavailable.",
        "old": """\
            ctx = (
                nullcontext()
                if self.first_compile or not self.evaluate_guards
                else torch.compiler.set_stance("fail_on_recompile")
            )""",
        "new": """\
            if self.first_compile or not self.evaluate_guards:
                ctx = nullcontext()
            elif hasattr(torch.compiler, "set_stance"):
                ctx = torch.compiler.set_stance("fail_on_recompile")
            else:
                ctx = nullcontext()""",
        "required": False,
    },
]


def get_full_path(relative_path: str) -> str:
    return os.path.join(SITE_PACKAGES, relative_path)


def _backup_path(file_path: str) -> str:
    return f"{file_path}.bak"


def apply_patch(patch: dict[str, object], dry_run: bool = False) -> str:
    file_path = get_full_path(str(patch["file"]))
    required = bool(patch.get("required", True))
    count = int(patch.get("count", 1))

    print(
        f"\n{'[DRY RUN] ' if dry_run else ''}"
        f"Patch: {patch['file']} (lines {patch['lines']})"
    )
    print(f"  Description: {patch['desc']}")

    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            content = handle.read()
    except FileNotFoundError:
        if required:
            print(f"  FAIL: File not found: {file_path}")
            return "failed"
        print(f"  SKIP: Optional file not found: {file_path}")
        return "skipped"

    old = str(patch["old"])
    new = str(patch["new"])

    if new in content and old not in content:
        print("  SKIP: Already patched.")
        return "skipped"

    if old not in content:
        if required:
            print("  FAIL: Original code not found.")
            return "failed"
        print("  SKIP: Optional patch target not found.")
        return "skipped"

    occurrences = content.count(old)
    replacements = occurrences if count == 0 else min(count, occurrences)
    print(
        f"  Found {occurrences} occurrence(s), "
        f"will replace {'all' if count == 0 else replacements}."
    )

    if dry_run:
        print("  OK: Would apply patch.")
        return "success"

    backup_path = _backup_path(file_path)
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
        print(f"  Backup created: {backup_path}")
    else:
        print(f"  Backup already exists: {backup_path}")

    if count == 0:
        new_content = content.replace(old, new)
    else:
        new_content = content.replace(old, new, count)

    with open(file_path, "w", encoding="utf-8") as handle:
        handle.write(new_content)

    print(f"  SUCCESS: Patch applied ({replacements} replacement(s)).")
    return "success"


def revert_all() -> None:
    print("=" * 64)
    print("  Reverting all patches from .bak backups")
    print("=" * 64)

    seen: set[str] = set()
    for patch in PATCHES:
        file_path = get_full_path(str(patch["file"]))
        if file_path in seen:
            continue
        seen.add(file_path)

        backup_path = _backup_path(file_path)
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, file_path)
            print(f"  Restored: {file_path}")
        else:
            print(f"  No backup found for: {file_path}")


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    revert = "--revert" in sys.argv

    if revert:
        revert_all()
        return

    print("=" * 64)
    print("  vLLM 0.19.x Compatibility Patch for PyTorch 2.5.1")
    print(f"  Site-packages: {SITE_PACKAGES}")
    print(f"  Total patches: {len(PATCHES)}")
    if dry_run:
        print("  Mode: DRY RUN")
    print("=" * 64)

    results = {"success": 0, "skipped": 0, "failed": 0}
    for patch in PATCHES:
        result = apply_patch(patch, dry_run=dry_run)
        results[result] += 1

    print("\n" + "=" * 64)
    print(
        f"  Results: {results['success']} applied | "
        f"{results['skipped']} skipped | {results['failed']} failed"
    )
    if not dry_run and results["success"] > 0:
        print("  Original files are backed up as .bak")
        print("  Revert with: python patch_torch251.py --revert")
    print("=" * 64)

    if results["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
