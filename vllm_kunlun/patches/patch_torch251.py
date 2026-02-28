"""
vLLM Compatibility Patch Script for PyTorch 2.5.1
==================================================

Why are these patches necessary?
--------------------------------
vLLM (v0.15.x) was developed and tested against newer versions of PyTorch
(2.7+), which introduced several API changes that are NOT available in
PyTorch 2.5.1. Running vLLM on PyTorch 2.5.1 without these patches will
result in AttributeError or TypeError exceptions at runtime.

Specifically, the following incompatibilities exist:

1. `torch._functorch.config.bundled_autograd_cache` — This configuration
   option was introduced in PyTorch 2.7+. It does not exist in PyTorch
   2.5.1, so any code referencing it (either as a direct attribute or via
   `torch._functorch.config.patch(...)`) will raise an AttributeError.

2. `patched_inline_call(self_: Any)` signature — The inline call hook
   signature changed between PyTorch versions. In PyTorch 2.5.1, it
   expects `(parent, func, args, kwargs)` rather than `(self_: Any)`.

3. `torch.Size(...)` constructor — In certain compilation contexts under
   PyTorch 2.5.1, `torch.Size()` does not work correctly during tracing.
   Using `torch.empty().size` is a compatible workaround.

4. `list[int]` type hint — Python 3.10 supports `list[int]` in type
   annotations, but certain runtime introspection paths (especially
   within torch.compile / FX tracing) may fail to resolve built-in
   generic aliases. Using `typing.List[int]` avoids this.

When should these patches be removed?
--------------------------------------
These patches are TEMPORARY workarounds for PyTorch 2.5.1 compatibility.
They should be REMOVED once the environment is upgraded to PyTorch 2.9+,
where all referenced APIs are stable and available. Keeping these patches
on PyTorch 2.9+ may cause unexpected behavior.

Usage:
------
    python patch_torch251.py           # Apply all patches
    python patch_torch251.py --dry-run # Preview changes without modifying files
    python patch_torch251.py --revert  # Restore all files from .bak backups
"""

import os
import shutil
import site
import sys

# ============================================================
# Base path for the target Python environment
# ============================================================
# Determine the site-packages directory dynamically to keep this
# script portable across different Python installations.
_site_packages_list = []
try:
    _site_packages_list = site.getsitepackages()
except Exception:
    _site_packages_list = []
if _site_packages_list:
    # Prefer the first returned site-packages directory.
    SITE_PACKAGES = _site_packages_list[0]
else:
    # Fallback: use the current environment prefix.
    SITE_PACKAGES = sys.prefix

# ============================================================
# Patch definitions
# Each patch contains:
#   - file:      target file path (relative to SITE_PACKAGES)
#   - desc:      human-readable description
#   - lines:     original line numbers (for reference only)
#   - old:       the exact code to find and replace
#   - new:       the replacement code
#   - count:     (optional) number of occurrences to replace.
#                Defaults to 1. Use 0 to replace ALL occurrences.
#
# IMPORTANT: When multiple patches target the same file, the
# import/header patches should come BEFORE the body patches to
# ensure correct dependency order.
# ============================================================
PATCHES = [
    # ----------------------------------------------------------
    # Patch 1: decorators.py — Fix inline call signature
    # ----------------------------------------------------------
    {
        "file": "vllm/compilation/decorators.py",
        "desc": (
            "Fix patched_inline_call signature. "
            "PyTorch 2.5.1 uses (parent, func, args, kwargs) instead of "
            "(self_: Any). The self_.f_code attribute does not exist in "
            "2.5.1; use func.get_code() instead."
        ),
        "lines": "505-508",
        "old": """\
        def patched_inline_call(self_: Any) -> Any:
            code = self_.f_code
            self.compilation_config.traced_files.add(code.co_filename)
            return inline_call(self_)""",
        "new": """\
        def patched_inline_call(parent, func, args, kwargs):
            # [PyTorch 2.5.1 compat] Use func.get_code() instead of
            # self_.f_code, which is not available in this version.
            code = func.get_code()

            # Note: vLLM 0.15.x uses self.compilation_config directly
            # (not self.vllm_config.compilation_config)
            self.compilation_config.traced_files.add(code.co_filename)

            # Pass all four arguments as expected by PyTorch 2.5.1
            return inline_call(parent, func, args, kwargs)""",
    },
    # ----------------------------------------------------------
    # Patch 2: layer.py — Fix torch.Size during tracing
    # ----------------------------------------------------------
    {
        "file": "vllm/attention/layer.py",
        "desc": (
            "Replace torch.Size() with torch.empty().size. "
            "In PyTorch 2.5.1, torch.Size() does not work correctly "
            "during torch.compile tracing. torch.empty().size produces "
            "a compatible result."
        ),
        "lines": "378-380",
        "old": """\
                output_shape = torch.Size(
                    (num_tokens, self.num_heads * self.head_size_v)
                )""",
        "new": """\
                output_shape = torch.empty(
                    (num_tokens, self.num_heads * self.head_size_v)
                ).size()""",
    },
    # ----------------------------------------------------------
    # Patch 3: piecewise_backend.py — Remove bundled_autograd_cache
    #          context manager (serialize path)
    # ----------------------------------------------------------
    {
        "file": "vllm/compilation/piecewise_backend.py",
        "desc": (
            "Remove bundled_autograd_cache context manager around "
            "serialization. torch._functorch.config.bundled_autograd_cache "
            "does not exist in PyTorch 2.5.1."
        ),
        "lines": "180-185",
        "old": """\
            with torch._functorch.config.patch("bundled_autograd_cache", True):
                entry = fn.serialize()

                f = io.BytesIO()
                StandaloneCompiledArtifactsPickler(f).dump(entry)
                result = f.getvalue()""",
        "new": """\
            entry = fn.serialize()

            f = io.BytesIO()
            StandaloneCompiledArtifactsPickler(f).dump(entry)
            result = f.getvalue()""",
    },
    # ----------------------------------------------------------
    # Patch 4: piecewise_backend.py — Remove bundled_autograd_cache
    #          context manager (compile path)
    # ----------------------------------------------------------
    {
        "file": "vllm/compilation/piecewise_backend.py",
        "desc": (
            "Remove bundled_autograd_cache context manager around "
            "compilation. Same reason as above — the config option is "
            "unavailable in PyTorch 2.5.1."
        ),
        "lines": "243-254",
        "old": """\
                with (
                    torch._functorch.config.patch("bundled_autograd_cache", True),
                ):
                    range_entry.runnable = self.vllm_backend.compiler_manager.compile(
                        self.graph,
                        args_list,
                        self.vllm_backend.inductor_config,
                        self.compilation_config,
                        compile_range=range_entry.compile_range,
                        graph_index=self.piecewise_compile_index,
                        num_graphs=self.total_piecewise_compiles,
                    )""",
        "new": """\
                range_entry.runnable = self.vllm_backend.compiler_manager.compile(
                    self.graph,
                    args_list,
                    self.vllm_backend.inductor_config,
                    self.compilation_config,
                    compile_range=range_entry.compile_range,
                    graph_index=self.piecewise_compile_index,
                    num_graphs=self.total_piecewise_compiles,
                )""",
    },
    # ----------------------------------------------------------
    # Patch 5: compiler_interface.py — Comment out unavailable config
    # ----------------------------------------------------------
    {
        "file": "vllm/compilation/compiler_interface.py",
        "desc": (
            "Comment out bundled_autograd_cache assignment. "
            "This attribute does not exist in torch._functorch.config "
            "on PyTorch 2.5.1 and will raise AttributeError."
        ),
        "lines": "654-656",
        "old": """\
def set_functorch_config() -> None:
    if not envs.VLLM_USE_MEGA_AOT_ARTIFACT:
        torch._functorch.config.bundled_autograd_cache = False""",
        "new": """\
def set_functorch_config() -> None:
    if not envs.VLLM_USE_MEGA_AOT_ARTIFACT:
        # [PyTorch 2.5.1 compat] bundled_autograd_cache is not available.
        # TODO: Restore this line after upgrading to PyTorch 2.9+
        # torch._functorch.config.bundled_autograd_cache = False
        pass""",
    },
    # ----------------------------------------------------------
    # Patch 6: parallel_state.py — Add List to typing imports
    #          (must be applied BEFORE Patch 7)
    # ----------------------------------------------------------
    {
        "file": "vllm/distributed/parallel_state.py",
        "desc": (
            "Add List to the typing import. Required by Patch 7 which "
            "changes list[int] to List[int]. Without this import, "
            "List would be undefined."
        ),
        "lines": "36",
        "old": "from typing import Any, Optional",
        "new": "from typing import Any, List, Optional",
    },
    # ----------------------------------------------------------
    # Patch 7: parallel_state.py — Use List[int] instead of list[int]
    # ----------------------------------------------------------
    {
        "file": "vllm/distributed/parallel_state.py",
        "desc": (
            "Replace built-in generic `list[int]` with `List[int]` from "
            "typing. While Python 3.10 supports `list[int]` in annotations, "
            "certain runtime introspection paths used by torch.compile / "
            "FX tracing may fail to resolve it. Using `typing.List[int]` "
            "is the safe, backward-compatible form."
        ),
        "lines": "173, 225",
        # Replace ALL occurrences in the file
        "count": 0,
        "old": "    output_shape: list[int],",
        "new": "    output_shape: List[int],",
    },
]


# ============================================================
# Core logic
# ============================================================


def get_full_path(relative_path: str) -> str:
    """Construct the full file path from SITE_PACKAGES."""
    return os.path.join(SITE_PACKAGES, relative_path)


def apply_patch(patch: dict, dry_run: bool = False) -> str:
    """
    Apply a single patch.
    Returns: 'success', 'skipped', or 'failed'
    """
    file_path = get_full_path(patch["file"])
    desc = patch["desc"]
    # How many occurrences to replace: default 1, use 0 for all
    count = patch.get("count", 1)

    print(
        f"\n{'[DRY RUN] ' if dry_run else ''}"
        f"Patch: {patch['file']} (lines {patch['lines']})"
    )
    print(f"  Description: {desc}")

    # 1. Read the file
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"  FAIL: File not found: {file_path}")
        return "failed"

    # 2. Check if already patched (idempotent)
    if patch["new"] in content and patch["old"] not in content:
        print("  SKIP: Already patched.")
        return "skipped"

    # 3. Verify the original code exists
    if patch["old"] not in content:
        print(
            "  FAIL: Original code not found. The file may have been "
            "modified or the vLLM version may differ."
        )
        return "failed"

    # Count how many occurrences will be replaced
    num_occurrences = content.count(patch["old"])
    replacements = num_occurrences if count == 0 else min(count, num_occurrences)
    print(
        f"  Found {num_occurrences} occurrence(s), "
        f"will replace {'all' if count == 0 else replacements}."
    )

    if dry_run:
        print("  OK: Would apply patch.")
        return "success"

    # 4. Backup (only once per file, never overwrite existing .bak)
    backup_path = file_path + ".bak"
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
        print(f"  Backup created: {backup_path}")
    else:
        print(f"  Backup already exists, not overwriting: {backup_path}")

    # 5. Replace and write back
    if count == 0:
        # Replace ALL occurrences
        new_content = content.replace(patch["old"], patch["new"])
    else:
        new_content = content.replace(patch["old"], patch["new"], count)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"  SUCCESS: Patch applied ({replacements} replacement(s)).")
    return "success"


def revert_all():
    """Restore all patched files from their .bak backups."""
    print("=" * 64)
    print("  Reverting all patches from .bak backups")
    print("=" * 64)

    seen = set()
    for patch in PATCHES:
        file_path = get_full_path(patch["file"])
        if file_path in seen:
            continue
        seen.add(file_path)

        backup_path = file_path + ".bak"
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, file_path)
            print(f"  Restored: {file_path}")
        else:
            print(f"  No backup found for: {file_path}")

    print("\nRevert complete.")


def main():
    dry_run = "--dry-run" in sys.argv
    revert = "--revert" in sys.argv

    if revert:
        revert_all()
        return

    print("=" * 64)
    print("  vLLM Compatibility Patch for PyTorch 2.5.1")
    print(f"  Total patches: {len(PATCHES)}")
    if dry_run:
        print("  Mode: DRY RUN (no files will be modified)")
    print("")
    print("  NOTE: These patches are TEMPORARY. Remove them after")
    print("        upgrading to PyTorch 2.9+.")
    print("=" * 64)

    results = {"success": 0, "skipped": 0, "failed": 0}

    for patch in PATCHES:
        result = apply_patch(patch, dry_run=dry_run)
        results[result] += 1

    print("\n" + "=" * 64)
    print(
        f"  Results: "
        f"{results['success']} applied  |  "
        f"{results['skipped']} skipped  |  "
        f"{results['failed']} failed  |  "
        f"{len(PATCHES)} total"
    )
    if not dry_run and results["success"] > 0:
        print("  Original files backed up as .bak")
        print("  To revert: python patch_torch251.py --revert")
    print("=" * 64)

    # Exit with non-zero code if any patch failed
    if results["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
