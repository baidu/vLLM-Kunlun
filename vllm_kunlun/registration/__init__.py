"""Registration infrastructure for the Kunlun vLLM plugin.

vLLM discovers this plugin at startup and calls ``vllm_kunlun.register()``,
which drives everything in this package: it first runs the startup stages in
``bootstrap``, then installs the import dispatcher from ``import_hooks`` so
the remaining adaptations happen automatically as vLLM imports its modules.

Modules, in suggested reading order:

    import_hooks:     The engine.  A custom ``__import__`` wrapper plus the
        post-import hook registry.  Start here to understand the mechanism;
        it contains no Kunlun-specific content of its own.
    module_redirects: Content, part 1.  Upstream vLLM modules replaced
        wholesale by Kunlun implementations before anyone imports them.
    compat_patches:   Content, part 2.  Targeted patches applied to upstream
        modules right after they finish importing.
    bootstrap:        Ordered startup helpers (CUDA extension stubs, operator
        registration, native extension loading, torch shims) that run during
        platform discovery, before the import dispatcher takes over.
"""
