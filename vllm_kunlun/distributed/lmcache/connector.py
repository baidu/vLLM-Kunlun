# Copyright (c) 2026 Baidu, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Use upstream registration unless this package supplies legacy cache views."""

from importlib.util import find_spec

if find_spec(f"{__package__}.legacy_layout") is None:
    from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
else:
    from .legacy_layout import LMCacheMPConnector

__all__ = ["LMCacheMPConnector"]
