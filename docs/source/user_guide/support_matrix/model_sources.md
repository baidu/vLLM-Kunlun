<!--
#
# Copyright (c) 2026 Baidu, Inc. All Rights Reserved.
#
# This file is a part of the vllm-kunlun project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
-->

# Model Source Guidance

This page gives practical guidance for finding and organizing model weights for
vLLM-Kunlun. It does not imply that vLLM-Kunlun publishes or mirrors weights on
any specific model hub.

## Supported Model Names and Weight Sources

The supported model matrix describes model families and runtime capabilities. A
model family name, such as Qwen3 or Qwen3-VL, is not always the same as the
exact `--model` value used at runtime. The `--model` value should point to the
actual weight source you have access to.

Common choices include:

- A Hugging Face repository ID, such as `org/model-name`, when the weights are
  available from Hugging Face and your environment can access that hub.
- A ModelScope repository ID or a local directory downloaded from ModelScope,
  when ModelScope is the preferred source in your environment.
- A private object storage link, such as BOS, if your deployment downloads
  weights from private storage before starting vLLM-Kunlun.
- An AI Studio or approved internal source, if your organization provides the
  model through an internal platform.
- A local filesystem path, such as `/data/models/qwen3`, after the weights have
  already been downloaded or mounted on the host.

## Using Local Paths

For reproducible deployments, downloading or mounting the model first and then
passing a local path to `--model` is often the most explicit option:

```bash
vllm serve /data/models/qwen3
```

The local directory should contain the model configuration, tokenizer files, and
weight files expected by the upstream model format. Keep the directory layout
compatible with the source from which the model was obtained.

## Mapping Sources to `--model`

Use this checklist when preparing a model:

1. Find the exact model variant that matches the supported model family.
2. Confirm that you are allowed to access and use the weights from the selected
   source.
3. Download, mount, or otherwise make the weights available to the runtime
   environment.
4. Pass either the accessible repository ID or the prepared local path as
   `--model`.
5. Keep a record of the source, revision, and local path used for deployment so
   the setup can be reproduced later.

If a model family is listed as supported but no public link is shown in the
matrix, use the model provider's official distribution channel, your
organization's approved source, or a local path prepared from those sources.
