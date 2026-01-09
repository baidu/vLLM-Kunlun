# Contributing

## Building and Testing
It's recommended to set up a local development environment to build vllm-kunlun and run tests
before you submit a PR.

### Run models locally

After completing Run lint setup which is shown in quicksatrt, you can run your changed locally:

```{code-block} bash
   :substitutions:

python -m vllm.entrypoints.openai.api_server \
      --host 0.0.0.0 \
      --port 8356 \
      --model your_modified_models \
      --gpu-memory-utilization 0.9 \
      --trust-remote-code \
      --max-model-len 32768 \
      --tensor-parallel-size 1 \
      --dtype float16 \
      --max_num_seqs 128 \
      --max_num_batched_tokens 32768 \
      --block-size 128 \
      --no-enable-prefix-caching \
      --no-enable-chunked-prefill \
      --distributed-executor-backend mp \
      --served-model-name your_modified_models \
      --compilation-config '{"splitting_ops": ["vllm.unified_attention", 
                                                "vllm.unified_attention_with_output",
                                                "vllm.unified_attention_with_output_kunlun",
                                                "vllm.mamba_mixer2", 
                                                "vllm.mamba_mixer", 
                                                "vllm.short_conv", 
                                                "vllm.linear_attention", 
                                                "vllm.plamo2_mamba_mixer", 
                                                "vllm.gdn_attention", 
                                                "vllm.sparse_attn_indexer"]}' \ 
```
Please save a screenshot of your service running successfully, and attach an accuracy report.

### Submit the commit

```bash
# Commit changed files using `-s`
git commit -sm "your commit info"
```

🎉 Congratulations! You have completed the development environment setup.


## PR Title and Classification

Only specific types of PRs will be reviewed. The PR title is prefixed appropriately to indicate the type of change. Please use one of the following:

- `[Attention]` for new features or optimization in attention.
- `[Communicator]` for new features or optimization in communicators.
- `[ModelRunner]` for new features or optimization in model runner.
- `[Platform]` for new features or optimization in platform.
- `[Worker]` for new features or optimization in worker.
- `[Core]` for new features or optimization  in the core vllm-kunlun logic (such as platform, attention, communicators, model runner)
- `[Kernel]` for changes affecting compute kernels and ops.
- `[Bugfix]` for bug fixes.
- `[Doc]` for documentation fixes and improvements.
- `[Test]` for tests (such as unit tests).
- `[CI]` for build or continuous integration improvements.
- `[Misc]` for PRs that do not fit the above categories. Please use this sparingly.

:::{note}
If the PR spans more than one category, please include all relevant prefixes.
:::

## Others

If you find any problem when contributing, you can join our slack group to talk with us and then feel free to submit a PR to improve the doc to help other developers. 
