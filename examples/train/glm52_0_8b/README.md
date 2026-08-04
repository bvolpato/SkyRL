# GLM-5.2 0.8B RL smoke

This fork-only harness exercises a complete single-GPU GRPO QLoRA loop for
`inference-optimization/GLM-5.2-0.8B-A0.8B`: vLLM rollout, rule reward, GRPO
advantages, FSDP backward, optimizer update, and LoRA weight sync.

It depends on SkyRL PRs #1968 and #1974 plus a vLLM v0.23 compatibility port.
Install the combined environment from this branch:

```bash
uv sync --frozen --extra fsdp --extra qlora --extra dev
VLLM_USE_PRECOMPILED=1 uv pip install --no-deps --editable \
  "git+https://github.com/bvolpato/vllm.git@bvolpato/glm-dsa-dense-fallback#egg=vllm"
```

Run one update or the validated 12-update smoke:

```bash
STEPS=1 bash examples/train/glm52_0_8b/run_grpo_qlora.sh
STEPS=12 bash examples/train/glm52_0_8b/run_grpo_qlora.sh
```

The dense MLA fallback is suitable only for short smoke sequences below this
checkpoint's `index_topk=2048`, where sparse top-k covers the complete prefix.
It is not native long-context DSA validation. Prefix caching stays disabled
because vLLM v0.23 compressed-cache gather kernels reject the model's 192-wide
head.
