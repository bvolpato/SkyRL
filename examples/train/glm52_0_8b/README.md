# GLM-5.2 0.8B boolean RL smoke

This fork-only harness exercises a complete single-GPU GRPO QLoRA loop for
`inference-optimization/GLM-5.2-0.8B-A0.8B`: vLLM rollout, exact boolean
truth-table reward, GRPO advantages, FSDP backward, optimizer update, and LoRA
weight sync. Prompts contain only symbolic truth tables such as `0^1=1`; each
sample completes one missing `0` or `1` output. vLLM's allowed-token mask limits
rollouts to the checkpoint's single-token `0` and `1` IDs, ensuring exact
rewards produce GRPO variance even when the base model otherwise emits random
text.

It depends on SkyRL PRs #1968 and #1974 plus a vLLM v0.23 compatibility port.
Install the combined environment from this branch:

```bash
uv sync --frozen --extra fsdp --extra qlora --extra dev
VLLM_USE_PRECOMPILED=1 uv pip install --no-deps --editable \
  "git+https://github.com/bvolpato/vllm.git@bvolpato/glm-dsa-dense-fallback#egg=vllm"
```

Run one update or a longer learning smoke:

```bash
STEPS=1 bash examples/train/glm52_0_8b/run_grpo_qlora.sh
STEPS=200 bash examples/train/glm52_0_8b/run_grpo_qlora.sh
```

On an RTX 5070 Ti, the 200-step overfit run improved deterministic accuracy on
the same 12 cases from 50% at step 0 to 91.7% at steps 160, 180, and 200. Mean
sampled reward rose from 0.4875 over the first 20 updates to 0.7750 over the
last 20. GRPO produced nonzero advantages and gradients on 179 of 200 updates;
zero-gradient updates occurred when all four samples for each prompt received
the same reward.

The dense MLA fallback is suitable only for short smoke sequences below this
checkpoint's `index_topk=2048`, where sparse top-k covers the complete prefix.
It is not native long-context DSA validation. Prefix caching and chunked prefill
stay disabled because vLLM v0.23 compressed-cache gather kernels reject the
model's 192-wide head.
