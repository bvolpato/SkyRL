#!/bin/bash
set -euo pipefail
set -x

DATA_DIR="${DATA_DIR:-/tmp/skyrl-glm-boolean}"
STEPS="${STEPS:-12}"

export VLLM_MLA_FORCE_DENSE=1
export RAY_DEDUP_LOGS=0
export TOKENIZERS_PARALLELISM=false

.venv/bin/python examples/train/glm52_0_8b/dataset.py --output-dir "$DATA_DIR"

.venv/bin/python -m examples.train.glm52_0_8b.main \
  "data.train_data=['$DATA_DIR/train.parquet']" \
  "data.val_data=['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.use_kl_loss=false \
  trainer.policy.model.path=inference-optimization/GLM-5.2-0.8B-A0.8B \
  trainer.policy.model.bitsandbytes_4bit.enabled=true \
  trainer.policy.model.lora.rank=8 \
  trainer.policy.model.lora.alpha=8 \
  "trainer.policy.model.lora.target_modules=['q_a_proj']" \
  trainer.policy.optimizer_config.lr=0.0005 \
  trainer.policy.use_torch_compile=false \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp \
  trainer.placement.policy_num_gpus_per_node=1 \
  trainer.placement.ref_num_gpus_per_node=1 \
  generator.inference_engine.num_engines=1 \
  generator.inference_engine.tensor_parallel_size=1 \
  trainer.epochs="$STEPS" \
  trainer.max_training_steps="$STEPS" \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=2 \
  trainer.policy_mini_batch_size=2 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.eval_before_train=true \
  trainer.eval_interval=20 \
  trainer.ckpt_interval=-1 \
  trainer.max_prompt_length=64 \
  trainer.flash_attn=false \
  trainer.remove_microbatch_padding=false \
  generator.sampling_params.max_generate_length=1 \
  "generator.sampling_params.additional_kwargs={allowed_token_ids:[15,16]}" \
  "generator.eval_sampling_params={max_generate_length: 1, temperature: 0.0, additional_kwargs: {allowed_token_ids: [15, 16]}}" \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.45 \
  generator.inference_engine.enable_prefix_caching=false \
  generator.inference_engine.max_num_batched_tokens=128 \
  generator.inference_engine.max_num_seqs=4 \
  generator.inference_engine.engine_init_kwargs.enable_chunked_prefill=false \
  generator.inference_engine.engine_init_kwargs.max_model_len=128 \
  generator.batched=true \
  generator.max_input_length=64 \
  generator.max_turns=1 \
  generator.n_samples_per_prompt=4 \
  environment.env_class=boolean_reward \
  trainer.logger=console \
  trainer.print_example_interval=-1 \
  trainer.log_path=/tmp/skyrl-glm-boolean-qlora-logs \
  trainer.ckpt_path=/tmp/skyrl-glm-boolean-qlora-ckpt \
  trainer.resume_mode=null \
  "$@"
