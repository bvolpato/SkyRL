"""``MegatronWorker.init_configs``: a ``transformer_config_kwargs`` key reaches the provider or raises.

Runs ``init_configs`` for Qwen3-0.6B (config and tokenizer only, no weights).

uv run --isolated --extra dev --extra megatron -- pytest tests/backends/skyrl_train/workers/megatron/test_transformer_config_kwargs.py
"""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.megatron

megatron_worker = pytest.importorskip("skyrl.backends.skyrl_train.workers.megatron.megatron_worker")

from skyrl.train.config import MegatronConfig, SkyRLTrainConfig  # noqa: E402


class _Worker(megatron_worker.MegatronWorker):
    def __init__(self):
        self.cfg = SkyRLTrainConfig().trainer
        self.strategy = SimpleNamespace()


@pytest.mark.parametrize(
    "key, rejected",
    [
        ("disable_parameter_transpose_cache", False),
        ("should_pad_vocab", False),
        ("disable_parameter_transpose_cach", True),
        ("rope_parameters", True),
        ("finalize", True),
    ],
)
def test_key_reaches_the_provider_or_raises(key, rejected):
    megatron_config = MegatronConfig(transformer_config_kwargs={key: True})
    worker = _Worker()

    def init_configs():
        worker.init_configs("Qwen/Qwen3-0.6B", megatron_config, {}, megatron_config.transformer_config_kwargs)

    if rejected:
        with pytest.raises(ValueError, match=rf"transformer_config_kwargs sets \['{key}'\]"):
            init_configs()
    else:
        init_configs()
        assert getattr(worker.provider, key) is True
