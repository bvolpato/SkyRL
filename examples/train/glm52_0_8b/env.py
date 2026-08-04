from typing import Any

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput


class AsciiRewardEnv(BaseTextEnv):
    """Reward responses by their fraction of ASCII alphabetic characters."""

    def __init__(self, env_config: Any = None, extras: dict[str, Any] | None = None):
        super().__init__()

    def step(self, action: str) -> BaseTextEnvStepOutput:
        ascii_letters = sum(character.isascii() and character.isalpha() for character in action)
        reward = ascii_letters / max(len(action), 1)
        return BaseTextEnvStepOutput(
            observations=[],
            reward=reward,
            done=True,
            metadata={"ascii_fraction": reward},
        )
