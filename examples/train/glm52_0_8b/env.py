import re
from typing import Any

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput


class BooleanRewardEnv(BaseTextEnv):
    """Reward the first emitted bit against a truth-table completion."""

    def __init__(self, env_config: Any = None, extras: dict[str, Any] | None = None):
        super().__init__()
        if extras is None:
            raise ValueError("extras are required")
        self.ground_truth = str(extras["reward_spec"]["ground_truth"])

    def step(self, action: str) -> BaseTextEnvStepOutput:
        match = re.search(r"[01]", action)
        answer = match.group(0) if match else None
        reward = 1.0 if answer == self.ground_truth else (0.0 if answer is not None else -0.25)
        return BaseTextEnvStepOutput(
            observations=[],
            reward=reward,
            done=True,
            metadata={"parsed_answer": answer, "ground_truth": self.ground_truth},
        )
