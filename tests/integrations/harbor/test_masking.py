"""Which failed Harbor trials are masked, and the plain Harbor generator's rows for them."""

import pytest

pytest.importorskip("harbor")

from examples.train_integrations.harbor.entrypoints.main_harbor import (  # noqa: E402
    HarborGeneratorConfig,
)
from skyrl.integration.harbor.harbor_generator import (  # noqa: E402
    HarborTrajectoryOutput,
    build_step_wise_generator_output,
)
from skyrl.integration.harbor.masking import mask_failed_trials  # noqa: E402
from skyrl.train.generators.base import TrajectoryID  # noqa: E402

pytestmark = pytest.mark.integrations


@pytest.mark.parametrize(
    "stop_reasons, fraction, masked_whole",
    [
        # 1 of 8 is below the default share: masked by itself, its siblings train.
        (["agent_timeout"] + ["complete"] * 7, 0.25, False),
        # 2 of 8 reaches it, and so does 1 of 4: the share counts, not the number.
        (["agent_timeout"] * 2 + ["complete"] * 6, 0.25, True),
        (["agent_timeout"] + ["complete"] * 3, 0.25, True),
        (["agent_timeout"] * 3 + ["complete"] * 13, 0.25, False),
        # 0 masks the instance on any timeout; 1 only once every trial timed out.
        (["agent_timeout"] + ["complete"] * 7, 0.0, True),
        (["agent_timeout"] * 7 + ["complete"], 1.0, False),
        # 7 of 25 is exactly 0.28, which comparing against 0.28 * 25 would miss.
        (["agent_timeout"] * 7 + ["complete"] * 18, 0.28, True),
        # A failure masks its instance whatever the fraction; a context-length stop is no failure.
        (["error", "agent_timeout", "complete"], 1.0, True),
        (["context_length", "complete"], 0.0, False),
    ],
)
def test_failed_trials_mask_themselves_or_their_instance(stop_reasons, fraction, masked_whole) -> None:
    # A second, healthy instance in the batch is never touched.
    trials = [("a", stop_reason) for stop_reason in stop_reasons] + [("b", "complete")]

    masked, counts = mask_failed_trials(trials, fraction)

    timeouts = stop_reasons.count("agent_timeout")
    alone = [stop_reason == "agent_timeout" for stop_reason in stop_reasons]
    assert masked == ([True] * len(stop_reasons) if masked_whole else alone) + [False]
    assert counts == {
        "num_timeout_trajectories": timeouts,
        "num_error_trajectories": stop_reasons.count("error"),
        "num_masked_instances": int(masked_whole),
        "num_instance_with_timeouts_unmasked": int(not masked_whole and timeouts > 0),
        "num_traj_masked_individually": 0 if masked_whole else timeouts,
    }


@pytest.mark.parametrize(
    "fraction, rewards, loss_masks", [(0.25, [0.0, 0.0], [[0], [0]]), (1.0, [0.0, 1.0], [[0], [1]])]
)
def test_the_plain_harbor_generator_masks_a_lone_timeout_by_itself(fraction, rewards, loss_masks) -> None:
    turn = [{"prompt_token_ids": [[1, 2]], "completion_token_ids": [[3]], "logprobs": [[-0.5]]}]
    timed_out = HarborTrajectoryOutput(trajectory_id=TrajectoryID("a", 0), stop_reason="agent_timeout")
    passed = HarborTrajectoryOutput(trajectory_id=TrajectoryID("a", 1), rollout_details=turn, reward=1.0)

    out = build_step_wise_generator_output(
        [timed_out, passed], overlong_filtering=False, timeout_mask_instance_fraction=fraction
    )

    assert out["rewards"] == rewards and out["loss_masks"] == loss_masks


def test_the_harbor_generator_config_still_fills_the_generator_defaults() -> None:
    assert HarborGeneratorConfig().eval_sampling_params is not None


@pytest.mark.parametrize("fraction", [-0.1, 1.5])
def test_the_timeout_fraction_must_be_between_zero_and_one(fraction) -> None:
    with pytest.raises(ValueError, match="timeout_mask_instance_fraction must be between 0 and 1"):
        HarborGeneratorConfig(timeout_mask_instance_fraction=fraction)
