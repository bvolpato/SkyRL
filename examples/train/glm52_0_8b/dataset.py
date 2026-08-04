import argparse
from pathlib import Path

from datasets import Dataset


PROMPTS = [
    "Write one short sentence about a bee.",
    "Write one short sentence about a cloud.",
    "Write one short sentence about a tree.",
    "Write one short sentence about a river.",
]


def build_dataset() -> Dataset:
    return Dataset.from_list(
        [
            {
                "data_source": "glm52_ascii_smoke",
                "prompt": [{"role": "user", "content": prompt}],
                "env_class": "ascii_reward",
                "reward_spec": {"method": "rule"},
                "extra_info": {"prompt_index": index},
            }
            for index, prompt in enumerate(PROMPTS)
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/skyrl-glm-ascii"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset()
    dataset.to_parquet(args.output_dir / "train.parquet")
    dataset.to_parquet(args.output_dir / "validation.parquet")


if __name__ == "__main__":
    main()
