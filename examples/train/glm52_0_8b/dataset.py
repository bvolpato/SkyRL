import argparse
from pathlib import Path

from datasets import Dataset


TRUTH_TABLES = {
    "&": ("0", "0", "0", "1"),
    "|": ("0", "1", "1", "1"),
    "^": ("0", "1", "1", "0"),
}
INPUTS = (("0", "0"), ("0", "1"), ("1", "0"), ("1", "1"))


def build_example(operator: str, missing_index: int) -> dict:
    outputs = TRUTH_TABLES[operator]
    rows = []
    for index, ((left, right), output) in enumerate(zip(INPUTS, outputs, strict=True)):
        suffix = "" if index == missing_index else output
        rows.append(f"{left}{operator}{right}={suffix}")

    return {
        "data_source": "glm52_boolean_smoke",
        "prompt": [{"role": "user", "content": "\n".join(rows)}],
        "env_class": "boolean_reward",
        "reward_spec": {"method": "rule", "ground_truth": outputs[missing_index]},
        "extra_info": {"operator": operator, "missing_index": missing_index},
    }


def build_dataset() -> Dataset:
    return Dataset.from_list(
        [build_example(operator, missing_index) for operator in TRUTH_TABLES for missing_index in range(len(INPUTS))]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/skyrl-glm-boolean"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset()
    dataset.to_parquet(args.output_dir / "train.parquet")
    dataset.to_parquet(args.output_dir / "validation.parquet")


if __name__ == "__main__":
    main()
