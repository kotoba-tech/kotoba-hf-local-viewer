import argparse
from collections import Counter

from datasets import Dataset, DatasetDict, load_dataset


def normalize_lang(lang: str) -> str:
    return lang.strip().casefold()


class LangFilter:

    def __init__(self, lang_list: list[str], num_proc: int = 1):
        self.lang_list = {normalize_lang(lang) for lang in lang_list}
        self.num_proc = num_proc

    def __call__(self, dataset: Dataset) -> Dataset:
        assert "timestamps" in dataset.column_names, "timestamps column is required"

        def map_fn(example: dict):
            timestamps = example["timestamps"]
            lang_count: Counter[str] = Counter()
            keep = True

            for timestamp in timestamps:
                lang = timestamp.get("lang") if isinstance(timestamp, dict) else None
                if not isinstance(lang, str) or not lang.strip():
                    keep = False
                    continue
                normalized_lang = normalize_lang(lang)
                lang_count[normalized_lang] += 1
                if normalized_lang not in self.lang_list:
                    keep = False

            timestamp_count = len(timestamps)
            if timestamp_count == 0:
                lang_per = {}
            else:
                lang_per = {
                    key: count / timestamp_count
                    for key, count in lang_count.items()
                }
            return {"filter": keep, "lang_per": lang_per}

        if self.num_proc > 1:
            return dataset.map(map_fn, num_proc=self.num_proc)
        return dataset.map(map_fn)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Annotate a dataset with lang filter metadata based on the lang field "
            "inside each timestamps item."
        )
    )
    parser.add_argument(
        "--repo-name",
        required=True,
        help="Hugging Face dataset repo name.",
    )
    parser.add_argument(
        "--config-name",
        default=None,
        help="Optional Hugging Face dataset config name.",
    )
    parser.add_argument(
        "--lang-list",
        nargs="+",
        required=True,
        help="Allowed languages for timestamps[*].lang.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional split name when the loaded dataset has multiple splits.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path for save_to_disk.",
    )
    parser.add_argument(
        "--keep-only",
        action="store_true",
        help="Drop rows whose computed filter value is false before saving.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=1,
        help="Number of processes for dataset.map.",
    )
    return parser.parse_args()


def load_input_dataset(repo_name: str, config_name: str | None, split: str | None) -> Dataset:
    if split:
        dataset = load_dataset(repo_name, name=config_name, split=split)
        if not isinstance(dataset, Dataset):
            raise TypeError(f"Expected Dataset for split '{split}', got {type(dataset).__name__}")
        return dataset

    dataset = load_dataset(repo_name, name=config_name)
    if isinstance(dataset, Dataset):
        return dataset
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Unsupported dataset type: {type(dataset).__name__}")
    if "train" in dataset:
        return dataset["train"]
    if len(dataset) == 1:
        return dataset[next(iter(dataset.keys()))]
    raise ValueError(f"--split is required. Available splits: {list(dataset.keys())}")


def main() -> None:
    args = parse_args()
    dataset = load_input_dataset(args.repo_name, args.config_name, args.split)
    lang_filter = LangFilter(args.lang_list, num_proc=args.num_proc)
    result = lang_filter(dataset)

    if args.keep_only:
        result = result.filter(lambda example: example["filter"])

    print(result)
    if len(result) > 0:
        print(result[0])

    if args.output:
        result.save_to_disk(args.output)
        print(f"saved_to={args.output}")


if __name__ == "__main__":
    main()
