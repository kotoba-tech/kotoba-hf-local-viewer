#!/usr/bin/env python3

import argparse
import os
import re
import shutil
from typing import Iterable

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


TIMESTAMP_COLUMN_RE = re.compile(r"^(?:orig_)?timestamps(?:_(?P<lang>[A-Za-z0-9_-]+))?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a dataset and exclude examples whose timestamp languages are not "
            "present in lang_list."
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Dataset path for load_from_disk, or dataset name/path for load_dataset.",
    )
    parser.add_argument(
        "--loader",
        choices=("load_from_disk", "load_dataset"),
        default="load_from_disk",
        help="How to load the dataset source.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional dataset config name when using load_dataset.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Optional data_dir passed to load_dataset.",
    )
    parser.add_argument(
        "--lang-list",
        nargs="+",
        required=True,
        help="Allowed languages. Rows with timestamp langs outside this list are excluded.",
    )
    parser.add_argument(
        "--timestamp-columns",
        nargs="+",
        default=None,
        help="Optional explicit timestamp columns. Defaults to auto-detecting timestamp columns.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Optional split names to filter. Defaults to all splits.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. If omitted, the script prints the summary only.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output directory.",
    )
    return parser.parse_args()


def normalize_lang(lang: str) -> str:
    return lang.strip().casefold()


def ensure_dataset_dict(ds_obj: Dataset | DatasetDict) -> DatasetDict:
    if isinstance(ds_obj, Dataset):
        return DatasetDict({"train": ds_obj})
    return ds_obj


def load_dataset_source(
    source: str,
    loader: str,
    config: str | None,
    data_dir: str | None,
) -> DatasetDict:
    if loader == "load_from_disk":
        return ensure_dataset_dict(load_from_disk(source))
    return ensure_dataset_dict(load_dataset(source, name=config, data_dir=data_dir))


def detect_timestamp_columns(column_names: Iterable[str]) -> list[str]:
    return [column_name for column_name in column_names if TIMESTAMP_COLUMN_RE.match(column_name)]


def extract_langs_from_timestamp_value(column_name: str, value) -> set[str]:
    langs: set[str] = set()

    match = TIMESTAMP_COLUMN_RE.match(column_name)
    if match and match.group("lang"):
        langs.add(normalize_lang(match.group("lang")))

    if isinstance(value, dict):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        return langs

    for item in items:
        if not isinstance(item, dict):
            continue
        lang = item.get("lang")
        if isinstance(lang, str) and lang.strip():
            langs.add(normalize_lang(lang))

    return langs


def build_row_filter(timestamp_columns: list[str], allowed_langs: set[str]):
    def keep_row(*timestamp_values) -> bool:
        found_langs: set[str] = set()
        for column_name, value in zip(timestamp_columns, timestamp_values):
            found_langs.update(extract_langs_from_timestamp_value(column_name, value))
        return found_langs.issubset(allowed_langs)

    return keep_row


def filter_dataset_dict(
    dataset_dict: DatasetDict,
    allowed_langs: set[str],
    explicit_timestamp_columns: list[str] | None,
    selected_splits: list[str] | None,
) -> tuple[DatasetDict, dict[str, dict[str, object]]]:
    split_names = selected_splits or list(dataset_dict.keys())
    missing_splits = [split_name for split_name in split_names if split_name not in dataset_dict]
    if missing_splits:
        raise ValueError(f"Unknown splits: {', '.join(missing_splits)}")

    filtered = DatasetDict()
    summaries: dict[str, dict[str, object]] = {}

    for split_name, split_ds in dataset_dict.items():
        if split_name not in split_names:
            filtered[split_name] = split_ds
            summaries[split_name] = {
                "timestamp_columns": [],
                "before": len(split_ds),
                "after": len(split_ds),
                "filtered": False,
            }
            continue

        timestamp_columns = explicit_timestamp_columns or detect_timestamp_columns(split_ds.column_names)
        missing_columns = [column_name for column_name in timestamp_columns if column_name not in split_ds.column_names]
        if missing_columns:
            raise ValueError(
                f"Split '{split_name}' is missing timestamp columns: {', '.join(missing_columns)}"
            )

        if not timestamp_columns:
            filtered[split_name] = split_ds
            summaries[split_name] = {
                "timestamp_columns": [],
                "before": len(split_ds),
                "after": len(split_ds),
                "filtered": False,
            }
            continue

        keep_row = build_row_filter(timestamp_columns, allowed_langs)
        filtered_split = split_ds.filter(
            keep_row,
            input_columns=timestamp_columns,
            desc=f"Filtering split '{split_name}'",
        )
        filtered[split_name] = filtered_split
        summaries[split_name] = {
            "timestamp_columns": timestamp_columns,
            "before": len(split_ds),
            "after": len(filtered_split),
            "filtered": True,
        }

    return filtered, summaries


def save_dataset(dataset_dict: DatasetDict, output_path: str, overwrite: bool) -> None:
    if os.path.exists(output_path):
        if not overwrite:
            raise FileExistsError(
                f"Output path already exists: {output_path}. Use --overwrite to replace it."
            )
        if os.path.isdir(output_path):
            shutil.rmtree(output_path)
        else:
            os.remove(output_path)
    dataset_dict.save_to_disk(output_path)


def print_summary(
    source: str,
    loader: str,
    output: str | None,
    allowed_langs: set[str],
    summaries: dict[str, dict[str, object]],
) -> None:
    print(f"source={source}")
    print(f"loader={loader}")
    print(f"allowed_langs={sorted(allowed_langs)}")
    for split_name, summary in summaries.items():
        timestamp_columns = summary["timestamp_columns"]
        before = summary["before"]
        after = summary["after"]
        filtered = summary["filtered"]
        if not filtered:
            print(f"split={split_name} before={before} after={after} timestamp_columns=[] filtered=no")
            continue
        print(
            f"split={split_name} before={before} after={after} "
            f"removed={before - after} timestamp_columns={timestamp_columns}"
        )
    if output:
        print(f"saved_to={output}")


def main() -> None:
    args = parse_args()
    allowed_langs = {normalize_lang(lang) for lang in args.lang_list}
    dataset_dict = load_dataset_source(args.source, args.loader, args.config, args.data_dir)
    filtered_dataset, summaries = filter_dataset_dict(
        dataset_dict=dataset_dict,
        allowed_langs=allowed_langs,
        explicit_timestamp_columns=args.timestamp_columns,
        selected_splits=args.splits,
    )

    if args.output:
        save_dataset(filtered_dataset, args.output, args.overwrite)

    print_summary(args.source, args.loader, args.output, allowed_langs, summaries)


if __name__ == "__main__":
    main()
