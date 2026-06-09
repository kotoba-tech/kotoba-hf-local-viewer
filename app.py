import io
import json
import os
import shutil
from itertools import islice
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
from datasets import Audio, Dataset, DatasetDict, IterableDataset, load_dataset, load_from_disk


st.set_page_config(page_title="Local HF Dataset Viewer", layout="wide")
st.title("Local Hugging Face Dataset Viewer (with Audio)")

HF_LOCAL_MODE = "hf_dataset"
HF_STREAMING_MODE = "hf_streaming_preview"
SOURCE_TYPE_LABELS = {
    "load_from_disk": "Local saved dataset",
    "data files / directory": "Local data files / directory",
    HF_LOCAL_MODE: "HF dataset: full local copy",
    HF_STREAMING_MODE: "HF dataset: streaming preview",
}


def ensure_dataset_dict(ds_obj):
    if isinstance(ds_obj, Dataset):
        return DatasetDict({"train": ds_obj})
    if isinstance(ds_obj, IterableDataset):
        return {"train": ds_obj}
    return ds_obj


def hf_dataset_to_folder_name(dataset_name: str, config_name: str = "") -> str:
    cleaned = dataset_name.strip().replace("/", "_")
    config_cleaned = config_name.strip().replace("/", "_")
    if config_cleaned:
        return f"{cleaned}__{config_cleaned}"
    return cleaned or "hf_dataset"


@st.cache_resource(show_spinner=False)
def load_local_dataset(
    mode: str,
    path: str,
    builder_name: str,
    hf_dataset_name: str,
    hf_config_name: str,
    hf_save_root: str,
):
    path = path.strip()
    builder_name = builder_name.strip()
    hf_dataset_name = hf_dataset_name.strip()
    hf_config_name = hf_config_name.strip()
    hf_save_root = hf_save_root.strip() or "."

    if mode == "load_from_disk":
        ds = load_from_disk(path)
        return ensure_dataset_dict(ds), path

    if mode == HF_LOCAL_MODE:
        local_path = os.path.join(
            hf_save_root,
            hf_dataset_to_folder_name(hf_dataset_name, hf_config_name),
        )
        if os.path.exists(local_path) and not os.path.isdir(local_path):
            raise ValueError(f"Target local dataset path is not a directory: {local_path}")

        if os.path.isdir(local_path):
            ds = load_from_disk(local_path)
        else:
            os.makedirs(hf_save_root, exist_ok=True)
            ds = load_dataset(hf_dataset_name, name=hf_config_name or None)
            ds.save_to_disk(local_path)
        return ensure_dataset_dict(ds), local_path

    if mode == HF_STREAMING_MODE:
        ds = load_dataset(hf_dataset_name, name=hf_config_name or None, streaming=True)
        return ensure_dataset_dict(ds), f"hf://{hf_dataset_name}"

    # builder_name examples: parquet / csv / json / audiofolder
    if builder_name:
        ds = load_dataset(builder_name, data_dir=path)
    else:
        ds = load_dataset(path)
    return ensure_dataset_dict(ds), path


@st.cache_resource(show_spinner=False)
def load_streaming_preview_rows(
    hf_dataset_name: str,
    hf_config_name: str,
    split: str,
    preview_rows: int,
) -> list[dict[str, Any]]:
    ds = load_dataset(
        hf_dataset_name.strip(),
        name=hf_config_name.strip() or None,
        split=split,
        streaming=True,
    )
    return list(islice(ds, max(0, int(preview_rows))))


def is_audio_like(value: Any) -> bool:
    if get_audio_payload(value) is not None:
        return True
    return hasattr(value, "get_all_samples") and hasattr(value, "metadata")


def get_audio_payload(audio_value: Any) -> dict[str, Any] | None:
    if (
        isinstance(audio_value, dict)
        and ("array" in audio_value or "path" in audio_value or "bytes" in audio_value)
        and ("sampling_rate" in audio_value or "array" in audio_value or "path" in audio_value or "bytes" in audio_value)
    ):
        return audio_value

    encoded = getattr(audio_value, "_hf_encoded", None)
    if (
        isinstance(encoded, dict)
        and ("array" in encoded or "path" in encoded or "bytes" in encoded)
        and ("sampling_rate" in encoded or "array" in encoded or "path" in encoded or "bytes" in encoded)
    ):
        return encoded

    return None


def sanitize_path_part(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip())
    return cleaned or "audio"


def get_audio_cache_root(dataset_path: str) -> str:
    abs_path = os.path.abspath(dataset_path)
    base_dir = abs_path if os.path.isdir(abs_path) else os.path.dirname(abs_path) or os.getcwd()
    return os.path.join(base_dir, "_audio_cache")


def get_audio_column_dir(dataset_path: str, split: str, column_name: str) -> str:
    return os.path.join(
        get_audio_cache_root(dataset_path),
        sanitize_path_part(split),
        sanitize_path_part(column_name),
    )


def get_generated_audio_filename(row_id: int, column_name: str) -> str:
    return f"{sanitize_path_part(column_name)}_{row_id:06d}.wav"


def get_audio_filename(audio_value: Any, row_id: int, column_name: str) -> str:
    payload = get_audio_payload(audio_value) or {}
    original_path = payload.get("path")
    if original_path:
        return os.path.basename(original_path)
    return get_generated_audio_filename(row_id, column_name)


def get_audio_format(path_value: str | None) -> str:
    ext = os.path.splitext(path_value or "")[1].lower()
    if ext == ".mp3":
        return "audio/mpeg"
    if ext == ".flac":
        return "audio/flac"
    if ext == ".ogg":
        return "audio/ogg"
    return "audio/wav"


def get_audio_path_value(audio_value: Any) -> str | None:
    payload = get_audio_payload(audio_value) or {}
    path_value = payload.get("path")
    return path_value if isinstance(path_value, str) and path_value else None


def get_audio_bytes_value(audio_value: Any) -> bytes | None:
    payload = get_audio_payload(audio_value) or {}
    raw_bytes = payload.get("bytes")
    if isinstance(raw_bytes, (bytes, bytearray)) and raw_bytes:
        return bytes(raw_bytes)
    return None


def get_audio_sampling_rate(audio_value: Any) -> int | None:
    payload = get_audio_payload(audio_value) or {}
    sampling_rate = payload.get("sampling_rate")
    if isinstance(sampling_rate, (int, float)):
        return int(sampling_rate)

    metadata = getattr(audio_value, "metadata", None)
    metadata_sr = getattr(metadata, "sample_rate", None)
    if isinstance(metadata_sr, (int, float)):
        return int(metadata_sr)

    return None


def normalize_audio_array(array: Any) -> np.ndarray:
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()

    arr = np.asarray(array)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim >= 2:
        arr = np.squeeze(arr)
        if arr.ndim == 2 and arr.shape[0] <= 8 and arr.shape[0] < arr.shape[1]:
            arr = arr.T

    return arr


def get_audio_num_samples(audio_value: Any) -> int | None:
    payload = get_audio_payload(audio_value) or {}
    array = payload.get("array")
    if array is not None:
        try:
            return int(normalize_audio_array(array).shape[0])
        except Exception:
            return None

    metadata = getattr(audio_value, "metadata", None)
    duration_seconds = getattr(metadata, "duration_seconds", None)
    sample_rate = getattr(metadata, "sample_rate", None)
    if isinstance(duration_seconds, (int, float)) and isinstance(sample_rate, (int, float)):
        return int(round(duration_seconds * sample_rate))

    if hasattr(audio_value, "get_all_samples"):
        try:
            samples = audio_value.get_all_samples()
            return int(normalize_audio_array(samples.data).shape[0])
        except Exception:
            return None

    return None


def get_audio_array_and_sampling_rate(audio_value: Any) -> tuple[np.ndarray | None, int | None]:
    payload = get_audio_payload(audio_value) or {}
    array = payload.get("array")
    sample_rate = get_audio_sampling_rate(audio_value)

    if array is None and hasattr(audio_value, "get_all_samples"):
        try:
            samples = audio_value.get_all_samples()
            array = samples.data
            sample_rate = getattr(samples, "sample_rate", sample_rate)
        except Exception:
            array = None

    if array is None or sample_rate is None:
        return None, sample_rate

    arr = normalize_audio_array(array)

    if arr.dtype != np.float32 and arr.dtype != np.float64:
        arr = arr.astype(np.float32)

    return arr, int(sample_rate)


def prepare_audio_files(ds, dataset_path: str, split: str, audio_columns: list[str]) -> dict[str, str]:
    column_dirs = {
        col: get_audio_column_dir(dataset_path, split, col)
        for col in audio_columns
    }
    pending_columns = {col: col_dir for col, col_dir in column_dirs.items() if not os.path.isdir(col_dir)}
    if not pending_columns:
        return column_dirs

    for col_dir in pending_columns.values():
        os.makedirs(col_dir, exist_ok=True)

    total_rows = len(ds)
    progress = st.progress(0, text="Saving audio files to local cache...")
    status = st.empty()
    update_every = max(1, min(200, total_rows // 100 or 1))

    for row_idx in range(total_rows):
        row = ds[row_idx]
        for col, col_dir in pending_columns.items():
            audio_value = row.get(col)
            if not is_audio_like(audio_value):
                continue

            original_file_path = os.path.join(col_dir, get_audio_filename(audio_value, row_idx, col))
            generated_file_path = os.path.join(col_dir, get_generated_audio_filename(row_idx, col))
            if os.path.exists(original_file_path) or os.path.exists(generated_file_path):
                continue

            audio_bytes = get_audio_bytes_value(audio_value)
            if audio_bytes is not None:
                with open(original_file_path, "wb") as f:
                    f.write(audio_bytes)
                continue

            raw_path = get_audio_path_value(audio_value)
            if raw_path and os.path.exists(raw_path):
                shutil.copyfile(raw_path, original_file_path)
                continue

            arr, sr = get_audio_array_and_sampling_rate(audio_value)
            if arr is not None and sr is not None:
                sf.write(generated_file_path, arr, sr, format="WAV")

        if row_idx == total_rows - 1 or (row_idx + 1) % update_every == 0:
            done = row_idx + 1
            progress.progress(done / total_rows, text="Saving audio files to local cache...")
            status.caption(f"Saved local audio files for {done:,} / {total_rows:,} rows")

    progress.empty()
    status.empty()
    return column_dirs


def resolve_audio_path(
    audio_value: Any,
    dataset_path: str,
    split: str,
    column_name: str,
    row_id: int,
    audio_dirs: dict[str, str],
) -> str | None:
    if not is_audio_like(audio_value):
        return None

    raw_path = get_audio_path_value(audio_value)
    if raw_path and os.path.isabs(raw_path) and os.path.exists(raw_path):
        return raw_path

    column_dir = audio_dirs.get(column_name) or get_audio_column_dir(dataset_path, split, column_name)
    candidate = os.path.join(column_dir, get_audio_filename(audio_value, row_id, column_name))
    if os.path.exists(candidate):
        return candidate

    generated_candidate = os.path.join(column_dir, get_generated_audio_filename(row_id, column_name))
    if os.path.exists(generated_candidate):
        return generated_candidate

    if raw_path and os.path.exists(raw_path):
        return raw_path

    return None


def to_audio_bytes(audio_value: Any, local_path: str | None = None) -> tuple[bytes | None, str]:
    """
    Convert a Hugging Face Audio cell into bytes plus a format string for st.audio().
    """
    if not is_audio_like(audio_value):
        return None, "audio/wav"

    if local_path and os.path.exists(local_path):
        with open(local_path, "rb") as f:
            return f.read(), get_audio_format(local_path)

    raw_bytes = get_audio_bytes_value(audio_value)
    if raw_bytes is not None:
        return raw_bytes, get_audio_format(get_audio_path_value(audio_value))

    arr, sr = get_audio_array_and_sampling_rate(audio_value)
    if arr is not None and sr is not None:
        buf = io.BytesIO()
        sf.write(buf, arr, sr, format="WAV")
        return buf.getvalue(), "audio/wav"

    # Fallback: raw file path
    path = get_audio_path_value(audio_value)
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return f.read(), get_audio_format(path)

    return None, get_audio_format(path)


def summarize_value(v: Any, include_audio_stats: bool = True) -> str:
    if is_audio_like(v):
        sr = get_audio_sampling_rate(v) or "unknown"
        if include_audio_stats:
            n = get_audio_num_samples(v)
            if n is not None:
                return f"[audio] sr={sr}, samples={n}"
        path = get_audio_path_value(v)
        if path:
            return f"[audio] {path}"
        return f"[audio] sr={sr}"
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)[:200]
    if isinstance(v, list):
        return json.dumps(v, ensure_ascii=False)[:200]
    return str(v)


def make_display_df(
    ds,
    start: int,
    end: int,
    include_audio_stats: bool = True,
) -> pd.DataFrame:
    rows = [ds[i] for i in range(start, end)]
    processed = []
    for row in rows:
        out = {}
        for k, v in row.items():
            out[k] = summarize_value(v, include_audio_stats=include_audio_stats)
        processed.append(out)
    return pd.DataFrame(processed)


def get_column_names(ds, rows: list[dict[str, Any]] | None = None) -> list[str]:
    column_names = getattr(ds, "column_names", None)
    if column_names:
        return list(column_names)

    features = getattr(ds, "features", None)
    if features:
        return list(features.keys())

    if rows:
        return list(rows[0].keys())

    return []


def get_audio_columns(ds, rows: list[dict[str, Any]] | None = None) -> list[str]:
    audio_columns = []
    features = getattr(ds, "features", None)
    if features:
        for col, feature in features.items():
            if isinstance(feature, Audio):
                audio_columns.append(col)

    if audio_columns or not rows:
        return audio_columns

    for col in rows[0].keys():
        if any(is_audio_like(row.get(col)) for row in rows[:5]):
            audio_columns.append(col)
    return audio_columns


with st.sidebar:
    st.header("Load dataset")
    mode = st.radio(
        "Source type",
        ["load_from_disk", "data files / directory", HF_LOCAL_MODE, HF_STREAMING_MODE],
        index=0,
        format_func=lambda value: SOURCE_TYPE_LABELS.get(value, value),
    )
    path = ""
    builder_name = ""
    hf_dataset_name = ""
    hf_config_name = ""
    hf_save_root = "."
    streaming_preview_n = 20

    if mode in (HF_LOCAL_MODE, HF_STREAMING_MODE):
        hf_dataset_name = st.text_input(
            "HF dataset name",
            value="",
            help="Example: your_org/your_dataset",
        )
        hf_config_name = st.text_input(
            "HF config name (optional)",
            value="",
            help="Example: default, en, or another dataset configuration name.",
        )
        if mode == HF_LOCAL_MODE:
            hf_save_root = st.text_input(
                "Local save root",
                value=".",
                help="The dataset will be saved in a subfolder named from the dataset id and config.",
            )
            if hf_dataset_name.strip():
                st.caption(
                    "Local folder: "
                    f"{os.path.join(hf_save_root, hf_dataset_to_folder_name(hf_dataset_name, hf_config_name))}"
                )
        else:
            streaming_preview_n = st.number_input(
                "Examples to preview",
                min_value=1,
                max_value=1000,
                value=20,
                step=5,
                help="Only the first N examples for the selected split are streamed into memory.",
            )
    else:
        path = st.text_input("Path", value="./data")
        builder_name = st.text_input(
            "Builder (optional)",
            value="",
            help="Examples: parquet, csv, json, audiofolder. Leave empty only if path itself is a dataset script / supported auto-load target."
        )

is_streaming_mode = mode == HF_STREAMING_MODE

if mode in (HF_LOCAL_MODE, HF_STREAMING_MODE) and not hf_dataset_name.strip():
    st.info("Enter a Hugging Face dataset name to load.")
    st.stop()

if mode not in (HF_LOCAL_MODE, HF_STREAMING_MODE) and not path:
    st.stop()

try:
    ds_dict, dataset_path = load_local_dataset(
        mode,
        path,
        builder_name,
        hf_dataset_name,
        hf_config_name,
        hf_save_root,
    )
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

splits = list(ds_dict.keys())
split = st.selectbox("Split", splits)
ds = ds_dict[split]

if is_streaming_mode:
    with st.spinner(f"Streaming first {int(streaming_preview_n):,} examples from `{split}`..."):
        preview_rows = load_streaming_preview_rows(
            hf_dataset_name,
            hf_config_name,
            split,
            int(streaming_preview_n),
        )
    row_source = preview_rows
    total_rows = len(preview_rows)
else:
    preview_rows = None
    row_source = ds
    total_rows = len(ds)

column_names = get_column_names(ds, preview_rows)

st.subheader("Dataset info")
if is_streaming_mode:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", "Unknown")
    c2.metric("Preview rows", total_rows)
    c3.metric("Columns", len(column_names))
    c4.metric("Split", split)
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", total_rows)
    c2.metric("Columns", len(column_names))
    c3.metric("Split", split)

with st.expander("Features", expanded=False):
    features = getattr(ds, "features", None)
    if features:
        st.write(features)
    else:
        st.write({"columns": column_names})

if is_streaming_mode:
    st.caption(
        f"Streaming preview from {dataset_path}: first {total_rows:,} examples are held in memory; "
        "no local dataset copy is saved."
    )
else:
    st.caption(f"Local dataset path: {dataset_path}")

if total_rows == 0:
    st.info("No rows are available for this preview.")
    st.stop()

audio_columns = get_audio_columns(ds, preview_rows)
st.write("**Detected audio columns:**", audio_columns if audio_columns else "None")

if audio_columns and is_streaming_mode:
    audio_dirs = {}
    st.caption("Streaming preview renders audio only for the selected row; it does not build a full audio cache.")
else:
    audio_dirs = prepare_audio_files(ds, dataset_path, split, audio_columns) if audio_columns else {}

st.subheader("Browse rows")
limit_min = 1 if total_rows < 5 else 5
limit_max = min(100, total_rows)
limit_default = min(20, limit_max)
start = st.number_input(
    "Start row",
    min_value=0,
    max_value=max(0, total_rows - 1),
    value=0,
    step=1,
)
limit = st.slider("Rows to display", min_value=limit_min, max_value=limit_max, value=limit_default)
end = min(start + limit, total_rows)

df = make_display_df(row_source, start, end, include_audio_stats=not is_streaming_mode)
df.insert(0, "__row_id__", list(range(start, end)))
st.dataframe(df, use_container_width=True, height=500)

st.subheader("Inspect one row")
row_id = st.number_input(
    "Row id",
    min_value=0,
    max_value=max(0, total_rows - 1),
    value=int(start),
    step=1,
)
row = row_source[int(row_id)]

left, right = st.columns([1, 1])

with left:
    st.write("**Raw row**")
    # Avoid dumping huge arrays directly
    compact_row = {}
    for k, v in row.items():
        if is_audio_like(v):
            local_path = None if is_streaming_mode else resolve_audio_path(v, dataset_path, split, k, int(row_id), audio_dirs)
            compact_row[k] = {
                "path": get_audio_path_value(v),
                "local_path": local_path,
                "sampling_rate": get_audio_sampling_rate(v),
                "array_len": get_audio_num_samples(v),
            }
        else:
            compact_row[k] = v
    st.json(compact_row)

with right:
    st.write("**Audio preview**")
    if not audio_columns:
        st.info("No audio columns detected in this row.")
    else:
        for col in audio_columns:
            audio_val = row[col]
            st.markdown(f"**Column: `{col}`**")

            sr = get_audio_sampling_rate(audio_val) or "unknown"
            num_samples = get_audio_num_samples(audio_val)
            path_val = get_audio_path_value(audio_val)
            local_path = None if is_streaming_mode else resolve_audio_path(audio_val, dataset_path, split, col, int(row_id), audio_dirs)

            meta = {
                "sampling_rate": sr,
                "num_samples": num_samples,
                "path": path_val,
                "local_path": local_path,
            }
            st.json(meta)

            audio_bytes, audio_format = to_audio_bytes(audio_val, local_path=local_path)
            if audio_bytes is not None:
                st.audio(audio_bytes, format=audio_format)
            else:
                st.warning("Could not render this audio cell.")

st.subheader("Simple text search")
query = st.text_input("Contains text in any non-audio field")
if query:
    matches = []
    scan_n = min(total_rows, 5000)
    for i in range(scan_n):
        r = row_source[i]
        text_parts = []
        for k, v in r.items():
            if is_audio_like(v):
                continue
            text_parts.append(f"{k}: {summarize_value(v)}")
        blob = " ".join(text_parts)
        if query.lower() in blob.lower():
            item = {"__row_id__": i}
            for k, v in r.items():
                item[k] = summarize_value(v, include_audio_stats=not is_streaming_mode)
            matches.append(item)
        if len(matches) >= 100:
            break

    search_scope = f"previewed {scan_n} rows" if is_streaming_mode else f"first {scan_n} rows"
    if matches:
        st.write(f"Found {len(matches)} matches in {search_scope}")
        st.dataframe(pd.DataFrame(matches), use_container_width=True)
    else:
        st.info(f"No matches in {search_scope}.")
