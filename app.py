import io
import json
import os
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
from datasets import Audio, Dataset, DatasetDict, load_dataset, load_from_disk


st.set_page_config(page_title="Local HF Dataset Viewer", layout="wide")
st.title("Local Hugging Face Dataset Viewer (with Audio)")


def ensure_dataset_dict(ds_obj):
    if isinstance(ds_obj, Dataset):
        return DatasetDict({"train": ds_obj})
    return ds_obj


def hf_dataset_to_folder_name(dataset_name: str) -> str:
    cleaned = dataset_name.strip().replace("/", "_")
    return cleaned or "hf_dataset"


@st.cache_resource(show_spinner=False)
def load_local_dataset(
    mode: str,
    path: str,
    builder_name: str,
    hf_dataset_name: str,
    hf_save_root: str,
):
    path = path.strip()
    builder_name = builder_name.strip()
    hf_dataset_name = hf_dataset_name.strip()
    hf_save_root = hf_save_root.strip() or "."

    if mode == "load_from_disk":
        ds = load_from_disk(path)
        return ensure_dataset_dict(ds), path

    if mode == "hf_dataset":
        local_path = os.path.join(hf_save_root, hf_dataset_to_folder_name(hf_dataset_name))
        if os.path.exists(local_path) and not os.path.isdir(local_path):
            raise ValueError(f"Target local dataset path is not a directory: {local_path}")

        if os.path.isdir(local_path):
            ds = load_from_disk(local_path)
        else:
            os.makedirs(hf_save_root, exist_ok=True)
            ds = load_dataset(hf_dataset_name)
            ds.save_to_disk(local_path)
        return ensure_dataset_dict(ds), local_path

    # builder_name examples: parquet / csv / json / audiofolder
    if builder_name:
        ds = load_dataset(builder_name, data_dir=path)
    else:
        ds = load_dataset(path)
    return ensure_dataset_dict(ds), path


def is_audio_like(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and ("array" in value or "path" in value or "bytes" in value)
        and ("sampling_rate" in value or "array" in value or "path" in value or "bytes" in value)
    )


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


def get_audio_filename(audio_value: dict, row_id: int, column_name: str) -> str:
    original_path = audio_value.get("path")
    if original_path:
        return os.path.basename(original_path)
    return f"{sanitize_path_part(column_name)}_{row_id:06d}.wav"


def get_audio_format(path_value: str | None) -> str:
    ext = os.path.splitext(path_value or "")[1].lower()
    if ext == ".mp3":
        return "audio/mpeg"
    if ext == ".flac":
        return "audio/flac"
    if ext == ".ogg":
        return "audio/ogg"
    return "audio/wav"


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
            if not isinstance(audio_value, dict):
                continue

            audio_bytes = audio_value.get("bytes")
            if not isinstance(audio_bytes, (bytes, bytearray)) or not audio_bytes:
                continue

            file_path = os.path.join(col_dir, get_audio_filename(audio_value, row_idx, col))
            if os.path.exists(file_path):
                continue

            with open(file_path, "wb") as f:
                f.write(audio_bytes)

        if row_idx == total_rows - 1 or (row_idx + 1) % update_every == 0:
            done = row_idx + 1
            progress.progress(done / total_rows, text="Saving audio files to local cache...")
            status.caption(f"Saved local audio files for {done:,} / {total_rows:,} rows")

    progress.empty()
    status.empty()
    return column_dirs


def resolve_audio_path(
    audio_value: dict,
    dataset_path: str,
    split: str,
    column_name: str,
    row_id: int,
    audio_dirs: dict[str, str],
) -> str | None:
    if not isinstance(audio_value, dict):
        return None

    raw_path = audio_value.get("path")
    if raw_path and os.path.isabs(raw_path) and os.path.exists(raw_path):
        return raw_path

    column_dir = audio_dirs.get(column_name) or get_audio_column_dir(dataset_path, split, column_name)
    candidate = os.path.join(column_dir, get_audio_filename(audio_value, row_id, column_name))
    if os.path.exists(candidate):
        return candidate

    if raw_path and os.path.exists(raw_path):
        return raw_path

    return None


def to_audio_bytes(audio_value: dict, local_path: str | None = None) -> tuple[bytes | None, str]:
    """
    Convert a Hugging Face Audio cell into bytes plus a format string for st.audio().
    """
    if not isinstance(audio_value, dict):
        return None, "audio/wav"

    if local_path and os.path.exists(local_path):
        with open(local_path, "rb") as f:
            return f.read(), get_audio_format(local_path)

    raw_bytes = audio_value.get("bytes")
    if isinstance(raw_bytes, (bytes, bytearray)) and raw_bytes:
        return bytes(raw_bytes), get_audio_format(audio_value.get("path"))

    array = audio_value.get("array", None)
    sr = audio_value.get("sampling_rate", None)

    # If decoded audio is available, use it directly
    if array is not None and sr is not None:
        arr = np.asarray(array)

        # soundfile expects shape (n,) or (n, channels)
        if arr.dtype != np.float32 and arr.dtype != np.float64:
            arr = arr.astype(np.float32)

        buf = io.BytesIO()
        sf.write(buf, arr, sr, format="WAV")
        return buf.getvalue(), "audio/wav"

    # Fallback: raw file path
    path = audio_value.get("path", None)
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return f.read(), get_audio_format(path)

    return None, get_audio_format(path)


def summarize_value(v: Any) -> str:
    if isinstance(v, dict):
        if is_audio_like(v):
            sr = v.get("sampling_rate", "unknown")
            arr = v.get("array")
            if arr is not None:
                try:
                    n = len(arr)
                except Exception:
                    n = "unknown"
                return f"[audio] sr={sr}, samples={n}"
            if v.get("path"):
                return f"[audio] {v['path']}"
        return json.dumps(v, ensure_ascii=False)[:200]
    if isinstance(v, list):
        return json.dumps(v, ensure_ascii=False)[:200]
    return str(v)


def make_display_df(ds, start: int, end: int) -> pd.DataFrame:
    rows = [ds[i] for i in range(start, end)]
    processed = []
    for row in rows:
        out = {}
        for k, v in row.items():
            out[k] = summarize_value(v)
        processed.append(out)
    return pd.DataFrame(processed)


with st.sidebar:
    st.header("Load dataset")
    mode = st.radio("Source type", ["load_from_disk", "data files / directory", "hf_dataset"], index=0)
    path = ""
    builder_name = ""
    hf_dataset_name = ""
    hf_save_root = "."

    if mode == "hf_dataset":
        hf_dataset_name = st.text_input(
            "HF dataset name",
            value="",
            help="Example: your_org/your_dataset",
        )
        hf_save_root = st.text_input(
            "Local save root",
            value=".",
            help="The dataset will be saved in a subfolder named from the dataset id.",
        )
        if hf_dataset_name.strip():
            st.caption(
                f"Local folder: {os.path.join(hf_save_root, hf_dataset_to_folder_name(hf_dataset_name))}"
            )
    else:
        path = st.text_input("Path", value="./data")
        builder_name = st.text_input(
            "Builder (optional)",
            value="",
            help="Examples: parquet, csv, json, audiofolder. Leave empty only if path itself is a dataset script / supported auto-load target."
        )

if mode == "hf_dataset" and not hf_dataset_name.strip():
    st.info("Enter a Hugging Face dataset name to load and save it locally.")
    st.stop()

if mode != "hf_dataset" and not path:
    st.stop()

try:
    ds_dict, dataset_path = load_local_dataset(mode, path, builder_name, hf_dataset_name, hf_save_root)
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

splits = list(ds_dict.keys())
split = st.selectbox("Split", splits)
ds = ds_dict[split]

st.subheader("Dataset info")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", len(ds))
c2.metric("Columns", len(ds.column_names))
c3.metric("Split", split)

with st.expander("Features", expanded=False):
    st.write(ds.features)

st.caption(f"Local dataset path: {dataset_path}")

audio_columns = []
for col, feature in ds.features.items():
    if isinstance(feature, Audio):
        audio_columns.append(col)

st.write("**Detected audio columns:**", audio_columns if audio_columns else "None")

audio_dirs = prepare_audio_files(ds, dataset_path, split, audio_columns) if audio_columns else {}

st.subheader("Browse rows")
start = st.number_input(
    "Start row",
    min_value=0,
    max_value=max(0, len(ds) - 1),
    value=0,
    step=1,
)
limit = st.slider("Rows to display", min_value=5, max_value=100, value=20)
end = min(start + limit, len(ds))

df = make_display_df(ds, start, end)
df.insert(0, "__row_id__", list(range(start, end)))
st.dataframe(df, use_container_width=True, height=500)

st.subheader("Inspect one row")
row_id = st.number_input(
    "Row id",
    min_value=0,
    max_value=max(0, len(ds) - 1),
    value=int(start),
    step=1,
)
row = ds[int(row_id)]

left, right = st.columns([1, 1])

with left:
    st.write("**Raw row**")
    # Avoid dumping huge arrays directly
    compact_row = {}
    for k, v in row.items():
        if is_audio_like(v):
            local_path = resolve_audio_path(v, dataset_path, split, k, int(row_id), audio_dirs)
            compact_row[k] = {
                "path": v.get("path"),
                "local_path": local_path,
                "sampling_rate": v.get("sampling_rate"),
                "array_len": None if v.get("array") is None else len(v["array"]),
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

            sr = audio_val.get("sampling_rate", "unknown") if isinstance(audio_val, dict) else "unknown"
            arr = audio_val.get("array", None) if isinstance(audio_val, dict) else None
            path_val = audio_val.get("path", None) if isinstance(audio_val, dict) else None
            local_path = resolve_audio_path(audio_val, dataset_path, split, col, int(row_id), audio_dirs)

            meta = {
                "sampling_rate": sr,
                "num_samples": None if arr is None else len(arr),
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
    scan_n = min(len(ds), 5000)
    for i in range(scan_n):
        r = ds[i]
        text_parts = []
        for k, v in r.items():
            if is_audio_like(v):
                continue
            text_parts.append(f"{k}: {summarize_value(v)}")
        blob = " ".join(text_parts)
        if query.lower() in blob.lower():
            item = {"__row_id__": i}
            for k, v in r.items():
                item[k] = summarize_value(v)
            matches.append(item)
        if len(matches) >= 100:
            break

    if matches:
        st.write(f"Found {len(matches)} matches in first {scan_n} rows")
        st.dataframe(pd.DataFrame(matches), use_container_width=True)
    else:
        st.info(f"No matches in first {scan_n} rows.")
