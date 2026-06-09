# Local HF Dataset Viewer

Streamlit app for browsing local or Hugging Face datasets, previewing rows, and rendering audio columns from locally cached files.

## Setup

Install `uv` if needed:

```bash
python -m pip install uv
```

Sync the project environment from [`pyproject.toml`](/Users/qiancheng/Documents/test/pyproject.toml):

```bash
uv sync
```

This creates `.venv` and installs:

- `datasets`
- `numpy`
- `streamlit`
- `pandas`
- `pyarrow`
- `soundfile`
- `torch`

Set your Hugging Face token:

```bash
export HF_TOKEN=your_hf_token
```

## Start Server

```bash
uv run streamlit run app.py
```

If you prefer an activated virtual environment:

```bash
source .venv/bin/activate
streamlit run app.py
```

## What The App Supports

- Load from a Hugging Face dataset name with an optional config name, save it to local disk, and reuse the saved copy on later runs
- Stream the first `N` examples from a Hugging Face dataset for a quick preview without saving the full dataset locally
- Load a dataset from `load_from_disk`
- Load from local data files or a local directory

When loading from Hugging Face Hub, the local folder name is derived from the dataset id by replacing `/` with `_`.
If a config name is provided, it is appended to the folder name so different configs are cached separately.
Streaming preview mode does not create this local dataset folder. It keeps only the selected split's first `N` streamed examples in memory, so total row count is unknown and browsing/search are limited to that preview buffer.

Example:

- `your_org/your_dataset` -> `your_org_your_dataset`
- `your_org/your_dataset` with config `en` -> `your_org_your_dataset__en`

## Audio Behavior

- Audio files are saved locally under `_audio_cache/<split>/<column_name>/`
- Audio saving is skipped if the target column folder already exists
- Decoded audio without a reusable source path is cached as a generated WAV file
- The app renders audio from the saved local files when available

## CLI Utilities

- `uv run python lang_filter.py --source <dataset_path_or_name> --lang-list ja en --output <output_dir>` filters rows whose timestamp languages fall outside the allowed language list
- `uv run python lang_filter_helper.py --repo-name <hf_dataset_name> --lang-list ja en --output <output_dir>` annotates rows with `filter` and `lang_per`, and can optionally keep only matching rows
