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
- `streamlit`
- `pandas`
- `pyarrow`
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

- Load from a Hugging Face dataset name, save it to local disk, and reuse the saved copy on later runs
- Load a dataset from `load_from_disk`
- Load from local data files or a local directory

When loading from Hugging Face Hub, the local folder name is derived from the dataset id by replacing `/` with `_`.

Example:

- `your_org/your_dataset` -> `your_org_your_dataset`

## Audio Behavior

- Audio files are saved locally under `_audio_cache/<split>/<column_name>/`
- Audio saving is skipped if the target column folder already exists
- The app renders audio from the saved local files when available
