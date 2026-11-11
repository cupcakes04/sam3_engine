# Installation

## Requirements

| Item | Requirement |
|---|---|
| Python | 3.12+ (`type Box = ...` syntax in `types.py`) |
| GPU | CUDA GPU strongly recommended (`device="cuda"`); `"cpu"` is accepted by the constructor |
| Packages used directly | `torch`, `torchvision`, `numpy`, `Pillow`, `opencv-python` (`cv2`, video loading), `matplotlib` (visualiser), plus Meta's `sam3` |
| Local files | one SAM 3 checkpoint (`.pt`) and the BPE vocab `bpe_simple_vocab_16e6.txt.gz` |

`pyproject.toml` declares **no** runtime dependencies; `sam3` and the libraries above must already be in your environment.

## 1. Install Meta `sam3`

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e ".[notebooks]"
```

`sam3` may pull `setuptools >= 81`, which removed `pkg_resources` (which `sam3` imports). Fix:

```bash
pip install "setuptools<81.0.0"
```

## 2. Install this package (editable)

From the repository root:

```bash
pip install -e .
```

This exposes the import name `sam3_engine` from anywhere. (The notebook falls back to `from src import ...` when run from the repo root without installing.)

## 3. Get the files

| File | Source | Notes |
|---|---|---|
| `sam3.pt` | Hugging Face `facebook/sam3` | Used for both image and video in the tutorial |
| `sam3.1_multiplex.pt` | Hugging Face `facebook/sam3` | Only for `video_version="sam3.1"`. README: *cannot do PVS and is fairly buggy – prefer `sam3`* |
| `bpe_simple_vocab_16e6.txt.gz` | Bundled at `assets/bpe_simple_vocab_16e6.txt.gz` | Pass this path as `bpe_path` |

The engine **never downloads** anything (`load_from_HF=False` is hard-coded). Paths must point to **files**; passing a directory raises `FileNotFoundError` even though the message mentions "directory".

## 4. Verify

```python
from sam3_engine import SAM3Engine

engine = SAM3Engine(
    image_checkpoint_path="/path/to/sam3.pt",
    video_checkpoint_path="/path/to/sam3.pt",
    bpe_path="/path/to/bpe_simple_vocab_16e6.txt.gz",
)
print(engine.confidence_threshold)   # 0.5 (constructor default)
```

Construction is **lazy**: this only validates the three paths. The heavy models load on first access to `engine.image` / `engine.video` (or `engine.encode_concept`, which touches `engine.image`). Pass `lazy_load=False` to load both immediately.

## Repository assets used by the tutorial

```
assets/images/{wildlife,human,trash,tea1,tea2,aquatic}.jpg
assets/videos/{01_dog,02_cups}.mp4
assets/bpe_simple_vocab_16e6.txt.gz
```

The tutorial uses relative paths like `./assets/images/wildlife.jpg`, so run the notebook with the repo root as the working directory.
