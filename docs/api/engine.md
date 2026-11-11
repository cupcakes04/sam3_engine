# `SAM3Engine`

```python
from sam3_engine import SAM3Engine
```

Unified entry point (`core.py`). Owns configuration and lazily creates the image and video sub-engines.

## Constructor

```python
SAM3Engine(
    image_checkpoint_path: str,
    video_checkpoint_path: str,
    bpe_path: str,
    video_version: Literal["sam3", "sam3.1"] = "sam3",
    device: Literal["cuda", "cpu"] = "cuda",
    precision: Literal["bfloat16", "float16", "float32"] = "bfloat16",
    confidence_threshold: float = 0.5,
    enable_inst_interactivity: bool = True,
    video_compile: bool = False,
    video_max_num_objects: int = 16,
    lazy_load: bool = True,
) -> None
```

| Parameter | Description |
|---|---|
| `image_checkpoint_path` | **Required.** Path to a local `.pt` file (e.g. `sam3.pt`). Loaded for PCS and image PVS. |
| `video_checkpoint_path` | **Required.** Path to a local `.pt` file. `sam3.pt` for `video_version="sam3"`; `sam3.1_multiplex.pt` for `"sam3.1"`. |
| `bpe_path` | **Required.** Path to `bpe_simple_vocab_16e6.txt.gz`. |
| `video_version` | `"sam3"` (recommended) or `"sam3.1"`. README: 3.1 multiplex cannot do PVS and is fairly buggy. |
| `device` | Passed to the image model builder. TF32 is enabled for matmul/cudnn when CUDA is available and `device` starts with `"cuda"`. |
| `precision` | Applied to the **image engine** via `torch.autocast("cuda", dtype)` around `initialise`, `encode_concept`, `predict_pcs`, `predict_pvs` (`"float32"` = no autocast). The **video** predictor hard-codes bfloat16 inside sam3, so a non-`"bfloat16"` value emits a `UserWarning` and video still runs in bfloat16. |
| `confidence_threshold` | Initial detection threshold (float, stored via `float()`). Notebook uses `0.4`. |
| `enable_inst_interactivity` | Builds the interactive (PVS) head in the image model. Must stay `True` to use `image.set_object/add_*/predict_pvs` and PCS→PVS refinement. |
| `video_compile` | Forwarded as `compile=` to `build_sam3_predictor`. |
| `video_max_num_objects` | Forwarded as `max_num_objects=`. Default 16. |
| `lazy_load` | `True`: build models on first use. `False`: build image and video models immediately. |

Raises `FileNotFoundError` if any of the three paths is not an existing **file** (directories are rejected). Returned paths are stored as absolute paths in `bpe_path`, `image_checkpoint_path`, `video_checkpoint_path`.

Public attributes: `device`, `precision`, `confidence_threshold`, `enable_inst_interactivity`, `video_compile`, `video_max_num_objects`, `bpe_path`, `image_checkpoint_path`, `video_checkpoint_path`, `video_version`.

## Properties

### `engine.image -> SAM3ImageEngine`
Builds on first access via `build_sam3_image_model(checkpoint_path, bpe_path, device, enable_segmentation=True, enable_inst_interactivity, load_from_HF=False)`. Subsequent accesses return the same object. Initial threshold = `engine.confidence_threshold` at build time.

### `engine.video -> SAM3VideoEngine`
Builds on first access via `build_sam3_predictor(checkpoint_path, bpe_path, version=video_version, compile=video_compile, max_num_objects=video_max_num_objects, async_loading_frames=False)`.

## Methods

### `set_threshold(threshold: float) -> None`
Sets `engine.confidence_threshold` and pushes it to each sub-engine that has already been built. Sub-engines built later pick up the new value at build time. Video detections computed earlier are not recomputed.

```python
engine.set_threshold(0.35)
print(engine.confidence_threshold)   # 0.35
```

### `encode_concept(text: str) -> CachedConcept`
Delegates to `engine.image.encode_concept(text)` (so it **loads the image model** if needed, even for video-only use; accepted trade-off). Returns the text-backbone output dict with an extra `"__text__"` key. Reusable with `engine.image.add_concept` and `engine.video.add_concept`. In video it acts only as a handle for the text: sam3 re-encodes (and caches per session) the text itself.

### `shutdown() -> None`
Closes the video session (if the video engine was built) and calls `torch.cuda.empty_cache()` when CUDA is available. Image state is not cleared. Models remain referenced by the engine.

## Example

```python
engine = SAM3Engine(
    device="cuda",
    video_checkpoint_path="/ckpt/sam3.pt",
    image_checkpoint_path="/ckpt/sam3.pt",
    bpe_path="/repo/assets/bpe_simple_vocab_16e6.txt.gz",
    video_version="sam3",
    precision="bfloat16",
    confidence_threshold=0.4,
)
engine.set_threshold(0.35)
giraffe = engine.encode_concept("giraffe")
engine.shutdown()
```
