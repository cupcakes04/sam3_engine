# Errors by Method

Exceptions raised by `sam3_engine` code itself (not by `sam3`/torch internals).

## `SAM3Engine`

| Where | Exception | Cause |
|---|---|---|
| `__init__` | `FileNotFoundError` | `bpe_path`, `image_checkpoint_path` or `video_checkpoint_path` isn't an existing file |

## `SAM3ImageEngine`

| Method | Exception | Cause |
|---|---|---|
| `initialise` | `ValueError` | empty image list |
| `initialise` | `TypeError` | not a PIL image / list / tuple |
| `add_concept` | `TypeError` | not `str`, `dict` or list/tuple |
| `predict_pcs` | `RuntimeError` | `initialise` not called, or no concept/exemplar staged |
| `predict_pvs` | `RuntimeError` | `initialise` not called |
| `set_image_idx` | `IndexError` | idx outside `[0, len(images)-1]` |
| `set_object`, `add_points`, `add_box`, `reset_pvs`, `del_object` | `KeyError` | called before `initialise` (no image dicts exist; not guarded) |
| `add_points` | `ValueError` | `len(points) != len(label)` |
| `SAM3ImageEngine.__init__` | `ValueError` | invalid `precision` |

## `SAM3VideoEngine`

| Method | Exception | Cause |
|---|---|---|
| `initialise` | `ValueError` | empty list |
| `initialise` | `TypeError` | not list/tuple/str |
| `initialise(str)` / `load_video_frames` | `FileNotFoundError` | path doesn't exist |
| `load_video_frames` | `ValueError` | cannot open video; no images in directory |
| `load_video_frames` | `TypeError` | unsupported source type |
| `set_frame_idx` | `IndexError` | frame index out of range (also when no frames loaded) |
| `add_points`, `add_box`, `add_concept`, `add_exemplar`, `del_object`, `predict_pvs`, `predict_pcs`, `propagate` | `RuntimeError` | `"Video session not initialised..."` |
| `add_concept` | `TypeError` | not `str`, `dict` or list/tuple |
| `add_points` | `ValueError` | `len(points) != len(label)` |
| `predict_pcs` | `RuntimeError` | `"No concepts registered..."` |
| `propagate` | `RuntimeError` | `"No prompts registered on any frame..."` |

## `Visualiser`

| Method | Exception | Cause |
|---|---|---|
| `visualise_batch` | `ValueError` | empty `results` |

## Messages and fixes

| Message | Fix |
|---|---|
| `Engine not initialised. Call engine.image.initialise(images) first.` | Call `engine.image.initialise([...])` |
| `Video session not initialised. Call engine.video.initialise(video) first.` | Call `engine.video.initialise(frames)` |
| `No concepts registered. Call add_concept() or add_exemplar() first.` | Stage a concept before `video.predict_pcs()` |
| `No prompts registered on any frame...` | `predict_pcs()` or add a point/box on ≥1 frame before `propagate()` |
| `Could not locate local SAM checkpoint...` / `...BPE tokenizer vocabulary...` | Pass a **file** path that exists |
| `ModuleNotFoundError: pkg_resources` | `pip install "setuptools<81"` |
