# Gotchas & Non-Obvious Behaviour

Everything here was verified by reading the source.

## Calls & shapes

### `add_points` is strict
Both `points` and `label` must be lists of equal length, on image and video. Shorthand forms (`(x, y)`, scalar label) raise `TypeError`/`ValueError`; there is no video default label.

### Scalar type detection
Single-vs-list detection uses `isinstance(x[0], (int, float))`. `np.float32` is not a `float`; cast with `float()`.

### Only the last box counts (image PVS)
`add_box` appends; prediction uses `obj.boxes[-1]`. Points all count.

### `name=` in `add_concept`
Image engine: ignored for `str`; honoured for `CachedConcept`; not forwarded to list elements. Video engine: honoured for `str` too.

## PCS

### `predict_pcs` wipes objects
It clears every image's objects before running (by design). PVS edits are lost. It takes no arguments and raises `RuntimeError` if nothing is staged.

### Text and exemplar mixing
The author's notebook note says not to combine them. The code evaluates both when both are staged (exemplar-derived objects are labelled `"exemplar"`). Prefer one mode per run.

### Video: `encode_concept` is only a text handle
sam3's video inference re-encodes the text via `backbone.forward_text` and caches it per session (keyed by the text). The `CachedConcept` you pass to `engine.video.add_concept` contributes only its text.

### `encode_concept` always loads the image model
Even in video-only workflows, `engine.encode_concept` instantiates `engine.image` (accepted). Pass raw strings to `engine.video.add_concept` to avoid loading it.

## PVS

### Selecting an existing PCS id refines it
`set_object(1)` after PCS selects the detected object (keeps label, logits). It does not create a new one. Use an unused id for a new object.

### Slots without predictions are invisible
`set_object(5)` alone adds nothing to `get_results()`. Same if the decoder returns an empty mask (the previous mask is kept).

### `del_object` and `set_object` need the right active image
Image slots are per image; always `set_image_idx(i)` first in batch sessions.

### `reset_pvs()` (image) clears only the active image; `reset_state()` clears all.

## Video

### Results appear only after `predict_*`
`add_points`/`add_box` run the model immediately but results are published to `get_results()` only via `predict_pvs()` (active frame) or `propagate()` (all frames).

### `predict_pvs()` runs no inference
It only parses stored output of the last prompt on the active frame. If nothing was stored it is a no-op.

### `predict_pcs()` resets the tracker
Prior keyframe prompts and manually-created objects are purged. Run it first.

### PCS ids depend on detections
IDs are sequential in concept order; the number of persons/dogs found determines which id is which. Always inspect before refining.

### `propagate()` replaces cached results
Everything previously in `get_results()` is overwritten by the propagated output. `del_object` after propagate edits cached results only; the tracker is told to remove the object, but you need to re-propagate for tracker-side consistency.

### `reset_state()` empties cached results but keeps the session
`get_results()` then returns `len(frames)` empty results.

### `initialise("video.mp4")` loads *all* frames
No stride/cap: memory-heavy. Use `load_video_frames(..., max_frames, stride)` and pass the list.

### Frame index ≠ source frame index when `stride > 1`
Returned list index `i` ↔ source frame `i*stride`. The engine has no timestamps.

### `video_version="sam3.1"`
README: multiplex model cannot do PVS and is buggy. Prefer `"sam3"`.

## Config

### `precision` scope
Applied to image inference only (autocast; `float32` = off). Video is hard-wired to bfloat16 by sam3 (`@torch.autocast` in `sam3_video_inference.py`, tracker `bf16_context`), so another value only warns. Changing precision may change scores/masks slightly versus the recorded tutorial outputs.

### Checkpoint paths must be files
Directories raise `FileNotFoundError` (message text mentions directories misleadingly).

### Threshold semantics
Image: strict `>`; applied at `predict_pcs` time (changing it afterwards doesn't alter existing objects). Video: sent with each request; takes effect for subsequent prompts/propagation.

### `enable_inst_interactivity=False`
Disables image PVS (`predict_inst`/interactive predictor absent). Keep `True` if you use PVS.

## Visualisation

- Colours are by position in the result, not by `object_id`.
- `show_image_with_grid` calls `plt.show()` itself and returns `None`; `visualise` returns an `Axes` (call `plt.show()` yourself in scripts).
- `visualise_batch` titles use the list index, not the original frame number.

## Environment

- `sam3` needs `pkg_resources` → `pip install "setuptools<81"`.
- Python 3.12+ (PEP 695 `type` statements).
- Warnings seen at startup (pynvml, `torch.jit.script`, `timm.models.layers`) are harmless.
