# `SAM3VideoEngine` (`engine.video`)

Obtain via `engine.video`. Operates on an in-memory `list[PIL.Image]`; no FPS or timestamps exist in the API. **Frame index = position in the list.**

## Quick reference

| Method | Returns | Needs session | Summary |
|---|---|---|---|
| `initialise(video)` | `None` | – | Start tracker session on frames |
| `close_session()` | `None` | no | Close and drop everything |
| `set_threshold(t)` | `None` | no | Set `output_prob_thresh` for subsequent requests |
| `set_frame_idx(i)` | `None` | frames present (IndexError) | Select keyframe |
| `set_object(obj_id, label="")` | `None` | no | Select object id (+ optional label) |
| `add_points(points, label)` | `None` | **yes** | Click prompts on active frame/object (lists, equal length; runs model now) |
| `add_box(box)` | `None` | **yes** | Box prompt on active frame/object (runs model now) |
| `add_concept(concept, name=None)` | `None` | **yes** | Stage text concept |
| `add_exemplar(box, label=1)` | `None` | **yes** | Stage exemplar box(es) |
| `predict_pcs()` | `None` | **yes** | Detect staged concepts on the active frame, seed tracker |
| `predict_pvs()` | `None` | **yes** | Refresh cached result of the active frame from last PVS output |
| `del_object(obj_id)` | `None` | **yes** | Remove object from tracker & all cached results |
| `propagate(max_frames=None)` | `list[Result]` | **yes** | Track across the video |
| `get_results()` | `list[Result]` | no | Snapshot of cached results |
| `reset_state()` / `reset_pvs()` | `None` | no | Reset tracker + prompts + results |
| `reset_pcs()` | `None` | no | Clear staged concepts only |

---

## `initialise(video)`

```python
initialise(video: list[Image.Image] | str) -> None
```
- `list`/`tuple` of PIL images (or arrays → `Image.fromarray`); converted to RGB. Empty → `ValueError`. Other types → `TypeError`.
- `str` → `load_video_frames(path)` with **no** `max_frames` / `stride` (every frame). For subsampling call `load_video_frames` yourself and pass the list.
- Closes any previous session first, resets all state (`active_frame_idx=0`, `active_obj_id=1`), creates `cached_results` = `len(frames)` empty `Result`s, then sends `start_session` with the frame list.
- All frames are expected to share one resolution.

## `set_frame_idx(frame_idx)`
`IndexError` if outside `[0, len(frames)-1]`. All subsequent prompt methods target this frame.

## `set_object(obj_id, label="")`
Sets `active_obj_id`; if `label` is non-empty stores it in `object_labels[obj_id]` (shown in `Result.labels`). Does not create anything in the tracker until a prompt is added.

## `add_points(points, label)`

`points=[(x, y), ...]` and `label=[1|0, ...]` are lists of equal length (`ValueError` otherwise). No scalar/single-tuple shorthand and no default label.
Normalises coordinates to `[0,1]` (clamped), sends an `add_prompt` for `(active_frame_idx, active_obj_id)` with `clear_old_points=False` → **points accumulate**, refining the same object. Marks the frame prompted and stores the returned frame outputs. The model runs immediately; call `predict_pvs()` to publish into `get_results()`.

## `add_box(box)`
Same as `add_points` but sends each box as two corner points with labels `2` (top-left) and `3` (bottom-right). Clamped to `[0,1]`.

## `add_concept(concept, name=None)` / `add_exemplar(box, label=1)` / `reset_pcs()`
Stage only; no inference. `add_concept` accepts `str`, `CachedConcept`, or a list; only the text string is kept. For a `str`, key = `name or concept`. For a `CachedConcept`, text = `"__text__"` (or `"name"`), key = `name or "__text__"`. Different keys with the same text are evaluated separately.

## `predict_pcs()`
Raises `RuntimeError` if nothing is staged. On the active frame:

1. Each text concept is sent as its own `add_prompt`; every returned instance gets a new sequential id starting at **1** and label = concept key.
2. If exemplars are staged, all are sent in one prompt; instances labelled `"exemplar"` get the next ids.
3. If anything was detected: the tracker session is **reset**, then each detection is re-seeded as object `id` from its box, and the frame is marked prompted.
4. `cached_results[active_frame]` = the accumulated detections.

Previously prompted frames and manually created objects are discarded by the tracker reset (step 3). Run PCS first, then add PVS refinements.

## `predict_pvs()`
If the active frame has stored outputs from `add_points`/`add_box`/`predict_pcs`, parses them into `cached_results[active_frame]`. Otherwise no-op. It runs **no** model inference.

## `del_object(obj_id)`
Sends `remove_object` for the active frame, stores returned outputs (if any), removes the id from every frame in `cached_results`, and drops its label. Requires a session.

## `propagate(max_frames=None)`

```python
propagate(max_frames: int | None = None) -> list[Result]
```
- `RuntimeError` if the session is missing, or if **no** frame has been prompted.
- Starts at `min(prompted_frame_indices)`; `max_frames` is forwarded as `max_frame_num_to_track`; no direction is passed (predictor default; the tutorial run tracked both sides of the anchor).
- Streams per-frame outputs into a fresh list of `len(frames)` `Result`s, **replacing** `cached_results`, and returns `get_results()`.
- Frames that never receive output stay as empty `Result`s.
- Progress bars (tqdm) are printed by the underlying predictor.

## `get_results()`
Returns a copy (new outer and inner lists) of `cached_results`; length == `len(frames)`. Before `propagate`, only frames you ran `predict_pcs`/`predict_pvs` on (and `del_object` filtering) contain data; the rest are empty.

## `reset_state()` / `reset_pvs()`
`reset_session` request to the tracker (swallowing exceptions), `active_obj_id=1`, clears prompts/labels/prompted frames/last outputs, `cached_results` = empty list per frame. Frames and session remain.

## `close_session()`
Sends `close_session` (ignoring errors), clears frames, outputs, results, prompts, labels. `session_id=None`. Called by `initialise` and `SAM3Engine.shutdown()`.

## Attributes you may read

`video` (frames), `session_id`, `active_frame_idx`, `active_obj_id`, `cached_results`, `object_labels`, `prompted_frame_indices`, `pcs_text_concepts`, `pcs_exemplar_boxes`, `pcs_exemplar_labels`, `confidence_threshold`.
