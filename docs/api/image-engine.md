# `SAM3ImageEngine` (`engine.image`)

Obtain via `engine.image`. Do not construct directly. Handles one session of 1..N images.

Constructor internals: `precision` (`"bfloat16"` default, set by `SAM3Engine`), `resolution=1008` (images are resized to 1008×1008, normalised mean/std 0.5), `confidence_threshold`.

**Precision:** `initialise`, `encode_concept`, `predict_pcs` and `predict_pvs` run inside `torch.autocast("cuda", dtype)` for `"bfloat16"` / `"float16"`. `"float32"` (or a non-CUDA device) means no autocast. Invalid values raise `ValueError`.

## Quick reference

| Method | Returns | Needs `initialise` | Summary |
|---|---|---|---|
| `initialise(images)` | `None` | – | Start session, cache features, reset all state |
| `set_threshold(t)` | `None` | no | Set detection threshold |
| `encode_concept(text)` | `CachedConcept` | no | Text → embeddings |
| `add_concept(concept, name=None)` | `None` | no | Stage text prompt(s) |
| `add_exemplar(box, label=1)` | `None` | no | Stage exemplar box(es) |
| `predict_pcs()` | `None` | **yes** | Run staged concepts on all images |
| `set_image_idx(idx)` | `None` | yes (IndexError) | Select active image |
| `set_object(obj_id)` | `None` | yes | Select/create slot on active image |
| `add_points(points, label)` | `None` | yes | Stage clicks on active object (lists, equal length) |
| `add_box(box)` | `None` | yes | Stage box on active object |
| `predict_pvs()` | `None` | **yes** | Decode dirty objects on active image |
| `del_object(obj_id)` | `None` | yes | Remove slot on active image |
| `reset_state()` | `None` | no | Clear all objects/prompts/concepts, all images |
| `reset_pcs()` | `None` | no | Clear staged concepts/exemplars |
| `reset_pvs()` | `None` | yes | Clear objects on active image |
| `get_results()` | `list[Result]` | no | Read-only snapshot |

---

## `initialise(images)`

```python
initialise(images: Image.Image | list[Image.Image]) -> None
```
- Accepts a single PIL image, or a list/tuple. Non-PIL list items are converted with `Image.fromarray(np.asarray(x))`. All are `.convert("RGB")`.
- Raises `ValueError` for an empty list, `TypeError` for other types.
- Sets `active_idx=0`, `active_obj_id=1`, empties objects for each image, clears staged PCS concepts/exemplars.
- Runs the ViT backbone on the stacked batch `(B, 3, 1008, 1008)` and caches features on the device. Also prepares the SAM2 neck (`conv_s0`/`conv_s1`) used by PVS.
- Calling again **replaces the session entirely**.

## `encode_concept(text)`
Runs the text backbone for `[text]`; returns the output dict plus `"__text__": text`. Wrapped in `torch.inference_mode`.

## `add_concept(concept, name=None)`

```python
add_concept(
    concept: str | CachedConcept | Sequence[str | CachedConcept],
    name: str | None = None,
) -> None
```
- `str` → encoded immediately; key = the string (`name` ignored).
- `CachedConcept` → key = `name` or its `"__text__"` or `"concept_<n>"`.
- list/tuple → each element added (the `name` argument is **not** propagated to elements).
- Other types → `TypeError`.
- Same key twice overwrites.
- Does not run detection.

## `add_exemplar(box, label=1)`
See [prompts-and-coordinates](../concepts/prompts-and-coordinates.md). Appends to `pcs_exemplar_boxes` / `pcs_exemplar_labels` (labels stored as `int`). Staged exemplars apply to **every** image in the batch.

## `predict_pcs()`

```python
predict_pcs() -> None
```

Takes no arguments. Behaviour:

1. `RuntimeError` if not initialised, or if no concept/exemplar is staged.
2. For each image: **clears that image's objects**, then
   - one grounding pass per staged text concept (label = concept key),
   - one grounding pass for all staged exemplars together (label = `"exemplar"`), using a dummy `"visual"` text prompt if the backbone has no language features.
3. Detections with probability `> confidence_threshold` and non-empty masks become object slots with ids `1, 2, …` (continuing across concepts on the same image).

Resulting slot: full-res bool mask (sigmoid > 0.5 after bilinear upsample), pixel `xyxy` box, score, labelled, with 288×288 logits saved for PVS refinement. Runs under the engine's `precision` autocast.

## `set_image_idx(idx)`
`IndexError` if `idx` outside `[0, len(images)-1]`. Does not change `active_obj_id`.

## `set_object(obj_id)`
Sets `active_obj_id = int(obj_id)`; if the slot doesn't exist on the **active image**, creates an empty `ObjectState` labelled `object_<id>`. For an existing slot (e.g. from PCS) it just selects it, preserving label, mask and logits.

## `add_points(points, label)`

```python
add_points(points: Sequence[Points], label: Sequence[Label]) -> None
```
Both arguments are **lists of equal length**: `points=[(x, y), ...]`, `label=[1|0, ...]`. No single-tuple or scalar-label shorthand. `ValueError` if the lengths differ. Appends to the active object (auto-creating the slot if missing) and marks it dirty.

## `add_box(box)`
Appends box(es) to the active object and marks dirty. Only the last box is used at prediction.

## `predict_pvs()`

For each object on the **active image** (ascending id) that is dirty **and** has ≥1 point or box:

- Prompt = all accumulated points + last box + previous mask logits (`mask_input`), `normalize_coords=True`.
- `multimask_output=True` only for the very first ambiguous prompt: no logits yet, no box, 1 point. The best of the candidate masks (max predicted IoU) is kept. Otherwise a single mask.
- If the resulting mask is non-empty: store mask (bool), score = predicted IoU, new low-res logits, and **recompute `box` from the mask extent**.
- If the mask is empty the previous mask is kept unchanged.
- Dirty flag cleared.
- Raises `RuntimeError` if not initialised. Objects on other images are not touched.

## `del_object(obj_id)`
Removes the slot on the **active image**; silently does nothing if absent.

## `reset_state()`, `reset_pcs()`, `reset_pvs()`
See [state-and-lifecycle](../concepts/state-and-lifecycle.md).

## `get_results()`
Returns `list[Result]` of length `len(images)` (empty list before `initialise`). Includes only slots with a mask, ascending `object_id`. A freshly selected, never-predicted slot is omitted. No inference. Masks are the stored arrays (not copies).

## Attributes you may read

`images`, `active_idx`, `active_obj_id`, `objects` (`dict[img][id] → ObjectState`), `pcs_text_concepts`, `pcs_exemplar_boxes`, `pcs_exemplar_labels`, `confidence_threshold`, `resolution`.

`ObjectState` fields: `obj_id, label, mask, box, score, logits, points, point_labels, boxes, dirty`.
