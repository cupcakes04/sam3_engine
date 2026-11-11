# Architecture & Mental Model

## Components

```
SAM3Engine  (core.py)            ← the only thing you construct
 ├─ .image  → SAM3ImageEngine    (image.py)  PCS + PVS on 1..N images
 ├─ .video  → SAM3VideoEngine    (video.py)  PCS + PVS + propagation on a frame list
 ├─ .encode_concept(text)        text → CachedConcept (via image engine)
 ├─ .set_threshold(t)
 └─ .shutdown()
Visualiser / load_video_frames / helpers   (vis.py, video.py)
```

`engine.image` and `engine.video` are **lazy properties**: the first access builds the model with `sam3.model_builder` (`build_sam3_image_model` / `build_sam3_predictor`) and caches the wrapper. They are independent sessions; initialising one does not affect the other.

## PCS vs PVS

| | PCS (concept) | PVS (visual) |
|---|---|---|
| Prompt | text phrase and/or exemplar boxes | clicks (pos/neg) and/or a box |
| Output | **all** instances matching the concept | **one** object per object slot |
| Entry points | `add_concept`, `add_exemplar`, `predict_pcs` | `set_object`, `add_points`, `add_box`, `predict_pvs` |
| Object IDs | assigned automatically `1, 2, 3, …` | you choose via `set_object(id)` |
| "Stuff" classes | works (e.g. `"trash pile"` → 1 region) | n/a |

Both write into the same **object slots**, so they interoperate (see [pcs-to-pvs](../guides/pcs-to-pvs.md)).

## Feature caching ("extract once, query many")

- `initialise(images|frames)` runs the heavy ViT backbone once and keeps features on the GPU (image engine) or in the predictor session (video engine).
- Clicks, boxes and concepts afterwards only run light decoders.
- `encode_concept("x")` produces text embeddings independent of any image/video. Reuse the returned `CachedConcept` freely.
  - Image engine: the cached tensors are actually consumed.
  - Video engine: only the embedded text string (`"__text__"`) is used; the SAM 3 video predictor re-encodes text itself. Passing a `CachedConcept` is accepted for API symmetry, not speed.

## Command–Query Separation

| Kind | Methods | Returns |
|---|---|---|
| Commands | `initialise`, `add_concept`, `add_exemplar`, `set_object`, `set_image_idx`, `set_frame_idx`, `add_points`, `add_box`, `predict_pcs`, `predict_pvs`, `del_object`, `reset_*`, `set_threshold`, `shutdown` | `None` |
| Queries | `get_results()` | `list[Result]`, pure read, **no inference** |
| Query+compute | `video.propagate()` | `list[Result]`, length == number of frames |
| Factory | `encode_concept` | `CachedConcept` |

`predict_*` never return results. Always call `get_results()`.

## Object slots

- Image engine: `objects[image_idx][obj_id] → ObjectState` (mask, box, score, 288×288 low-res logits, accumulated points / point labels / boxes, dirty flag, label).
- Video engine: slots live inside the SAM 3 tracker session; the wrapper keeps `object_labels[obj_id] → str` and `cached_results`.
- `get_results()` only emits slots that **have a mask**. A slot created by `set_object()` without a successful `predict_pvs()` is invisible.

## Dirty tracking (image PVS)

`add_points` / `add_box` mark the active object dirty. `predict_pvs()` runs the decoder **only for dirty objects of the active image**, then clears the flag. Stage several objects, call `predict_pvs()` once.

## Refinement uses the previous mask

After any prediction the object stores 288×288 mask logits. Subsequent PVS calls feed them back as `mask_input`, so each extra click *refines* instead of restarting. This also applies to objects born from PCS (their logits are saved), which is what makes PCS→PVS refinement work.

## PCS → tracker bridge (video)

SAM 3 detects text concepts with a semantic slot (`obj_id = 0`). `video.predict_pcs()` therefore:

1. Runs each text concept and the exemplar group on the active frame, collecting masks/boxes.
2. Assigns sequential IDs `1, 2, …` (concept insertion order, then exemplars).
3. **Resets the tracker session** (purges the temporary detector artefacts / ghost tracks).
4. Re-seeds each detection as an independent tracker object using its box (corner labels 2 and 3).
5. Marks the frame as prompted.

## Propagation anchoring

`propagate()` sets `start_frame_index = min(prompted_frames)` and passes no explicit direction, so the SAM 3 predictor's default applies. In the tutorial run (24 frames, keyframe 3) the progress bars show 21 frames then 3 frames, i.e. forward from the anchor and backward to frame 0. The return value is always `len(frames)` entries; frames without detections have empty lists.

## Confidence threshold

- Image: detections kept when `score > threshold` (strictly greater). Score = `sigmoid(logit) × sigmoid(presence_logit)` when the model provides a presence head.
- Video: threshold is sent as `output_prob_thresh` **with every request** (`add_prompt`, `propagate`). Changing it later does not retroactively alter already-computed frames.
- `SAM3Engine.set_threshold` updates both sub-engines (those already built; later-built ones read the engine value on creation).
