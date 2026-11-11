# State & Lifecycle

## Engine lifecycle

```
SAM3Engine(...)           validates 3 paths only
   ├─ engine.image        first access → builds image model (GPU memory)
   ├─ engine.video        first access → builds video predictor (GPU memory)
   ├─ engine.encode_concept(...)   → touches engine.image
   └─ engine.shutdown()   closes video session, torch.cuda.empty_cache()
```

`shutdown()` does **not** delete the loaded models; it only closes the active video session and empties the CUDA cache. Drop your references (`del engine`) to actually release weights.

## Image engine state

| State | Set by | Cleared by |
|---|---|---|
| `images`, cached backbone features | `initialise` | `initialise` (replace) |
| `active_idx` (image) | `initialise` → 0, `set_image_idx` | `initialise` |
| `active_obj_id` | `initialise` → 1, `set_object`, `reset_state` / `reset_pvs` → 1 | — |
| `objects[img][obj_id]` | `predict_pcs` (auto ids), `set_object`, `add_*`, `predict_pvs` | see table below |
| PCS text concepts / exemplars | `add_concept`, `add_exemplar` | `reset_pcs`, `reset_state`, `initialise` |

### What each image reset does

| Method | Objects | PCS concepts | Backbone cache | Scope |
|---|---|---|---|---|
| `initialise(imgs)` | all cleared (new dicts) | cleared | **recomputed** | whole engine |
| `reset_state()` | **all images** cleared | cleared | kept | whole engine |
| `reset_pcs()` | untouched | cleared | kept | whole engine |
| `reset_pvs()` | **active image** cleared, `active_obj_id=1` | untouched | kept | active image |
| `del_object(id)` | one slot on **active image** | untouched | kept | active image |

### PCS side effects

`predict_pcs()` **clears all objects on every image** and rebuilds them from the staged concepts. Any PVS-created objects or refinements are lost. Run PCS first, PVS afterwards. A new `predict_pcs()` always restarts ids at 1.

## Video engine state

| State | Set by | Cleared by |
|---|---|---|
| `video` frames, `session_id` | `initialise` | `close_session`, `initialise` |
| `active_frame_idx` | `initialise` → 0, `set_frame_idx` | `initialise` |
| `active_obj_id` | `initialise` → 1, `set_object`, `reset_state` → 1 | — |
| PCS text / exemplars | `add_concept`, `add_exemplar` | `reset_pcs`, `reset_state`, `close_session`, `initialise` |
| `object_labels` | `set_object(label=)`, `predict_pcs` | `del_object` (that id), `reset_state`, `close_session` |
| `prompted_frame_indices` | `add_points`, `add_box`, `predict_pcs` | `reset_state`, `close_session`, `initialise` |
| `cached_results` | `predict_pvs`, `predict_pcs`, `propagate`, `del_object` (filters) | `reset_state` (→ empty list per frame), `close_session` (→ `[]`) |

### What each video reset does

| Method | Tracker session | Prompts/labels | `cached_results` |
|---|---|---|---|
| `initialise(frames)` | closed, new session started | cleared | new empty |
| `reset_state()` | `reset_session` request (frames & session id kept) | cleared | emptied (len == frames) |
| `reset_pvs()` | identical to `reset_state()` | | |
| `reset_pcs()` | untouched | PCS staging only | untouched |
| `close_session()` | closed, `session_id=None` | cleared | `[]`, frames dropped |

## Required call order

| You want | Must have called first |
|---|---|
| `image.predict_pcs / predict_pvs` | `image.initialise` (else `RuntimeError`) |
| `image.set_object / add_points / add_box / del_object / set_image_idx` | `image.initialise` (else `KeyError`/`IndexError`; not guarded) |
| `video.add_*`, `predict_*`, `del_object`, `propagate` | `video.initialise` (else `RuntimeError`) |
| `video.propagate` | at least one of `predict_pcs`, `add_points`, `add_box` (else `RuntimeError`) |
| `video.predict_pcs` | at least one `add_concept`/`add_exemplar` (else `RuntimeError`) |

## Typical state machines

**Image**

```
initialise → [add_concept/add_exemplar → predict_pcs] → get_results
           → [set_image_idx → set_object → add_points/add_box → predict_pvs] → get_results
           → [del_object] → get_results
```

**Video**

```
initialise → set_frame_idx → [add_concept → predict_pcs | set_object → add_points/add_box → predict_pvs]
           → (optionally refine more frames) → propagate → list[Result] (len == frames)
```
