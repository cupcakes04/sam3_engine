# sam3_engine Documentation

`sam3_engine` is a thin, stateful Python facade over Meta's native **SAM 3** (`sam3` package) for:

- **PCS** – Promptable *Concept* Segmentation (text / exemplar-box prompts → *all* matching instances)
- **PVS** – Promptable *Visual* Segmentation (click / box prompts → one specific instance, refinable)
- **Video tracking** – prompt on any frame(s), then propagate through an in-memory `list[PIL.Image]`

Everything returns one uniform structure: `list[Result]`.

> Python **3.12+** is required (the package uses the `type X = ...` statement).
> Documentation was written against source version `0.1.0` and the `workflow.ipynb` tutorial.

## Map

| Folder | File | What it answers |
|---|---|---|
| `getting-started/` | [installation.md](getting-started/installation.md) | Install, checkpoints, BPE vocab, env pitfalls |
| | [quickstart.md](getting-started/quickstart.md) | Smallest working image + video script |
| `concepts/` | [architecture.md](concepts/architecture.md) | Mental model: caching, slots, CQS, PCS↔PVS bridge |
| | [data-types.md](concepts/data-types.md) | `Box`, `Mask`, `Points`, `Label`, `Result`, `CachedConcept` |
| | [prompts-and-coordinates.md](concepts/prompts-and-coordinates.md) | Coordinate system and **exact accepted argument shapes** |
| | [state-and-lifecycle.md](concepts/state-and-lifecycle.md) | What each method mutates / resets; call-order rules |
| `api/` | [engine.md](api/engine.md) | `SAM3Engine` (constructor, `image`, `video`, `encode_concept`, `shutdown`) |
| | [image-engine.md](api/image-engine.md) | `SAM3ImageEngine` – every method |
| | [video-engine.md](api/video-engine.md) | `SAM3VideoEngine` – every method |
| | [visualiser.md](api/visualiser.md) | `Visualiser`, `load_video_frames`, `tensor_to_numpy`, `get_distinct_colors` |
| `guides/` | [image-pcs.md](guides/image-pcs.md) | Text concepts, concept caching, exemplar boxes |
| | [image-pvs.md](guides/image-pvs.md) | Clicks, boxes, refinement, adding/deleting objects |
| | [pcs-to-pvs.md](guides/pcs-to-pvs.md) | Detect with PCS, then refine/delete with PVS |
| | [batch-images.md](guides/batch-images.md) | Multi-image sessions and per-image PVS |
| | [video-tracking.md](guides/video-tracking.md) | Keyframe PCS/PVS and `propagate()` |
| | [full-walkthrough.md](guides/full-walkthrough.md) | **Complete end-to-end example**, cell-by-cell mirror of `workflow.ipynb` with recorded outputs |
| `reference/` | [gotchas.md](reference/gotchas.md) | Every non-obvious behaviour, quirk and error, verified against source |
| | [errors.md](reference/errors.md) | Exceptions raised, by method |

## Reading order

1. `getting-started/installation.md` → `quickstart.md`
2. `concepts/*` (15 minutes; removes almost all ambiguity)
3. The guide matching your task
4. `api/*` as reference, `reference/gotchas.md` before shipping anything
