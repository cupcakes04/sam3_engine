# Prompts & Coordinates

## Coordinate system

- Origin `(0, 0)` = **top-left pixel** of the original image/frame.
- `x` increases right (width axis), `y` increases **down** (height axis).
- Every public method takes **absolute pixels in the original image resolution** (not the 1008×1008 model resolution, not normalised).
- Internally the engines convert (normalise, clamp to `[0, 1]` in the video engine) for you.
- Use `Visualiser.show_image_with_grid(image, points=[...])` to read coordinates off the picture first (see [visualiser](../api/visualiser.md)). The tutorial does this before every prompt.

## Exact accepted argument shapes

Boxes and exemplars: the engines inspect `x[0]` to decide "single item" vs "list of items": if it is an `int`/`float` it is a single item. **`np.float32` scalars are not `float` subclasses and will be misread as nested.** Convert with `float(...)`.

### Points

Strict, uniform form on **both** engines: two lists of equal length.

```python
engine.image.add_points(points=[(400, 200)], label=[1])                      # one click
engine.image.add_points(points=[(400, 200), (420, 220)], label=[1, 0])       # several
```

| Rule | Detail |
|---|---|
| `points` | `Sequence[(x, y)]` |
| `label` | `Sequence[0 | 1]`, one per point, **required** (no default) |
| Mismatch | `ValueError` |
| Not accepted | `add_points((x, y), 1)`, `add_points((x, y), [1])`, `add_points([(x, y)], 1)` |

### Boxes

| Call | Valid |
|---|---|
| single | `add_box((x1, y1, x2, y2))` |
| multiple | `add_box([(x1, y1, x2, y2), (x1, y1, x2, y2)])` |

Image engine caveat: boxes are accumulated per object, but **only the most recently added box is sent to the model** (`obj.boxes[-1]`). Points accumulate and are all used.

### Exemplars (PCS)

| Call | Valid |
|---|---|
| single | `add_exemplar((x1, y1, x2, y2), label=1)` (default label `1`) |
| multiple | `add_exemplar([b1, b2], [1, 0])` or `add_exemplar([b1, b2], 0)` (broadcast) |

`label=1` ⇒ "find more things like this". `label=0` ⇒ "not things like this".

### Text concepts

| Call | Valid |
|---|---|
| string | `add_concept("giraffe")` (encodes immediately on image engine) |
| cached | `add_concept(engine.encode_concept("giraffe"))` |
| list | `add_concept(["giraffe", cached_zebra])` |
| naming | `name=` is honoured **only for `CachedConcept` dicts**; for a `str` the phrase itself is the name. (In video it is honoured for strings too.) |

Concept names are dictionary keys: adding the same name twice **overwrites** rather than duplicates.

## How prompts are encoded internally (for reference)

| Prompt | Image engine | Video engine |
|---|---|---|
| Box (PVS) | `box=np.array([x1,y1,x2,y2])` to `predict_inst` (pixels, `normalize_coords=True`) | two points `[[nx1,ny1],[nx2,ny2]]`, labels `[2, 3]` (top-left, bottom-right corner), relative coords |
| Point | pixel coords + 0/1 labels | normalised `[nx, ny]` clamped to `[0,1]`, labels 0/1 |
| Exemplar box | normalised `cx, cy, w, h`, bool positive flag, appended to the geometric prompt | normalised `x, y, w, h`, `bounding_box_labels` 0/1 |
| Text | `forward_text` embeddings merged into the backbone dict | raw string in `add_prompt` |

## Choosing prompts

- Many instances of a nameable thing → text concept.
- Many instances, hard to name → positive exemplar box (add negative exemplars to exclude look-alikes).
- One specific object / fix a bad mask → PVS box and/or clicks.
- Whole-region "stuff" (`"trash pile"`) → text concept; refine with PVS.
