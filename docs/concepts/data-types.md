# Data Types

Defined in `sam3_engine/types.py`, all re-exported from `sam3_engine`.

```python
from sam3_engine import Box, Mask, Points, Label, Result, CachedConcept
```

| Name | Definition | Meaning |
|---|---|---|
| `Box` | `Tuple[float, float, float, float]` | `(x_min, y_min, x_max, y_max)`, absolute pixels |
| `Mask` | `np.ndarray` | 2-D `bool` array, shape `(H, W)` of the source image/frame |
| `Points` | `Tuple[float, float]` | `(x, y)`, absolute pixels (x = column, y = row) |
| `Label` | `Literal[0, 1]` | `1` positive / foreground / positive exemplar; `0` negative / background / negative exemplar |
| `CachedConcept` | `Dict[str, torch.Tensor]` | Output of `encode_concept` (plus a `"__text__"` string key holding the phrase) |

## `Result`

```python
class Result(TypedDict):
    masks:      list[Mask]     # bool (H, W)
    boxes:      list[Box]      # (x1, y1, x2, y2) absolute pixels, floats
    scores:     list[float]
    labels:     list[str]
    object_ids: list[int]
```

Invariants:

- All five lists have identical length N (one entry per object with a mask).
- Index `i` across the lists describes the same object.
- Image engine: ordered by ascending `object_id`.
- `get_results()` returns **one `Result` per image** (image engine) or **per frame** (video engine), in input order.
- Empty image/frame ⇒ five empty lists (never `None`).

### Field semantics

| Field | PCS | PVS (image) | Video |
|---|---|---|---|
| `scores[i]` | detection probability (× presence score if available) | SAM predicted IoU of the chosen mask | tracker `out_probs`, else `1.0` |
| `boxes[i]` | model box, scaled to pixels | recomputed from mask extent `(xs.min, ys.min, xs.max, ys.max)` | model's normalised xywh converted to xyxy pixels; else mask extent |
| `labels[i]` | the concept phrase (or `"exemplar"`) | `"object_<id>"` unless the slot kept a PCS label | concept phrase / `"exemplar"` / `label` given to `set_object` / `"object_<id>"` |
| `object_ids[i]` | `1, 2, …` in concept order | the id you chose | stable across frames (tracking identity) |

### Reading a Result

```python
r = results[0]
n = len(r["masks"])
for oid, lab, sc, bx, mk in zip(r["object_ids"], r["labels"], r["scores"], r["boxes"], r["masks"]):
    x1, y1, x2, y2 = bx
    area = int(mk.sum())          # pixels
```

## Notes on `type` statements

`Box`, `Mask`, `Points`, `Label`, `CachedConcept` are PEP 695 type aliases (`type X = ...`), usable in annotations only, not as runtime constructors.
