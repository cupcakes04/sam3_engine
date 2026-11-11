# Visualiser & Utilities

```python
from sam3_engine import Visualiser, load_video_frames, tensor_to_numpy, get_distinct_colors
```

All plotting uses matplotlib; call `plt.show()` yourself where noted (notebook usage).

## `Visualiser` (static/class methods; never instantiate)

### `Visualiser.visualise(result, image, alpha=0.5, show_boxes=True, show_labels=True, title=None, ax=None) -> plt.Axes`
Draws masks (colour per object), boxes and captions `ID:<id> | <label> | <score:.2f>`. Creates a `10×8` figure if `ax` is `None`. Default title `"Detections: N objects"`. Colours are assigned by object *position* in the result, not by `object_id`, so colours may shift between frames.

### `Visualiser.visualise_batch(results, images, cols=2, alpha=0.5, figsize_per_img=(7, 6)) -> plt.Figure`
Grid of `visualise` calls; cell titles are `"Frame / Image {i} ({n} objects)"` (index within the supplied lists, not original frame index). `ValueError` if `results` is empty. Unused cells are hidden. Zips `results` with `images` (shorter wins).

### `Visualiser.overlay(result, image, alpha=0.5) -> np.ndarray`
`(H, W, 3)` `uint8` array with masks blended; no boxes/labels.

### `Visualiser.overlay_masks(image, masks, alpha=0.5, colors=None) -> np.ndarray`
Lower-level: `image` is PIL or array (auto-scaled if max > 1), `colors` an `(N, 3)` float array in `[0,1]`.

### `Visualiser.show_image_with_grid(image_input, points=None, major_step=100, minor_step=20, figsize=(12, 10), title="", point_color="#FF3366", point_size=60) -> None`
Dark-theme image with labelled axes and graph-paper grid, so you can read **absolute pixel coordinates** for prompts. Optional `points=[(x, y), ...]` are plotted and annotated `P0: (x, y)`. Origin top-left; y axis inverted. Calls `plt.show()` itself and returns `None`.

Typical use: plot candidate prompt points first, check them, then pass the same numbers to `add_points` / `add_box`.

```python
Visualiser.show_image_with_grid(human, points=[(200, 0), (600, 400), (400, 200)])
```

## `load_video_frames(source, max_frames=None, stride=1) -> list[Image.Image]`

| `source` | Behaviour |
|---|---|
| video file path (`.mp4`, `.avi`, …) | OpenCV decode, BGR→RGB |
| directory | image files ending `.jpg/.jpeg/.png/.bmp`, **sorted lexicographically** by filename; `ValueError` if none |
| `list`/`tuple` of paths and/or PIL images | converted to RGB; non-existent path strings are silently skipped |

- `stride=k` keeps source indices `0, k, 2k, …` (index counted over the source, so keep indices in the returned list are `0..` consecutively).
- `max_frames` caps the number of **kept** frames.
- Errors: `FileNotFoundError` (path doesn't exist), `ValueError` (can't open video / empty dir), `TypeError` (unsupported source type).
- Note: with `stride=2`, returned frame `i` corresponds to source frame `2*i`. The engine has no notion of time; map back yourself if you need timestamps.

```python
frames = load_video_frames("./assets/videos/01_dog.mp4", max_frames=24, stride=2)  # 24 frames
```

## `tensor_to_numpy(x) -> np.ndarray`
Detaches, casts `bfloat16`/`float16` → `float32`, moves to CPU. Arrays pass through `np.asarray`.

## `get_distinct_colors(n) -> np.ndarray`
`(n, 3)` floats; `tab20` for `n ≤ 20`, else `gist_rainbow`. `n ≤ 0` → shape `(0, 3)`.
