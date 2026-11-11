# Guide: Image PVS (clicks & boxes)

Goal: segment **one specific object**, then refine it interactively.

## Preparation: read coordinates off the image

```python
human = Image.open("./assets/images/human.jpg").convert("RGB")
Visualiser.show_image_with_grid(human, points=[(200, 0), (600, 400), (400, 50), (400, 200), (450, 650)])
```

## Step 1 – first prompt

```python
engine.image.initialise([human])
engine.image.set_object(1)                                   # slot id 1 on image 0
engine.image.add_box(box=(200, 0, 600, 400))                 # (x1, y1, x2, y2)
engine.image.add_points(points=[(400, 200)], label=[1])      # 1 = foreground
engine.image.predict_pvs()
res1 = engine.image.get_results()
Visualiser.visualise(res1[0], human, title="PVS Step 1")
plt.show()
```

## Step 2 – refine with a negative click

```python
engine.image.set_object(1)                                   # same slot
engine.image.add_points(points=[(400, 50)], label=[0])       # 0 = background
engine.image.predict_pvs()                                   # uses previous mask logits + all points
res2 = engine.image.get_results()
```

## What happens under the hood

- Points **accumulate** on the slot; the **latest box** replaces earlier ones.
- The previous 288×288 mask logits are fed back, so each call refines.
- First prompt of a single click only (no box, no earlier mask) → multi-mask mode and the highest-IoU candidate is kept. Every other case → one mask.
- Result `box` is recomputed from the mask; `score` is the model's IoU estimate.
- Only *dirty* objects on the *active* image are recomputed.

## Several objects at once

```python
engine.image.set_object(2); engine.image.add_points(points=[(950, 1000)], label=[1])
engine.image.set_object(3); engine.image.add_box(box=(1300, 800, 1500, 950))
engine.image.predict_pvs()                  # decodes both 2 and 3 in one call
```

## Deleting

```python
engine.image.del_object(1)                  # active image only; no error if absent
```

## Restarting a single object

There is no per-object clear. Options: `del_object(id)` then `set_object(id)` again (fresh slot, no logits), or `reset_pvs()` to wipe the active image's objects.

## Selecting the image

PVS commands target `active_idx` (default 0). Use `engine.image.set_image_idx(i)` first for multi-image sessions. See [batch-images](batch-images.md).

## Label naming

PVS-created slots are labelled `object_<id>`. PCS-created slots keep their concept label even after refinement.
