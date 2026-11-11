# Guide: Batch images

One `initialise([...])` call can hold many images. Backbone features for all are computed together and cached.

## PCS across the batch

```python
tea1 = Image.open("./assets/images/tea1.jpg").convert("RGB")
tea2 = Image.open("./assets/images/tea2.jpg").convert("RGB")

engine.image.initialise([tea1, tea2])
tea     = engine.encode_concept("tea")
biscuit = engine.encode_concept("biscuit")

engine.image.reset_state()                 # (optional here) clear anything staged
engine.image.add_concept(tea)
engine.image.add_concept(biscuit)
engine.image.predict_pcs()                 # runs on BOTH images
results = engine.image.get_results()       # len == 2; results[0] ↔ tea1, results[1] ↔ tea2

Visualiser.visualise_batch(results=results, images=[tea1, tea2])
plt.show()
```

Rules:
- `results[i]` corresponds to `images[i]` in the order given to `initialise`.
- Staged concepts/exemplars apply to every image.
- Each image gets its own independent object-id numbering starting at 1.

## PVS on one image of the batch

PVS commands act on the **active image** only.

```python
Visualiser.show_image_with_grid(tea2, points=[(100, 150), (300, 320)])

engine.image.set_image_idx(1)                      # target tea2
engine.image.set_object(1)                         # slot 1 on image 1
engine.image.add_box(box=(100, 150, 300, 320))     # whole cup (cup ≠ tea) → coarser mask in the demo
engine.image.predict_pvs()                         # only image 1 is touched
results_pvs = engine.image.get_results()           # still len == 2

Visualiser.visualise_batch(results=results_pvs, images=[tea1, tea2])
plt.show()
```

`results_pvs[0]` is unchanged from the PCS run; `results_pvs[1]` now has slot 1 replaced by the box-prompted mask (slot 1 existed from PCS, so it is refined, keeping label e.g. `"tea"`).

## Memory note

All images' features sit in GPU memory simultaneously. Large batches of high-resolution images can OOM; split into several sessions if needed (each `initialise` replaces the previous session).
