# Guide: PCS → PVS transition

PCS populates object slots (`1, 2, 3, …`). Any slot can then be selected, refined or deleted with PVS commands, and new slots can be added.

## 1. Detect with PCS ("stuff" example)

```python
trash = Image.open("./assets/images/trash.jpg").convert("RGB")
Visualiser.show_image_with_grid(trash, points=[(100, 1500), (400, 1800), (400, 1700), (950, 1000), (1300, 800), (1500, 950)])

engine.image.initialise([trash])
trash_concept = engine.encode_concept("trash pile")
engine.image.add_concept(trash_concept)
engine.image.predict_pcs()                     # no arguments: runs the staged concepts
results_trans = engine.image.get_results()
print("Initial PCS Object IDs:", results_trans[0]["object_ids"])   # [1]
```

## 2. Refine slot 1 with PVS

```python
engine.image.set_image_idx(0)
engine.image.set_object(1)                                 # selects the PCS-created slot
engine.image.add_box(box=(100, 1500, 400, 1800))           # or add_points(points=[(400, 1700)], label=[1])
engine.image.predict_pvs()
results_refined = engine.image.get_results()
```

The slot keeps its label `"trash pile"`, its previous logits seed the refinement, and its box/score are overwritten from the new mask. (The tutorial author flags this as a demo only: the box narrows "trash pile" to a tyre.)

## 3. Add new objects

```python
engine.image.set_object(2)
engine.image.add_points(points=[(950, 1000)], label=[1])
engine.image.set_object(3)
engine.image.add_box(box=(1300, 800, 1500, 950))
engine.image.predict_pvs()
final_results = engine.image.get_results()
```

Pick new IDs that don't collide with existing PCS IDs (check `get_results()[0]["object_ids"]` first), otherwise you refine the existing object.

## 4. Delete

```python
engine.image.del_object(1)
final_res = engine.image.get_results()     # slot 1 gone; 2 and 3 remain
```

## Do not re-run `predict_pcs()` afterwards

`predict_pcs()` wipes all objects on all images, including PVS work. Order is always: **PCS → PVS → read**. If you need to redo PCS, you will redo your PVS edits.
