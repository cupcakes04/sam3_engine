# Guide: Image PCS (concepts & exemplars)

Goal: find **all instances** of something in one or more images.

## Steps

```python
engine.image.initialise([img])                 # 1. cache backbone features
giraffe = engine.encode_concept("giraffe")     # 2. (optional) pre-encode, reusable anywhere
zebra   = engine.encode_concept("zebra")
engine.image.add_concept(giraffe)              # 3. stage prompts
engine.image.add_concept(zebra)
engine.image.predict_pcs()                     # 4. run (returns None)
results = engine.image.get_results()           # 5. read
```

Recorded output (wildlife.jpg, threshold 0.35):

```
Detected 6 objects on wildlife image.
  [ID 1] giraffe (0.93) -> (726.1, 234.7, 1053.0, 483.0)
  [ID 2] giraffe (0.96) -> (251.1, 166.8, 734.2, 475.9)
  [ID 3] zebra (0.96) -> (964.2, 397.7, 1288.1, 652.0)
  [ID 4] zebra (0.97) -> (38.5, 400.3, 427.8, 637.4)
  [ID 5] zebra (0.96) -> (737.9, 447.7, 1008.1, 640.3)
  [ID 6] zebra (0.96) -> (419.5, 436.9, 771.3, 640.3)
```

IDs are assigned concept by concept in insertion order (giraffes first, then zebras), continuing the counter.

## Variants of staging concepts

```python
engine.image.add_concept("giraffe")                       # raw string (encoded now)
engine.image.add_concept(giraffe_cached)                  # CachedConcept from encode_concept
engine.image.add_concept(["giraffe", zebra_cached])       # list, mixed str / cached
engine.image.predict_pcs()                                # ALWAYS argument-free
```

`predict_pcs()` takes **no arguments** (like `predict_pvs()`): stage with `add_concept` / `add_exemplar`, then call it. With nothing staged it raises `RuntimeError`.

## Exemplar boxes

Condition on example boxes instead of words. `label=1` positive, `label=0` negative.

```python
engine.image.reset_state()                                  # drop previous text concepts/objects
engine.image.add_exemplar(box=(50, 400, 450, 650), label=1)   # a zebra  → "find things like this"
engine.image.add_exemplar(box=(250, 150, 750, 450), label=0)  # a giraffe → "not like this"
engine.image.predict_pcs()
res = engine.image.get_results()                            # all objects labelled "exemplar"
```

Recorded: `Detected 4 objects via exemplar grounding.` (the four zebras; giraffes excluded).

Rules:
- Exemplar-derived objects have label `"exemplar"`.
- The tutorial warns text concepts and exemplars should not be mixed (*"can't be used with image concepts above hybridly"*). The code will evaluate both if both are staged (text objects first, then exemplar objects), but the author-recommended practice is to use one mode per run; call `reset_state()` / `reset_pcs()` when switching.
- Staged exemplars apply to every image in the session.

## "Stuff" classes

Amorphous regions work as concepts, e.g. `"trash pile"` returns one region (`object_ids == [1]` in the tutorial).

## Tuning

- Too many false positives → `engine.set_threshold(0.5)` or higher, re-run `predict_pcs()`.
- Missing instances → lower the threshold (tutorial uses 0.35–0.4).
- Threshold is strict (`score > threshold`).

## Reusing a concept across images / video

`CachedConcept`s are image/video independent: encode once, add to `engine.image` and `engine.video` as needed.
