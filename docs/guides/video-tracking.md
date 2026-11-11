# Guide: Video tracking

All video work happens on an in-memory `list[PIL.Image]`. The result of `propagate()` is a `list[Result]` with exactly one entry per frame.

## 1. Load & initialise

```python
frames = load_video_frames("./assets/videos/01_dog.mp4", max_frames=24, stride=2)
print(f"Loaded {len(frames)} frames from video.")      # Loaded 24 frames from video.
engine.video.initialise(frames)
tgt_idx = 3
```

Inspect the keyframe to choose prompts:

```python
Visualiser.show_image_with_grid(frames[tgt_idx], points=[(395, 450), (200, 200), (450, 600), (300, 300), (450, 350), (750, 550), (600, 400)])
```

## 2a. Keyframe PCS

```python
engine.video.set_frame_idx(tgt_idx)
person = engine.encode_concept("person")
dog    = engine.encode_concept("dog")

engine.video.reset_state()
engine.video.add_concept(person)
engine.video.add_concept(dog)
engine.video.predict_pcs()                         # runs now on frame 3, seeds tracker
frame_results = engine.video.get_results()         # len == 24; only frame 3 populated so far

Visualiser.visualise(frame_results[tgt_idx], frames[tgt_idx], title="Keyframe 3 PCS")
plt.show()
```

IDs follow concept insertion order: person detections first, then dog detections. **Print `frame_results[tgt_idx]["object_ids"]` and `["labels"]` before prompting refinements.**

Refine/extend on the same frame (tutorial: add a ball to an existing object by clicking it):

```python
engine.video.set_object(3)                          # an id taken from the printout above
engine.video.add_points(points=[(400, 450)], label=[1]) # model runs immediately; clicks accumulate
engine.video.predict_pvs()                          # publish to get_results()
frame_results = engine.video.get_results()
```

## 2b. Keyframe PVS (SAM2-style, no text)

```python
engine.video.reset_state()

engine.video.set_object(1)
engine.video.add_points(points=[(600, 400)], label=[1])        # dog: foreground click
engine.video.predict_pvs()

engine.video.set_object(2)
engine.video.add_box(box=(200, 200, 450, 600))               # person: box
engine.video.predict_pvs()

frame_results = engine.video.get_results()
```

Optional naming: `engine.video.set_object(1, label="dog")`.

You may prompt additional frames (`set_frame_idx(k)` then the same calls); propagation starts at the earliest prompted frame.

## 3. Propagate

```python
video_results = engine.video.propagate()           # optional: propagate(max_frames=N)
print(len(video_results))                          # 24

idx = [0, len(frames)//3, 2*len(frames)//3, len(frames)-1]
Visualiser.visualise_batch(results=[video_results[i] for i in idx],
                           images=[frames[i] for i in idx], cols=2)
plt.show()
```

Recorded: `Successfully propagated tracking across 24 frames!` with two progress bars (`0/21`, `0/3`): 21 frames after keyframe 3 and 3 before it.

Reading per-object tracks:

```python
track = {}                                        # obj_id -> {frame_idx: mask}
for f, r in enumerate(video_results):
    for oid, m in zip(r["object_ids"], r["masks"]):
        track.setdefault(oid, {})[f] = m
```

## Editing

- Remove an object everywhere: `engine.video.del_object(oid)` (also removes it from already-propagated results; re-propagate if you want the tracker to stop following it).
- Start over: `engine.video.reset_state()` (keeps frames), or `engine.video.initialise(new_frames)`.

## Order matters

`predict_pcs()` resets the tracker → do PCS **before** PVS clicks/boxes, and PVS refinements **before** `propagate()`.

## Cleanup

```python
engine.shutdown()      # closes the video session, empties CUDA cache
```
