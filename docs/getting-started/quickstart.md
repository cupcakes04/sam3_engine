# Quickstart

Two minimal scripts. Both are complete and runnable after [installation](installation.md). Adjust the three paths.

## Image: find everything matching a phrase

```python
from PIL import Image
import matplotlib.pyplot as plt
from sam3_engine import SAM3Engine, Visualiser

engine = SAM3Engine(
    image_checkpoint_path="/path/to/sam3.pt",
    video_checkpoint_path="/path/to/sam3.pt",
    bpe_path="/path/to/bpe_simple_vocab_16e6.txt.gz",
    device="cuda",
    confidence_threshold=0.4,
)

img = Image.open("assets/images/wildlife.jpg").convert("RGB")

engine.image.initialise([img])            # 1. cache vision features (list of images)
engine.image.add_concept("giraffe")       # 2. stage text prompts
engine.image.add_concept("zebra")
engine.image.predict_pcs()                # 3. run detection (returns None)
results = engine.image.get_results()      # 4. read (returns list[Result], one per image)

r = results[0]
for oid, label, score, box in zip(r["object_ids"], r["labels"], r["scores"], r["boxes"]):
    print(oid, label, round(score, 2), box)   # box = (x1, y1, x2, y2) absolute pixels

Visualiser.visualise(r, img, title="Wildlife")
plt.show()

engine.shutdown()
```

## Video: prompt one frame, track everywhere

```python
from sam3_engine import SAM3Engine, load_video_frames, Visualiser
import matplotlib.pyplot as plt

# engine = SAM3Engine(...)  as above

frames = load_video_frames("assets/videos/01_dog.mp4", max_frames=24, stride=2)  # list[PIL.Image]
engine.video.initialise(frames)

engine.video.set_frame_idx(3)             # choose keyframe
engine.video.add_concept("dog")
engine.video.predict_pcs()                # detect on frame 3, seed tracker
video_results = engine.video.propagate()  # list[Result], len == len(frames)

Visualiser.visualise(video_results[10], frames[10])
plt.show()

engine.shutdown()
```

## The five rules to remember

1. **Commands return `None`; `get_results()` / `propagate()` return `list[Result]`.**
2. **`initialise()` first.** It resets everything for that sub-engine.
3. **All coordinates are absolute pixels, origin top-left, `(x, y)` for points and `(x1, y1, x2, y2)` for boxes.**
4. **`Result` index = image index (image engine) or frame index (video engine).**
5. **Object IDs are per-image / per-video integers starting at 1; they are what you pass to `set_object` / `del_object`.**

Continue with [architecture](../concepts/architecture.md), or jump to the [full walkthrough](../guides/full-walkthrough.md).
