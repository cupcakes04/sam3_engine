# Full Usage Walkthrough

One continuous, runnable script mirroring `workflow.ipynb` (44 cells) in order, with the outputs recorded in the notebook. Run from the repository root so `./assets/...` resolves. Edit the three checkpoint paths.

Each section is a notebook cell group; blocks are meant to be executed sequentially in one Python session (state carries over).

## 0. Imports & engine

```python
import os, sys
from PIL import Image
import matplotlib.pyplot as plt

try:
    from sam3_engine import SAM3Engine, Visualiser, load_video_frames   # editable install
except ImportError:
    from src import SAM3Engine, Visualiser, load_video_frames           # run from repo root

engine = SAM3Engine(
    device="cuda",
    video_checkpoint_path="/home/nic/projects/rin/workspace/checkpoints/pretrained/sam3vlm/sam3.pt",
    image_checkpoint_path="/home/nic/projects/rin/workspace/checkpoints/pretrained/sam3vlm/sam3.pt",
    bpe_path="/home/nic/projects/rin/workspace/checkpoints/pretrained/sam3vlm/bpe_simple_vocab_16e6.txt.gz",
    video_version="sam3",
    precision="bfloat16",
    confidence_threshold=0.4,
)
print("SAM 3 Engine successfully initialized!")
```
Output: `SAM 3 Engine successfully initialized!` (plus harmless FutureWarnings from torch / timm / pkg_resources).

## 1. Dynamic threshold

```python
engine.set_threshold(0.35)
print(f"Active confidence threshold: {engine.confidence_threshold}")   # 0.35
```

## 2. Image PCS with concept caching (wildlife)

```python
wildlife = Image.open("./assets/images/wildlife.jpg").convert("RGB")

engine.image.initialise([wildlife])
text_giraffe = engine.encode_concept("giraffe")
text_zebra   = engine.encode_concept("zebra")
engine.image.add_concept(text_giraffe)
engine.image.add_concept(text_zebra)
engine.image.predict_pcs()
results = engine.image.get_results()

print(f"Detected {len(results[0]['masks'])} objects on wildlife image.")
for obj_id, label, score, box in zip(results[0]["object_ids"], results[0]["labels"],
                                     results[0]["scores"], results[0]["boxes"]):
    print(f"  [ID {obj_id}] {label} ({score:.2f}) -> {box}")

Visualiser.visualise(results[0], wildlife, title="Wildlife Concept Grounding (PCS)")
plt.show()
```
Output: 6 objects: IDs 1–2 giraffe (0.93, 0.96), IDs 3–6 zebra (0.96, 0.97, 0.96, 0.96).

## 2b. Exemplar grounding

```python
Visualiser.show_image_with_grid(wildlife, points=[(50, 400), (450, 650), (250, 150), (750, 450)])

engine.image.reset_state()                                        # drop text concepts & objects
engine.image.add_exemplar(box=(50, 400, 450, 650),  label=1)      # positive: a zebra
engine.image.add_exemplar(box=(250, 150, 750, 450), label=0)      # negative: a giraffe
engine.image.predict_pcs()
results_ex = engine.image.get_results()
print(f"Detected {len(results_ex[0]['masks'])} objects via exemplar grounding.")   # 4
Visualiser.visualise(results_ex[0], wildlife, title="Visual Exemplar Grounding")
```

## 3. Interactive PVS (human)

```python
human = Image.open("./assets/images/human.jpg").convert("RGB")
Visualiser.show_image_with_grid(human, points=[(200, 0), (600, 400), (400, 50), (400, 200), (450, 650)])

engine.image.initialise([human])

# Step 1: box + positive click on object slot 1
engine.image.set_object(1)
engine.image.add_box(box=(200, 0, 600, 400))
engine.image.add_points(points=[(400, 200)], label=[1])
engine.image.predict_pvs()
results_pvs1 = engine.image.get_results()
Visualiser.visualise(results_pvs1[0], human, title="PVS Step 1: Single Foreground Click"); plt.show()

# Step 2: refine with a negative click
engine.image.set_object(1)
engine.image.add_points(points=[(400, 50)], label=[0])
engine.image.predict_pvs()
results_pvs2 = engine.image.get_results()
Visualiser.visualise(results_pvs2[0], human, title="PVS Step 2: Refined with Negative Click"); plt.show()
```

## 4. PCS → PVS transition (trash)

```python
trash = Image.open("./assets/images/trash.jpg").convert("RGB")
Visualiser.show_image_with_grid(trash, points=[(100, 1500), (400, 1800), (400, 1700), (950, 1000), (1300, 800), (1500, 950)])

# 'stuff' segmentation
engine.image.initialise([trash])
trash_concept = engine.encode_concept("trash pile")
engine.image.add_concept(trash_concept)
engine.image.predict_pcs()
results_trans = engine.image.get_results()
print("Initial PCS Object IDs:", results_trans[0]["object_ids"])        # [1]
Visualiser.visualise(result=results_trans[0], image=trash); plt.show()

# refine PCS object 1 with a box (or add_points)
engine.image.set_image_idx(0)
engine.image.set_object(1)
engine.image.add_box(box=(100, 1500, 400, 1800))
engine.image.predict_pvs()
results_refined = engine.image.get_results()
Visualiser.visualise(results_refined[0], trash, title="PCS -> PVS Refinement"); plt.show()

# add two more objects
engine.image.set_image_idx(0)
engine.image.set_object(2); engine.image.add_points(points=[(950, 1000)], label=[1])
engine.image.set_object(3); engine.image.add_box(box=(1300, 800, 1500, 950))
engine.image.predict_pvs()
final_results = engine.image.get_results()
Visualiser.visualise(final_results[0], trash, title="After adding objects 2 and 3"); plt.show()

# delete object 1
engine.image.del_object(1)
final_res = engine.image.get_results()
Visualiser.visualise(final_res[0], trash, title="After deleting object 1"); plt.show()
```

## 5. Batch inference (tea)

```python
tea1 = Image.open("./assets/images/tea1.jpg").convert("RGB")
tea2 = Image.open("./assets/images/tea2.jpg").convert("RGB")

engine.image.initialise([tea1, tea2])
tea_concept     = engine.encode_concept("tea")
biscuit_concept = engine.encode_concept("biscuit")

engine.image.reset_state()
engine.image.add_concept(tea_concept)
engine.image.add_concept(biscuit_concept)
engine.image.predict_pcs()
results = engine.image.get_results()
Visualiser.visualise_batch(results=results, images=[tea1, tea2]); plt.show()

# PVS on image index 1 only
Visualiser.show_image_with_grid(tea2, points=[(100, 150), (300, 320)])
engine.image.set_image_idx(1)
engine.image.set_object(1)
engine.image.add_box(box=(100, 150, 300, 320))
engine.image.predict_pvs()
results_pvs = engine.image.get_results()
Visualiser.visualise_batch(results=results_pvs, images=[tea1, tea2]); plt.show()
```

## 6. Video tracking (dog)

```python
video_path = "./assets/videos/01_dog.mp4"
frames = load_video_frames(video_path, max_frames=24, stride=2)
print(f"Loaded {len(frames)} frames from video.")                 # 24
engine.video.initialise(frames)
tgt_idx = 3

Visualiser.show_image_with_grid(frames[tgt_idx], points=[(395, 450), (200, 200), (450, 600), (300, 300), (450, 350), (750, 550), (600, 400)])

# 6a. Keyframe PCS
engine.video.set_frame_idx(tgt_idx)
person_concept = engine.encode_concept("person")
dog_concept    = engine.encode_concept("dog")
engine.video.reset_state()
engine.video.add_concept(person_concept)
engine.video.add_concept(dog_concept)
engine.video.predict_pcs()
frame_results_pcs = engine.video.get_results()
Visualiser.visualise(result=frame_results_pcs[tgt_idx], image=frames[tgt_idx],
                     title=f"Keyframe {tgt_idx} PCS (Person & Dog)"); plt.show()

# 6b. Add a click to object 3 (ball joined to an existing object, per the notebook note)
engine.video.set_object(3)
engine.video.add_points(points=[(400, 450)], label=[1])
engine.video.predict_pvs()
frame_results_pvs = engine.video.get_results()
Visualiser.visualise(result=frame_results_pvs[tgt_idx], image=frames[tgt_idx],
                     title=f"Keyframe {tgt_idx} after PVS click"); plt.show()

# 6c. Propagate
video_results = engine.video.propagate()
print(f"Successfully propagated tracking across {len(video_results)} frames!")   # 24
sample = [0, len(frames)//3, 2*len(frames)//3, len(frames)-1]
Visualiser.visualise_batch(results=[video_results[i] for i in sample],
                           images=[frames[i] for i in sample], cols=2); plt.show()
```
Progress bars shown by the predictor: `0/21` and `0/3`.

```python
# 6d. Pure PVS keyframe prompting (SAM2-style) – start clean
engine.video.reset_state()

engine.video.set_object(1)
engine.video.add_points(points=[(600, 400)], label=[1])  # dog
engine.video.predict_pvs()

engine.video.set_object(2)
engine.video.add_box(box=(200, 200, 450, 600))          # person
engine.video.predict_pvs()

frame_results_pvs = engine.video.get_results()
Visualiser.visualise(result=frame_results_pvs[tgt_idx], image=frames[tgt_idx],
                     title=f"Keyframe {tgt_idx} PVS Prompting"); plt.show()

video_results = engine.video.propagate()
print(f"Successfully propagated tracking across {len(video_results)} frames!")   # 24
Visualiser.visualise_batch(results=[video_results[i] for i in sample],
                           images=[frames[i] for i in sample], cols=2); plt.show()
```

> After 6a the object IDs depend on what the detector found. In your own code, print `frame_results_pcs[tgt_idx]["object_ids"]` and `["labels"]` before choosing the id in `set_object(...)`.

## 7. Cleanup

```python
engine.shutdown()
print("Engine shut down and GPU memory freed.")
```
