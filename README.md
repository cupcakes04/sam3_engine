# SAM 3 Engine (`sam3_engine`)

A standalone, first-principles Python engine built directly on Meta's native **SAM 3** and **SAM 3.1 Object Multiplex** architecture (`vendors/sam3`).

Interactive workflows and step-by-step demonstrations are available in [notebook](./workflow.ipynb).

---

## Setups

Please follow installation at sam3 repo [here](https://github.com/facebookresearch/sam3.git)

```bash
# 1. Clone the repository into your preferred folder (e.g. vendor/sam3 or ./sam3)
git clone https://github.com/facebookresearch/sam3.git

# 2. Enter and install in editable mode with notebook extras (required)
cd sam3
pip install -e ".[notebooks]"
```

> *Sam3 may install setuptools >= 81.0.0, which completely removed 'pkg_resources' (used in sam3 repo).*
> ```bash
> pip install "setuptools<81.0.0"
> ```

**Notes:**
- Checkpoint weights for sam3, sam3.1 is in their hugging face repo [here](https://huggingface.co/facebook/sam3)
- Bpe vocab file is at [here](./assets/bpe_simple_vocab_16e6.txt.gz)
- sam3.1 multiplex cannot do PVS and is fairly buggy (try to use sam3 only)

To install the sam3 engine in editable mode:
```bash
pip install -e .
```

---

## 1. Architectural Mental Model

The engine decouples open-vocabulary concept grounding (PCS) and promptable visual segmentation (PVS) while maintaining unified state management across images and videos.

### 1.1 Vision Backbone & Feature Caching
- **Extract Once, Query Many**: When `initialise()` is called (either with images or video frames), high-resolution ViT image embeddings are computed and cached in GPU memory once. Subsequent clicks, boxes, and concept prompts query against cached features without re-running the heavy vision backbone.
- **Concept Invariance**: Concepts encoded via `engine.encode_concept("person")` produce text embeddings that are completely independent of specific images or videos and can be reused freely.

### 1.2 Command-Query Separation (CQS)
- **Commands (`-> None`)**: Methods that mutate internal state or stage prompts (`set_object()`, `add_points()`, `add_box()`, `predict_pcs()`, `predict_pvs()`, `del_object()`, `reset_state()`) return `None`.
- **Query (`get_results() -> list[Result]`)**: A pure read-only accessor that formats and returns the active state across all image or video slots without triggering redundant GPU inference.
- **Propagation (`propagate() -> list[Result]`)**: The video propagation pipeline streams mask updates across all frames and returns a synchronized `list[Result]` whose length strictly matches `len(video)`.

### 1.3 Computational Efficiency & Zero-Redundancy Guarantees
- **Image PVS Dirty Tracking**: Staging multiple objects before calling `predict_pvs()` tracks which objects have uncomputed prompts via a dirty flag. The engine only runs the lightweight mask decoder for objects that have modified prompts.
- **Clean Multiplex-to-Tracker Bridging**: Meta SAM 3 uses a single semantic text detection slot (`obj_id = 0`). When grounding keyframes via PCS, detections are neatly transferred to independent SAM 2 instance tracking slots (`obj_id = 1, 2, ...`). The tracker session is reset prior to seeding to purge temporary text artifacts, preventing duplicate masks and ghost tracks.
- **Explicit Propagation Anchoring**: Rather than relying on fragile heuristics, video propagation is explicitly anchored to the earliest prompted keyframe (`start_frame_index = min(prompted_frames)`), allowing seamless bidirectional tracking across past and future frames.

---

## 2. Type Primitives

```python
from typing import Literal, Tuple, TypedDict
import numpy as np

type Box = Tuple[float, float, float, float]
"""Absolute pixel coordinates [x_min, y_min, x_max, y_max]."""

type Mask = np.ndarray
"""2D boolean NumPy array of shape (H, W)."""

type Points = Tuple[float, float]
"""Absolute pixel coordinates (x, y)."""

type Label = Literal[0, 1]
"""1 for positive (foreground click / positive exemplar), 0 for negative (background click / filter)."""

class Result(TypedDict):
    masks: list[Mask]
    boxes: list[Box]
    scores: list[float]
    labels: list[str]
    object_ids: list[int]
```

---

## 3. High-Level Engine Capabilities

### 3.1 Open-Vocabulary Concept Grounding (PCS)
- Detects arbitrary objects using natural language queries (`"giraffe"`, `"person"`, `"ball"`).
- Supports spatial exemplar bounding boxes for visual few-shot grounding.
- Automatically populates instance slots ready for interactive refinement.

### 3.2 Interactive Instance Segmentation (PVS)
- Click-based prompting: positive foreground clicks (`label=1`) and negative background clicks (`label=0`).
- Bounding box prompting: absolute bounding boxes `[x1, y1, x2, y2]`.
- Multi-object support: configure multiple objects independently or simultaneously on any keyframe.

### 3.3 State Management & Editing
- `del_object(obj_id)`: Removes a specific object from tracking and memory across all frames without wiping the video session.
- `reset_state()`: Clears active prompts, masks, and object slots while retaining extracted backbone video features.
- `get_results()`: Pure inspection query returning current predictions with human-readable concept names.

### 3.4 In-Memory Video Propagation
- Video tracking operates natively on `list[Image.Image]`, avoiding container video compression artifacts and framerate mismatches.
- Keyframe prompts on any frame (e.g. frame 0, frame 3, or mid-video) propagate bidirectionally to track objects through occlusions, motion, and scale variations.

