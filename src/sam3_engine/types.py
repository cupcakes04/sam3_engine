from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union
import numpy as np
import torch
from PIL import Image

# -------------------------------------------------------------------------
# Exchange Primitives
# -------------------------------------------------------------------------
type Box = Tuple[float, float, float, float]
"""Absolute pixel coordinates [x_min, y_min, x_max, y_max]."""

type Mask = np.ndarray
"""2D boolean NumPy array of shape (H, W)."""

type Points = Tuple[float, float]
"""Absolute pixel coordinates (x, y)."""

type Label = Literal[0, 1]
"""1 for positive (foreground click / positive exemplar), 0 for negative (background click / negative filter)."""


# -------------------------------------------------------------------------
# Result Structure (TypedDict)
# -------------------------------------------------------------------------
class Result(TypedDict):
    """
    Unified result for a single image or video frame.
    """
    masks: List[Mask]
    boxes: List[Box]
    scores: List[float]
    labels: List[str]
    object_ids: List[int]


# -------------------------------------------------------------------------
# Concept Structures
# -------------------------------------------------------------------------
type CachedConcept = Dict[str, torch.Tensor]
"""Precomputed concept text embeddings mapping from text backbone forward_text."""

class Concepts(TypedDict, total=False):
    """
    Optional container for active concepts to evaluate in PCS.
    """
    text: List[Union[str, CachedConcept]]
    exemplar: List[Box]
    exemplar_labels: List[Label]


