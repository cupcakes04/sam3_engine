from __future__ import annotations

import contextlib
import copy
import functools
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.v2 as v2

from sam3_engine.types import Box, CachedConcept, Label, Mask, Points, Result
from sam3_engine.vis import tensor_to_numpy
from sam3.model.data_misc import FindStage, interpolate
from sam3.model import box_ops


def _to_pil_list(images: Union[Image.Image, Sequence[Image.Image]]) -> List[Image.Image]:
    """Ensures input is a list of PIL Images."""
    if isinstance(images, Image.Image):
        return [images.convert("RGB")]
    elif isinstance(images, (list, tuple)):
        if len(images) == 0:
            raise ValueError("images list cannot be empty.")
        return [img.convert("RGB") if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img)).convert("RGB") for img in images]
    raise TypeError(f"Expected PIL Image or Sequence[PIL.Image], got {type(images)}")


_PRECISION_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _with_precision(fn):
    """Runs the wrapped method under the engine's autocast precision context."""

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with self._precision_context():
            return fn(self, *args, **kwargs)

    return wrapper


class ObjectState:
    """Internal container for an object slot on a given image."""

    def __init__(self, obj_id: int, label: str = ""):
        self.obj_id = obj_id
        self.label = label or f"object_{obj_id}"
        self.mask: Optional[Mask] = None
        self.box: Optional[Box] = None
        self.score: float = 1.0
        self.logits: Optional[np.ndarray] = None  # (1, 288, 288) low-res mask logits for PVS
        self.points: List[Points] = []
        self.point_labels: List[Label] = []
        self.boxes: List[Box] = []
        self.dirty: bool = False

    def to_result_entry(self) -> Optional[Dict[str, Any]]:
        if self.mask is None:
            return None
        return {
            "mask": self.mask,
            "box": self.box if self.box is not None else (0.0, 0.0, 0.0, 0.0),
            "score": float(self.score),
            "label": str(self.label),
            "obj_id": int(self.obj_id),
        }


class SAM3ImageEngine:
    """
    Ground-zero native SAM 3 Image Engine supporting:
    - Multi-image Vision Backbone Feature Caching.
    - Concept Caching (Text and Geometric Exemplar prompts) for PCS.
    - Interactive Object Management and Progressive Refinement for PVS.
    - Seamless Bidirectional PCS <-> PVS Transition.
    """

    def __init__(
        self,
        model: Any,
        device: str = "cuda",
        resolution: int = 1008,
        confidence_threshold: float = 0.5,
        precision: str = "bfloat16",
    ):
        if precision not in _PRECISION_DTYPES:
            raise ValueError(f"precision must be one of {list(_PRECISION_DTYPES)}, got {precision!r}")
        self.model = model
        self.device = device
        self.precision = precision
        self.resolution = resolution
        self.confidence_threshold = confidence_threshold

        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(resolution, resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # Image state & cache
        self.images: List[Image.Image] = []
        self.active_idx: int = 0
        self.active_obj_id: int = 1
        self.backbone_out: Optional[Dict[str, Any]] = None
        # Mapping: img_idx -> {obj_id: ObjectState}
        self.objects: Dict[int, Dict[int, ObjectState]] = {}

        # Active PCS concept prompts (per image_idx, or broadcast)
        self.pcs_text_concepts: Dict[str, CachedConcept] = {}
        self.pcs_exemplar_boxes: List[Box] = []
        self.pcs_exemplar_labels: List[Label] = []

    def set_threshold(self, threshold: float) -> None:
        """Sets the dynamic confidence threshold for predictions."""
        self.confidence_threshold = float(threshold)

    def _precision_context(self):
        """torch.autocast for bfloat16/float16 on CUDA; no-op for float32 or CPU."""
        dtype = _PRECISION_DTYPES[self.precision]
        if dtype == torch.float32 or not str(self.device).startswith("cuda"):
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=dtype)

    # -------------------------------------------------------------------------
    # 2. Session Initialization & Feature Extraction
    # -------------------------------------------------------------------------
    @torch.inference_mode()
    @_with_precision
    def initialise(self, images: Union[Image.Image, List[Image.Image]]) -> None:
        """
        Initializes an image session with one or more PIL images.
        Extracts and caches ViT vision backbone features once in GPU memory.
        Resets any previous prompts, objects, and sessions.
        """
        self.images = _to_pil_list(images)
        self.active_idx = 0
        self.active_obj_id = 1
        self.objects = {i: {} for i in range(len(self.images))}
        self.pcs_text_concepts.clear()
        self.pcs_exemplar_boxes.clear()
        self.pcs_exemplar_labels.clear()

        # Transform and stack all images into (B, 3, 1008, 1008)
        processed_tensors = []
        for img in self.images:
            tensor = v2.functional.to_image(img).to(self.device)
            processed_tensors.append(self.transform(tensor))
        batch_tensor = torch.stack(processed_tensors, dim=0)

        # Precompute visual backbone features once
        self.backbone_out = self.model.backbone.forward_image(batch_tensor)

        # Setup SAM2 neck features for interactive instance interactivity (PVS)
        if (
            self.model.inst_interactive_predictor is not None
            and "sam2_backbone_out" in self.backbone_out
        ):
            sam2_out = self.backbone_out["sam2_backbone_out"]
            sam2_out["backbone_fpn"][0] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    sam2_out["backbone_fpn"][0]
                )
            )
            sam2_out["backbone_fpn"][1] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    sam2_out["backbone_fpn"][1]
                )
            )

    # -------------------------------------------------------------------------
    # 2.1 Concept Caching & PCS Prompting
    # -------------------------------------------------------------------------
    @torch.inference_mode()
    @_with_precision
    def encode_concept(self, text: str) -> CachedConcept:
        """
        Precomputes and returns raw text embedding dict from the model's text backbone.
        Image and video invariant: can be cached and reused anywhere.
        """
        res = self.model.backbone.forward_text([text], device=self.device)
        res["__text__"] = text
        return res

    def add_concept(
        self,
        concept: Union[str, CachedConcept, Sequence[Union[str, CachedConcept]]],
        name: Optional[str] = None,
    ) -> None:
        """
        Adds text concept prompt(s) (either raw string or precomputed CachedConcept dict)
        to the active PCS configuration.
        """
        if isinstance(concept, (list, tuple)):
            for c in concept:
                self.add_concept(c)
            return

        if isinstance(concept, str):
            c_name = concept
            c_tensor = self.encode_concept(concept)
        elif isinstance(concept, dict):
            c_name = name or concept.get("__text__") or f"concept_{len(self.pcs_text_concepts) + 1}"
            c_tensor = concept
        else:
            raise TypeError(f"Expected str or CachedConcept dict, got {type(concept)}")

        self.pcs_text_concepts[c_name] = c_tensor

    def add_exemplar(
        self,
        box: Union[Box, Sequence[Box]],
        label: Union[Label, Sequence[Label]] = 1,
    ) -> None:
        """
        Adds geometric exemplar box prompt(s) [x1, y1, x2, y2] to the active PCS configuration.
        """
        if isinstance(box[0], (int, float)):
            boxes = [box]
            labels = [label] if isinstance(label, (int, float)) else list(label)
        else:
            boxes = list(box)
            labels = list(label) if isinstance(label, (list, tuple)) else [label] * len(boxes)

        for b, l in zip(boxes, labels):
            self.pcs_exemplar_boxes.append(tuple(b))
            self.pcs_exemplar_labels.append(int(l))

    def reset_state(self) -> None:
        """
        Unified state reset across all objects, prompts, and concepts.
        Preserves cached visual backbone features so re-initialisation is unnecessary.
        """
        for img_idx in range(len(self.images)):
            self.objects[img_idx].clear()
        self.active_obj_id = 1
        self.pcs_text_concepts.clear()
        self.pcs_exemplar_boxes.clear()
        self.pcs_exemplar_labels.clear()

    def reset_pcs(self) -> None:
        """Resets all registered PCS text and exemplar concepts."""
        self.pcs_text_concepts.clear()
        self.pcs_exemplar_boxes.clear()
        self.pcs_exemplar_labels.clear()

    def get_results(self) -> List[Result]:
        """
        Returns the current prediction results across all images without running model inference.
        Pure read-only state accessor.
        """
        return self._assemble_all_results()

    # -------------------------------------------------------------------------
    # PCS Prediction
    # -------------------------------------------------------------------------
    @torch.inference_mode()
    @_with_precision
    def predict_pcs(self) -> None:
        """
        Evaluates the staged concepts (add_concept / add_exemplar) across all initialized
        images and updates internal object states. Takes no arguments (mirrors predict_pvs):
        stage prompts first, then call this.
        Clears all existing objects on every image before running.
        Populates internal object state so detections can be inspected via get_results()
        or interactively refined with PVS.
        """
        if self.backbone_out is None or len(self.images) == 0:
            raise RuntimeError("Engine not initialised. Call engine.image.initialise(images) first.")
        if len(self.pcs_text_concepts) == 0 and len(self.pcs_exemplar_boxes) == 0:
            raise RuntimeError("No concepts registered. Call add_concept() or add_exemplar() first.")

        # Process each image in the batch
        for img_idx, img in enumerate(self.images):
            img_w, img_h = img.size
            # Clear previous objects on this image for fresh PCS run
            self.objects[img_idx].clear()
            next_obj_id = 1

            # 1. Evaluate Text Concepts
            for concept_name, text_outputs in self.pcs_text_concepts.items():
                grounding_backbone = dict(self.backbone_out)
                grounding_backbone.update(text_outputs)

                find_stage = FindStage(
                    img_ids=torch.tensor([img_idx], device=self.device, dtype=torch.long),
                    text_ids=torch.tensor([0], device=self.device, dtype=torch.long),
                    input_boxes=None,
                    input_boxes_mask=None,
                    input_boxes_label=None,
                    input_points=None,
                    input_points_mask=None,
                )

                outputs = self.model.forward_grounding(
                    backbone_out=grounding_backbone,
                    find_input=find_stage,
                    geometric_prompt=self.model._get_dummy_prompt(),
                    find_target=None,
                )

                next_obj_id = self._consume_grounding_outputs(
                    outputs=outputs,
                    img_idx=img_idx,
                    img_w=img_w,
                    img_h=img_h,
                    label_name=concept_name,
                    start_obj_id=next_obj_id,
                )

            # 2. Evaluate Exemplar Concepts
            if len(self.pcs_exemplar_boxes) > 0:
                grounding_backbone = dict(self.backbone_out)
                if "language_features" not in grounding_backbone:
                    dummy_text = self.model.backbone.forward_text(["visual"], device=self.device)
                    grounding_backbone.update(dummy_text)

                find_stage = FindStage(
                    img_ids=torch.tensor([img_idx], device=self.device, dtype=torch.long),
                    text_ids=torch.tensor([0], device=self.device, dtype=torch.long),
                    input_boxes=None,
                    input_boxes_mask=None,
                    input_boxes_label=None,
                    input_points=None,
                    input_points_mask=None,
                )

                geo_prompt = self.model._get_dummy_prompt()
                for b, l in zip(self.pcs_exemplar_boxes, self.pcs_exemplar_labels):
                    x1, y1, x2, y2 = b[:4]
                    cx = ((x1 + x2) / 2.0) / img_w
                    cy = ((y1 + y2) / 2.0) / img_h
                    bw = max(x2 - x1, 0.0) / img_w
                    bh = max(y2 - y1, 0.0) / img_h
                    is_pos = bool(l == 1)

                    box_tensor = torch.tensor([[cx, cy, bw, bh]], device=self.device, dtype=torch.float32).view(1, 1, 4)
                    lbl_tensor = torch.tensor([[is_pos]], device=self.device, dtype=torch.bool).view(1, 1)
                    geo_prompt.append_boxes(box_tensor, lbl_tensor)

                outputs = self.model.forward_grounding(
                    backbone_out=grounding_backbone,
                    find_input=find_stage,
                    geometric_prompt=geo_prompt,
                    find_target=None,
                )

                next_obj_id = self._consume_grounding_outputs(
                    outputs=outputs,
                    img_idx=img_idx,
                    img_w=img_w,
                    img_h=img_h,
                    label_name="exemplar",
                    start_obj_id=next_obj_id,
                )

    def _consume_grounding_outputs(
        self,
        outputs: Dict[str, torch.Tensor],
        img_idx: int,
        img_w: int,
        img_h: int,
        label_name: str,
        start_obj_id: int,
    ) -> int:
        """Parses model grounding tensors, filters by threshold, and populates ObjectState."""
        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        out_probs = out_logits.sigmoid()

        if "presence_logit_dec" in outputs:
            presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
            out_probs = (out_probs * presence_score).squeeze(-1)
        else:
            out_probs = out_probs.squeeze(-1)

        keep = out_probs > self.confidence_threshold
        if not keep.any():
            return start_obj_id

        filtered_probs = out_probs[keep]
        filtered_masks = out_masks[keep]
        filtered_bbox = out_bbox[keep]

        # Convert normalized cxcywh to absolute pixel xyxy
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(filtered_bbox)
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h], device=self.device)
        boxes_xyxy = boxes_xyxy * scale_fct[None, :]

        # Interpolate full resolution masks
        interp_masks = interpolate(
            filtered_masks.unsqueeze(1),
            (img_h, img_w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()

        # Interpolate low resolution mask logits (288, 288) for PVS feedback
        low_res_logits = interpolate(
            filtered_masks.unsqueeze(1),
            (288, 288),
            mode="bilinear",
            align_corners=False,
        )

        probs_np = tensor_to_numpy(filtered_probs)
        boxes_np = tensor_to_numpy(boxes_xyxy)
        masks_np = tensor_to_numpy(interp_masks > 0.5)
        logits_np = tensor_to_numpy(low_res_logits)

        current_id = start_obj_id
        for i in range(len(probs_np)):
            mask_2d = np.squeeze(masks_np[i])
            if mask_2d.ndim != 2 or not mask_2d.any():
                continue

            obj = ObjectState(obj_id=current_id, label=label_name)
            obj.mask = mask_2d.astype(bool)
            obj.box = (
                float(boxes_np[i, 0]),
                float(boxes_np[i, 1]),
                float(boxes_np[i, 2]),
                float(boxes_np[i, 3]),
            )
            obj.score = float(probs_np[i])
            obj.logits = logits_np[i : i + 1]  # (1, 288, 288)
            self.objects[img_idx][current_id] = obj
            current_id += 1

        return current_id

    # -------------------------------------------------------------------------
    # 2.2 Image PVS (Interactive Segmentation)
    # -------------------------------------------------------------------------
    def set_image_idx(self, idx: int) -> None:
        """Selects the active image index for interactive PVS work."""
        if idx < 0 or idx >= len(self.images):
            raise IndexError(f"Image index {idx} out of range [0, {len(self.images) - 1}]")
        self.active_idx = idx

    def set_object(self, obj_id: int) -> None:
        """Selects or creates an object slot on the currently active image."""
        self.active_obj_id = int(obj_id)
        if self.active_obj_id not in self.objects[self.active_idx]:
            self.objects[self.active_idx][self.active_obj_id] = ObjectState(
                obj_id=self.active_obj_id, label=f"object_{self.active_obj_id}"
            )

    def add_box(self, box: Union[Box, Sequence[Box]]) -> None:
        """Adds bounding box prompt(s) in absolute pixel coordinates [x1, y1, x2, y2]."""
        self._ensure_active_object()
        obj = self.objects[self.active_idx][self.active_obj_id]
        if isinstance(box[0], (int, float)):
            obj.boxes.append(tuple(box))
        else:
            for b in box:
                obj.boxes.append(tuple(b))
        obj.dirty = True

    def add_points(
        self,
        points: Sequence[Points],
        label: Sequence[Label],
    ) -> None:
        """
        Adds point prompts in absolute pixel coordinates.
        `points` is a list of (x, y); `label` is a list of 1 (positive) / 0 (negative)
        with exactly one label per point.
        """
        if len(points) != len(label):
            raise ValueError(
                f"points and label must have equal length, got {len(points)} points and {len(label)} labels."
            )
        self._ensure_active_object()
        obj = self.objects[self.active_idx][self.active_obj_id]
        for p, l in zip(points, label):
            obj.points.append((float(p[0]), float(p[1])))
            obj.point_labels.append(int(l))
        obj.dirty = True

    def del_object(self, obj_id: int) -> None:
        """Deletes an object slot on the currently active image."""
        self.objects[self.active_idx].pop(int(obj_id), None)

    def reset_pvs(self) -> None:
        """Resets all PVS object slots and prompts on the active image."""
        self.objects[self.active_idx].clear()
        self.active_obj_id = 1

    @torch.inference_mode()
    @_with_precision
    def predict_pvs(self) -> None:
        """
        Runs interactive instance prediction (PVS) for any object with uncomputed prompts on the active image.
        Uses accumulated points, boxes, and previous mask logits for refinement.
        Updates internal object states; call get_results() to inspect the updated predictions.
        """
        if self.backbone_out is None or len(self.images) == 0:
            raise RuntimeError("Engine not initialised. Call engine.image.initialise(images) first.")

        for obj_id, obj in sorted(self.objects[self.active_idx].items()):
            if obj.dirty and (len(obj.points) > 0 or len(obj.boxes) > 0):
                self._run_pvs_inference_for_object(obj)
                obj.dirty = False

    def _run_pvs_inference_for_object(self, obj: ObjectState) -> None:
        """Executes model.predict_inst for a single object slot."""
        img = self.images[self.active_idx]
        orig_w, orig_h = img.size

        # Prepare single image backbone features for SAM2 interactive predictor
        sam2_out = self.backbone_out.get("sam2_backbone_out")
        if sam2_out is not None:
            sliced_sam2 = {
                "backbone_fpn": [x[self.active_idx : self.active_idx + 1] for x in sam2_out["backbone_fpn"]],
                "vision_pos_enc": [x[self.active_idx : self.active_idx + 1] for x in sam2_out["vision_pos_enc"]],
            }
            single_backbone = {"sam2_backbone_out": sliced_sam2}
        else:
            single_backbone = {}

        single_state = {
            "original_height": orig_h,
            "original_width": orig_w,
            "backbone_out": single_backbone,
        }

        pts_arr = np.array(obj.points, dtype=np.float32) if len(obj.points) > 0 else None
        lbls_arr = np.array(obj.point_labels, dtype=np.int32) if len(obj.point_labels) > 0 else None
        box_arr = np.array(obj.boxes[-1], dtype=np.float32) if len(obj.boxes) > 0 else None
        mask_input = obj.logits

        # For ambiguous prompts (single click without prior mask/box), SAM requires
        # multimask_output=True so the 3-token head disambiguates object/part levels.
        # For non-ambiguous prompts (subsequent clicks with mask_input or boxes), multimask_output=False.
        use_multimask = (mask_input is None and box_arr is None and pts_arr is not None and len(pts_arr) <= 1)

        masks_np, ious_np, low_res_np = self.model.predict_inst(
            single_state,
            point_coords=pts_arr,
            point_labels=lbls_arr,
            box=box_arr,
            mask_input=mask_input,
            multimask_output=use_multimask,
            return_logits=False,
            normalize_coords=True,
        )

        if use_multimask and masks_np.ndim >= 3 and masks_np.shape[0] > 1:
            best_idx = int(np.argmax(ious_np))
            mask_2d = masks_np[best_idx] > 0.5
            best_score = float(ious_np[best_idx])
            best_logits = low_res_np[best_idx : best_idx + 1]
        else:
            mask_2d = np.squeeze(masks_np) > 0.5
            best_score = float(np.squeeze(ious_np))
            best_logits = low_res_np if low_res_np.ndim == 3 else low_res_np[None, ...]

        if mask_2d.ndim == 2 and mask_2d.any():
            obj.mask = mask_2d
            obj.score = best_score
            obj.logits = best_logits  # (1, 288, 288)
            # Recompute bounding box from refined mask
            ys, xs = np.where(mask_2d)
            obj.box = (
                float(xs.min()),
                float(ys.min()),
                float(xs.max()),
                float(ys.max()),
            )

    def _ensure_active_object(self) -> None:
        if self.active_obj_id not in self.objects[self.active_idx]:
            self.set_object(self.active_obj_id)

    def _assemble_all_results(self) -> List[Result]:
        """Assembles list[Result] for all images matching input index order."""
        results: List[Result] = []
        for img_idx in range(len(self.images)):
            masks: List[Mask] = []
            boxes: List[Box] = []
            scores: List[float] = []
            labels: List[str] = []
            object_ids: List[int] = []

            for obj_id, obj in sorted(self.objects[img_idx].items()):
                entry = obj.to_result_entry()
                if entry is not None:
                    masks.append(entry["mask"])
                    boxes.append(entry["box"])
                    scores.append(entry["score"])
                    labels.append(entry["label"])
                    object_ids.append(entry["obj_id"])

            results.append(
                {
                    "masks": masks,
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels,
                    "object_ids": object_ids,
                }
            )
        return results
