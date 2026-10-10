from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Union
import cv2
import numpy as np
from PIL import Image
import torch

from sam3_engine.types import Box, CachedConcept, Concepts, Label, Mask, Points, Result
from sam3_engine.vis import tensor_to_numpy


def load_video_frames(
    source: Union[str, Sequence[Union[str, Image.Image]]],
    max_frames: Optional[int] = None,
    stride: int = 1,
) -> List[Image.Image]:
    """
    Utility to load video frames into a list of PIL Images from:
    - A video file (.mp4, .avi, etc.)
    - A directory of image files
    - A list of image filepaths or PIL Images
    """
    if isinstance(source, (list, tuple)):
        frames: List[Image.Image] = []
        for i, item in enumerate(source):
            if i % stride != 0:
                continue
            if max_frames is not None and len(frames) >= max_frames:
                break
            if isinstance(item, Image.Image):
                frames.append(item.convert("RGB"))
            elif isinstance(item, str) and os.path.exists(item):
                frames.append(Image.open(item).convert("RGB"))
        return frames

    if not isinstance(source, str):
        raise TypeError(f"Unsupported source type: {type(source)}")

    if os.path.isdir(source):
        files = sorted(
            [
                os.path.join(source, f)
                for f in os.listdir(source)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ]
        )
        if not files:
            raise ValueError(f"No image files found in directory: {source}")
        frames = []
        for i, f in enumerate(files):
            if i % stride != 0:
                continue
            if max_frames is not None and len(frames) >= max_frames:
                break
            frames.append(Image.open(f).convert("RGB"))
        return frames

    elif os.path.isfile(source):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {source}")

        frames = []
        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % stride == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(rgb))
                    if max_frames is not None and len(frames) >= max_frames:
                        break
                frame_idx += 1
        finally:
            cap.release()
        return frames
    else:
        raise FileNotFoundError(f"Video source does not exist: {source}")


class SAM3VideoEngine:
    """
    Ground-zero native SAM 3 / SAM 3.1 Video Engine.
    - Accepts raw PIL images directly (list[Image.Image]), eliminating FPS/timestamp ambiguity.
    - Supports Keyframe PCS & PVS Prompting.
    - Propagates bidirectional tracking across the video.
    - Strictly returns list[Result] matching the input video length 1-to-1.
    """

    def __init__(
        self,
        predictor: Any,
        confidence_threshold: float = 0.5,
    ):
        self.predictor = predictor
        self.confidence_threshold = confidence_threshold

        self.video: List[Image.Image] = []
        self.session_id: Optional[str] = None
        self.active_frame_idx: int = 0
        self.active_obj_id: int = 1
        self.last_frame_outputs: Dict[int, Dict[str, Any]] = {}
        self.cached_results: List[Result] = []

        self.pcs_text_concepts: Dict[str, str] = {}
        self.pcs_exemplar_boxes: List[Box] = []
        self.pcs_exemplar_labels: List[Label] = []
        self.object_labels: Dict[int, str] = {}
        self.prompted_frame_indices: Set[int] = set()

    def set_threshold(self, threshold: float) -> None:
        """Sets the dynamic confidence threshold for predictions."""
        self.confidence_threshold = float(threshold)

    # -------------------------------------------------------------------------
    # 3. Session Initialization
    # -------------------------------------------------------------------------
    def initialise(self, video: Union[List[Image.Image], str]) -> None:
        """
        Initializes a video tracking session with a list of PIL Images.
        Deletes any previous video session.
        """
        self.close_session()

        if isinstance(video, str):
            self.video = load_video_frames(video)
        elif isinstance(video, (list, tuple)):
            if len(video) == 0:
                raise ValueError("Video frames list cannot be empty.")
            self.video = [
                img.convert("RGB") if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img)).convert("RGB")
                for img in video
            ]
        else:
            raise TypeError(f"Expected list[Image.Image] or video file path, got {type(video)}")

        self.active_frame_idx = 0
        self.active_obj_id = 1
        self.last_frame_outputs = {}
        self.cached_results = self._create_empty_results()
        self.pcs_text_concepts.clear()
        self.pcs_exemplar_boxes.clear()
        self.pcs_exemplar_labels.clear()
        self.object_labels.clear()
        self.prompted_frame_indices.clear()

        # Start session natively passing the list of PIL Images
        response = self.predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": self.video,
            }
        )
        self.session_id = response["session_id"]

    def close_session(self) -> None:
        """Closes and cleans up the active video session."""
        if self.session_id is not None:
            try:
                self.predictor.handle_request(
                    {
                        "type": "close_session",
                        "session_id": self.session_id,
                    }
                )
            except Exception:
                pass
            self.session_id = None
        self.video = []
        self.last_frame_outputs = {}
        self.cached_results = []
        self.pcs_text_concepts.clear()
        self.pcs_exemplar_boxes.clear()
        self.pcs_exemplar_labels.clear()
        self.object_labels.clear()
        self.prompted_frame_indices.clear()

    # -------------------------------------------------------------------------
    # 3.1 Keyframe PCS & PVS Prompting
    # -------------------------------------------------------------------------
    def set_frame_idx(self, frame_idx: int) -> None:
        """Selects the active keyframe index for prompting."""
        if frame_idx < 0 or frame_idx >= len(self.video):
            raise IndexError(f"Frame index {frame_idx} out of bounds [0, {len(self.video) - 1}]")
        self.active_frame_idx = frame_idx

    def set_object(self, obj_id: int, label: str = "") -> None:
        """Sets the target object ID for keyframe prompts with optional label."""
        self.active_obj_id = int(obj_id)
        if label:
            self.object_labels[self.active_obj_id] = label

    def add_box(self, box: Union[Box, Sequence[Box]]) -> None:
        """
        Adds bounding box prompt(s) in absolute pixel coordinates [x1, y1, x2, y2]
        to the active object on the active keyframe.
        """
        self._ensure_session()
        if isinstance(box[0], (int, float)):
            boxes = [box]
        else:
            boxes = list(box)

        img_w, img_h = self.video[self.active_frame_idx].size
        norm_coords = []
        point_labels = []
        for b in boxes:
            x1, y1, x2, y2 = b[:4]
            nx1 = max(0.0, min(1.0, float(x1) / img_w))
            ny1 = max(0.0, min(1.0, float(y1) / img_h))
            nx2 = max(0.0, min(1.0, float(x2) / img_w))
            ny2 = max(0.0, min(1.0, float(y2) / img_h))
            # SAM prompt encoder encodes boxes as top-left corner (label 2) and bottom-right corner (label 3)
            norm_coords.extend([[nx1, ny1], [nx2, ny2]])
            point_labels.extend([2, 3])

        response = self.predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": self.session_id,
                "frame_index": self.active_frame_idx,
                "obj_id": self.active_obj_id,
                "points": norm_coords,
                "point_labels": point_labels,
                "clear_old_points": False,
                "rel_coordinates": True,
                "output_prob_thresh": self.confidence_threshold,
            }
        )
        self.prompted_frame_indices.add(self.active_frame_idx)
        if response and "outputs" in response:
            self.last_frame_outputs[self.active_frame_idx] = response["outputs"]

    def add_points(
        self,
        points: Union[Points, Sequence[Points]],
        label: Union[Label, Sequence[Label]] = 1,
    ) -> None:
        """
        Adds point prompt(s) in absolute pixel coordinates (x, y) with label (1=pos, 0=neg)
        to the active object on the active keyframe.
        """
        self._ensure_session()
        if isinstance(points[0], (int, float)):
            pts_list = [points]
            lbls_list = [label]
        else:
            pts_list = list(points)
            lbls_list = list(label) if isinstance(label, (list, tuple)) else [label] * len(pts_list)

        img_w, img_h = self.video[self.active_frame_idx].size
        norm_coords = [
            [
                max(0.0, min(1.0, float(p[0]) / img_w)),
                max(0.0, min(1.0, float(p[1]) / img_h)),
            ]
            for p in pts_list
        ]
        lbls = [int(l) for l in lbls_list]

        response = self.predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": self.session_id,
                "frame_index": self.active_frame_idx,
                "obj_id": self.active_obj_id,
                "points": norm_coords,
                "point_labels": lbls,
                "clear_old_points": False,
                "rel_coordinates": True,
                "output_prob_thresh": self.confidence_threshold,
            }
        )
        self.prompted_frame_indices.add(self.active_frame_idx)
        if response and "outputs" in response:
            self.last_frame_outputs[self.active_frame_idx] = response["outputs"]

    def del_object(self, obj_id: int) -> None:
        """Removes an object from tracking on the active keyframe."""
        self._ensure_session()
        response = self.predictor.handle_request(
            {
                "type": "remove_object",
                "session_id": self.session_id,
                "frame_index": self.active_frame_idx,
                "obj_id": int(obj_id),
            }
        )
        if response and "outputs" in response:
            self.last_frame_outputs[self.active_frame_idx] = response["outputs"]
        elif self.active_frame_idx in self.last_frame_outputs:
            self.last_frame_outputs.pop(self.active_frame_idx, None)

        # Filter out obj_id from cached_results across all frames
        for r in self.cached_results:
            if obj_id in r["object_ids"]:
                indices_to_keep = [i for i, oid in enumerate(r["object_ids"]) if oid != obj_id]
                r["masks"] = [r["masks"][i] for i in indices_to_keep]
                r["boxes"] = [r["boxes"][i] for i in indices_to_keep]
                r["scores"] = [r["scores"][i] for i in indices_to_keep]
                r["labels"] = [r["labels"][i] for i in indices_to_keep]
                r["object_ids"] = [r["object_ids"][i] for i in indices_to_keep]

        self.object_labels.pop(int(obj_id), None)

    def reset_state(self) -> None:
        """
        Unified state reset: resets tracker states, object slots, prompts, and cached outputs.
        Preserves video frames and active session so re-initialisation is unnecessary.
        """
        if self.session_id is not None:
            try:
                self.predictor.handle_request(
                    {
                        "type": "reset_session",
                        "session_id": self.session_id,
                    }
                )
            except Exception:
                pass
        self.active_obj_id = 1
        self.last_frame_outputs.clear()
        self.pcs_text_concepts.clear()
        self.pcs_exemplar_boxes.clear()
        self.pcs_exemplar_labels.clear()
        self.object_labels.clear()
        self.prompted_frame_indices.clear()
        self.cached_results = self._create_empty_results()

    def reset_pvs(self) -> None:
        """Resets active PVS object slots and tracker states."""
        self.reset_state()

    def get_results(self) -> List[Result]:
        """
        Returns the current prediction results across all video frames without running model inference.
        Pure read-only state accessor.
        """
        return [
            {
                "masks": list(r["masks"]),
                "boxes": list(r["boxes"]),
                "scores": list(r["scores"]),
                "labels": list(r["labels"]),
                "object_ids": list(r["object_ids"]),
            }
            for r in self.cached_results
        ]

    def add_concept(
        self,
        concept: Union[str, CachedConcept, Sequence[Union[str, CachedConcept]]],
        name: Optional[str] = None,
    ) -> None:
        """
        Adds open-vocabulary text concept prompt to the active PCS configuration.
        Does not evaluate model inference until predict_pcs() is called.
        """
        self._ensure_session()
        if isinstance(concept, (list, tuple)):
            for c in concept:
                self.add_concept(c)
            return

        if isinstance(concept, str):
            text_str = concept
            c_name = name or concept
        elif isinstance(concept, dict):
            text_str = str(concept.get("__text__") or concept.get("name", "concept"))
            c_name = name or str(concept.get("__text__") or f"concept_{len(self.pcs_text_concepts) + 1}")
        else:
            raise TypeError(f"Expected str or CachedConcept dict, got {type(concept)}")

        self.pcs_text_concepts[c_name] = text_str

    def add_exemplar(
        self,
        box: Union[Box, Sequence[Box]],
        label: Union[Label, Sequence[Label]] = 1,
    ) -> None:
        """
        Adds geometric exemplar bounding box(es) [x1, y1, x2, y2] in absolute pixel coordinates.
        Does not evaluate model inference until predict_pcs() is called.
        """
        self._ensure_session()
        if isinstance(box[0], (int, float)):
            boxes = [box]
            labels = [label] if isinstance(label, (int, float)) else list(label)
        else:
            boxes = list(box)
            labels = list(label) if isinstance(label, (list, tuple)) else [label] * len(boxes)

        for b, l in zip(boxes, labels):
            self.pcs_exemplar_boxes.append(tuple(b))
            self.pcs_exemplar_labels.append(int(l))

    def reset_pcs(self) -> None:
        """Resets active PCS text and exemplar concepts."""
        self.pcs_text_concepts.clear()
        self.pcs_exemplar_boxes.clear()
        self.pcs_exemplar_labels.clear()

    def predict_pvs(self) -> None:
        """
        Evaluates Promptable Visual Segmentation (PVS) on the active keyframe.
        Updates internal frame outputs; call get_results() to inspect the predictions.
        """
        self._ensure_session()
        if self.active_frame_idx in self.last_frame_outputs:
            self.cached_results[self.active_frame_idx] = self._parse_frame_outputs(
                self.last_frame_outputs[self.active_frame_idx],
                frame_idx=self.active_frame_idx,
            )

    def predict_pcs(self) -> None:
        """
        Evaluates active PCS text and exemplar concepts on the active keyframe.
        Each concept is evaluated with its own prompt, accurately labeled, and
        registered into the tracker session.
        Updates internal frame outputs; call get_results() to inspect the predictions.
        """
        self._ensure_session()
        if len(self.pcs_text_concepts) == 0 and len(self.pcs_exemplar_boxes) == 0:
            raise RuntimeError("No concepts registered. Call add_concept() or add_exemplar() first.")

        img_w, img_h = self.video[self.active_frame_idx].size
        accumulated_masks: List[Mask] = []
        accumulated_boxes: List[Box] = []
        accumulated_scores: List[float] = []
        accumulated_labels: List[str] = []
        accumulated_obj_ids: List[int] = []
        current_obj_id = 1

        # 1. Evaluate Text Concepts individually to ensure accurate detections & labels
        for concept_name, text_str in self.pcs_text_concepts.items():
            response = self.predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": self.session_id,
                    "frame_index": self.active_frame_idx,
                    "text": text_str,
                    "output_prob_thresh": self.confidence_threshold,
                }
            )
            if response and "outputs" in response:
                res = self._parse_frame_outputs(response["outputs"], frame_idx=self.active_frame_idx)
                for m, b, s in zip(res["masks"], res["boxes"], res["scores"]):
                    accumulated_masks.append(m)
                    accumulated_boxes.append(b)
                    accumulated_scores.append(s)
                    accumulated_labels.append(concept_name)
                    accumulated_obj_ids.append(current_obj_id)
                    self.object_labels[current_obj_id] = concept_name
                    current_obj_id += 1

        # 2. Evaluate Exemplar Concepts
        if len(self.pcs_exemplar_boxes) > 0:
            norm_boxes_xywh = []
            box_labels = []
            for b, l in zip(self.pcs_exemplar_boxes, self.pcs_exemplar_labels):
                x1, y1, x2, y2 = b[:4]
                nx1 = max(0.0, min(1.0, float(x1) / img_w))
                ny1 = max(0.0, min(1.0, float(y1) / img_h))
                nw = max(0.0, min(1.0 - nx1, float(max(x2 - x1, 0.0)) / img_w))
                nh = max(0.0, min(1.0 - ny1, float(max(y2 - y1, 0.0)) / img_h))
                norm_boxes_xywh.append([nx1, ny1, nw, nh])
                box_labels.append(int(l))

            response = self.predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": self.session_id,
                    "frame_index": self.active_frame_idx,
                    "bounding_boxes": norm_boxes_xywh,
                    "bounding_box_labels": box_labels,
                    "clear_old_boxes": False,
                    "rel_coordinates": True,
                    "output_prob_thresh": self.confidence_threshold,
                }
            )
            if response and "outputs" in response:
                res = self._parse_frame_outputs(response["outputs"], frame_idx=self.active_frame_idx)
                for m, b, s in zip(res["masks"], res["boxes"], res["scores"]):
                    accumulated_masks.append(m)
                    accumulated_boxes.append(b)
                    accumulated_scores.append(s)
                    accumulated_labels.append("exemplar")
                    accumulated_obj_ids.append(current_obj_id)
                    self.object_labels[current_obj_id] = "exemplar"
                    current_obj_id += 1

        # 3. Cleanly bridge PCS detections into SAM2 instance tracking session:
        # Reset the tracker session so temporary text-prompt detector state (which created obj_id 0) is purged!
        if len(accumulated_boxes) > 0:
            self.predictor.handle_request(
                {
                    "type": "reset_session",
                    "session_id": self.session_id,
                }
            )
            for b, oid in zip(accumulated_boxes, accumulated_obj_ids):
                x1, y1, x2, y2 = b[:4]
                nx1 = max(0.0, min(1.0, float(x1) / img_w))
                ny1 = max(0.0, min(1.0, float(y1) / img_h))
                nx2 = max(0.0, min(1.0, float(x2) / img_w))
                ny2 = max(0.0, min(1.0, float(y2) / img_h))
                resp = self.predictor.handle_request(
                    {
                        "type": "add_prompt",
                        "session_id": self.session_id,
                        "frame_index": self.active_frame_idx,
                        "obj_id": int(oid),
                        "points": [[nx1, ny1], [nx2, ny2]],
                        "point_labels": [2, 3],
                        "clear_old_points": False,
                        "rel_coordinates": True,
                        "output_prob_thresh": self.confidence_threshold,
                    }
                )
                if resp and "outputs" in resp:
                    self.last_frame_outputs[self.active_frame_idx] = resp["outputs"]
            self.prompted_frame_indices.add(self.active_frame_idx)

        self.cached_results[self.active_frame_idx] = {
            "masks": accumulated_masks,
            "boxes": accumulated_boxes,
            "scores": accumulated_scores,
            "labels": accumulated_labels,
            "object_ids": accumulated_obj_ids,
        }

    # -------------------------------------------------------------------------
    # 3.2 Propagation
    # -------------------------------------------------------------------------
    def propagate(self, max_frames: Optional[int] = None) -> List[Result]:
        """
        Propagates all keyframe prompts throughout the video.
        Aggregates streaming frame updates and returns a complete list[Result]
        whose length strictly matches len(video) and preserves frame order.
        """
        self._ensure_session()
        total_frames = len(self.video)
        results: List[Result] = [
            {
                "masks": [],
                "boxes": [],
                "scores": [],
                "labels": [],
                "object_ids": [],
            }
            for _ in range(total_frames)
        ]

        if not self.prompted_frame_indices:
            raise RuntimeError(
                "No prompts registered on any frame. Call predict_pcs() or add_points()/add_box() on at least one frame before propagation."
            )

        start_frame = min(self.prompted_frame_indices)

        req_payload = {
            "type": "propagate_in_video",
            "session_id": self.session_id,
            "start_frame_index": start_frame,
            "max_frame_num_to_track": max_frames,
            "output_prob_thresh": self.confidence_threshold,
        }

        stream_gen = self.predictor.handle_stream_request(req_payload)

        for item in stream_gen:
            frame_idx = item.get("frame_index")
            if frame_idx is None or frame_idx < 0 or frame_idx >= total_frames:
                continue

            outputs = item.get("outputs")
            if outputs:
                self.last_frame_outputs[frame_idx] = outputs
            results[frame_idx] = self._parse_frame_outputs(outputs, frame_idx=frame_idx)

        self.cached_results = results
        return self.get_results()

    def _create_empty_results(self) -> List[Result]:
        """Creates an empty list[Result] of length len(self.video)."""
        return [
            {
                "masks": [],
                "boxes": [],
                "scores": [],
                "labels": [],
                "object_ids": [],
            }
            for _ in range(len(self.video))
        ]

    def _parse_frame_outputs(self, outputs: Dict[str, Any], frame_idx: Optional[int] = None) -> Result:
        """Parses model frame outputs into a unified Result TypedDict."""
        out_masks = outputs.get("out_binary_masks")
        out_boxes = outputs.get("out_boxes_xywh")
        out_obj_ids = outputs.get("out_obj_ids")
        out_probs = outputs.get("out_probs")

        if out_masks is None or len(out_masks) == 0:
            return {
                "masks": [],
                "boxes": [],
                "scores": [],
                "labels": [],
                "object_ids": [],
            }

        target_idx = self.active_frame_idx if frame_idx is None else frame_idx
        img_w, img_h = self.video[target_idx].size if (0 <= target_idx < len(self.video)) else (1.0, 1.0)

        masks_np = tensor_to_numpy(out_masks)
        boxes_np = tensor_to_numpy(out_boxes) if out_boxes is not None else None
        obj_ids_np = tensor_to_numpy(out_obj_ids) if out_obj_ids is not None else None
        probs_np = tensor_to_numpy(out_probs) if out_probs is not None else None

        num_objs = len(masks_np)
        frame_masks: List[Mask] = []
        frame_boxes: List[Box] = []
        frame_scores: List[float] = []
        frame_labels: List[str] = []
        frame_obj_ids: List[int] = []

        for i in range(num_objs):
            m = np.squeeze(masks_np[i]) > 0.5
            if m.ndim != 2 or not m.any():
                continue

            obj_id = int(obj_ids_np[i]) if obj_ids_np is not None and i < len(obj_ids_np) else (i + 1)
            score = float(probs_np[i]) if probs_np is not None and i < len(probs_np) else 1.0

            if boxes_np is not None and i < len(boxes_np) and boxes_np[i].any():
                bx, by, bw, bh = boxes_np[i][:4]
                # out_boxes_xywh from video predictor is normalized [0, 1], scale to absolute pixels
                x1 = float(bx * img_w)
                y1 = float(by * img_h)
                x2 = float((bx + bw) * img_w)
                y2 = float((by + bh) * img_h)
                box_xyxy: Box = (x1, y1, x2, y2)
            else:
                ys, xs = np.where(m)
                box_xyxy = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))

            frame_masks.append(m)
            frame_boxes.append(box_xyxy)
            frame_scores.append(score)
            label = self.object_labels.get(obj_id, f"object_{obj_id}")
            frame_labels.append(label)
            frame_obj_ids.append(obj_id)

        return {
            "masks": frame_masks,
            "boxes": frame_boxes,
            "scores": frame_scores,
            "labels": frame_labels,
            "object_ids": frame_obj_ids,
        }

    def _ensure_session(self) -> None:
        if self.session_id is None or len(self.video) == 0:
            raise RuntimeError("Video session not initialised. Call engine.video.initialise(video) first.")
