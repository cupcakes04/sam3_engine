from __future__ import annotations

import os
from typing import Any, Optional, Literal
import torch

from sam3_engine.image import SAM3ImageEngine
from sam3_engine.types import CachedConcept
from sam3_engine.video import SAM3VideoEngine


def _resolve_checkpoint(checkpoint_path: str) -> str:
    """
    Enforces local checkpoint loading without Hugging Face downloads.
    Supports checkpoint_dir pointing to either a directory or directly to a .pt file.
    """
    if os.path.isfile(checkpoint_path):
        return os.path.abspath(checkpoint_path)
    
    raise FileNotFoundError(
        f"Could not locate local SAM checkpoint. "
        f"Please provide an explicit 'checkpoint_path' pointing to the pretrained directory of .pt file."
    )


def _resolve_bpe(bpe_path: str,) -> str:
    """Resolves path to BPE tokenizer vocabulary."""
    if os.path.isfile(bpe_path):
        return os.path.abspath(bpe_path)

    raise FileNotFoundError(
        f"Could not locate local BPE tokenizer vocabulary. "
        f"Please provide an explicit 'bpe_path' pointing to the pretrained directory of .pt file."
    )



class SAM3Engine:
    """
    Unified entrypoint facade for Meta SAM 3 / SAM 3.1.

    Enforces local checkpoint loading from `workspace/checkpoints/pretrained/sam3vlm`.
    SAM 3 has a single foundation checkpoint (sam3.1_multiplex.pt or sam3.pt)
    housing both the detector (for PCS) and tracker (for PVS & Video).
    """

    def __init__(
        self,
        image_checkpoint_path: str,
        video_checkpoint_path: str,
        bpe_path: str,
        video_version: Literal['sam3', 'sam3.1'] = 'sam3',
        device: Literal["cuda", "cpu"] = "cuda",
        precision: Literal["bfloat16", "float16", "float32"] = "bfloat16",
        confidence_threshold: float = 0.5,
        enable_inst_interactivity: bool = True,
        video_compile: bool = False,
        video_max_num_objects: int = 16,
        lazy_load: bool = True,
    ) -> None:
        self.device = device
        self.precision = precision
        self.confidence_threshold = float(confidence_threshold)
        self.enable_inst_interactivity = enable_inst_interactivity
        self.video_compile = video_compile
        self.video_max_num_objects = video_max_num_objects

        # Enforce local checkpoint resolution
        # Image model (PCS & PVS) requires sam3.pt (which contains detector and tracker weights).
        # Video model (tracking) uses sam3.1_multiplex.pt when version="sam3.1".
        self.bpe_path = _resolve_bpe(bpe_path)
        self.image_checkpoint_path = _resolve_checkpoint(image_checkpoint_path)
        self.video_checkpoint_path = _resolve_checkpoint(video_checkpoint_path)
        self.video_version = video_version

        # Enable TensorFloat-32 for Ampere GPUs if available
        if torch.cuda.is_available() and device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self._image_engine: Optional[SAM3ImageEngine] = None
        self._video_engine: Optional[SAM3VideoEngine] = None

        if not lazy_load:
            _ = self.image
            _ = self.video

    def set_threshold(self, threshold: float) -> None:
        """Dynamically adjusts confidence threshold across image and video sub-engines."""
        self.confidence_threshold = float(threshold)
        if self._image_engine is not None:
            self._image_engine.set_threshold(self.confidence_threshold)
        if self._video_engine is not None:
            self._video_engine.set_threshold(self.confidence_threshold)

    def encode_concept(self, text: str) -> CachedConcept:
        """
        Precomputes and returns raw text embeddings dictionary for open-vocabulary concepts.
        Image and video invariant: can be passed to either engine.image or engine.video PCS.
        """
        return self.image.encode_concept(text)

    @property
    def image(self) -> SAM3ImageEngine:
        """Lazily initializes and returns the Image Engine."""
        if self._image_engine is None:
            from sam3.model_builder import build_sam3_image_model

            model = build_sam3_image_model(
                checkpoint_path=self.image_checkpoint_path,
                bpe_path=self.bpe_path,
                device=self.device,
                enable_segmentation=True,
                enable_inst_interactivity=self.enable_inst_interactivity,
                load_from_HF=False,
            )
            self._image_engine = SAM3ImageEngine(
                model=model,
                device=self.device,
                confidence_threshold=self.confidence_threshold,
            )
        return self._image_engine

    @property
    def video(self) -> SAM3VideoEngine:
        """Lazily initializes and returns the Video Engine."""
        if self._video_engine is None:
            from sam3.model_builder import build_sam3_predictor

            predictor = build_sam3_predictor(
                checkpoint_path=self.video_checkpoint_path,
                bpe_path=self.bpe_path,
                version=self.video_version,
                compile=self.video_compile,
                max_num_objects=self.video_max_num_objects,
                async_loading_frames=False,
            )
            self._video_engine = SAM3VideoEngine(
                predictor=predictor,
                confidence_threshold=self.confidence_threshold,
            )
        return self._video_engine

    def shutdown(self) -> None:
        """Closes active sessions and frees GPU memory."""
        if self._video_engine is not None:
            self._video_engine.close_session()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
