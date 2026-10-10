from __future__ import annotations

from typing import List, Optional, Sequence, Union
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import matplotlib.ticker as ticker

from sam3_engine.types import Box, Mask, Result


def tensor_to_numpy(tensor_or_array: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Safely converts a PyTorch tensor (including bfloat16 and float16) or array to NumPy."""
    if isinstance(tensor_or_array, torch.Tensor):
        t = tensor_or_array.detach()
        if t.dtype in (torch.bfloat16, torch.float16):
            t = t.float()
        return t.cpu().numpy()
    return np.asarray(tensor_or_array)


def get_distinct_colors(num_colors: int) -> np.ndarray:
    """Generates distinct RGB colors for segmentation masks."""
    if num_colors <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    cmap = plt.get_cmap("tab20" if num_colors <= 20 else "gist_rainbow")
    colors = cmap(np.linspace(0, 1, num_colors))[:, :3]
    return colors


class Visualiser:
    """
    Visualization engine for Result TypedDicts.
    """

    @staticmethod
    def overlay_masks(
        image: Union[Image.Image, np.ndarray],
        masks: Sequence[Mask],
        alpha: float = 0.5,
        colors: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Blends binary masks onto an RGB image.
        """
        if isinstance(image, Image.Image):
            img_np = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        else:
            img_np = tensor_to_numpy(image).astype(np.float32)
            if img_np.max() > 1.0:
                img_np = img_np / 255.0

        num_masks = len(masks)
        if num_masks == 0:
            return (img_np * 255).astype(np.uint8)

        if colors is None:
            colors = get_distinct_colors(num_masks)

        output = img_np.copy()
        for i, m in enumerate(masks):
            mask_2d = np.squeeze(tensor_to_numpy(m)) > 0.5
            if not mask_2d.any():
                continue
            color = colors[i % len(colors)]
            for c in range(3):
                output[mask_2d, c] = output[mask_2d, c] * (1.0 - alpha) + color[c] * alpha

        return (np.clip(output, 0.0, 1.0) * 255).astype(np.uint8)

    @classmethod
    def overlay(
        cls,
        result: Result,
        image: Image.Image,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Returns the image with masks overlaid as an (H, W, 3) uint8 numpy array."""
        return cls.overlay_masks(image, result["masks"], alpha=alpha)

    @classmethod
    def visualise(
        cls,
        result: Result,
        image: Image.Image,
        alpha: float = 0.5,
        show_boxes: bool = True,
        show_labels: bool = True,
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Renders inline matplotlib visualization with masks, bounding boxes, and object labels.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        masks = result["masks"]
        boxes = result["boxes"]
        scores = result["scores"]
        labels = result["labels"]
        object_ids = result["object_ids"]

        display_img = cls.overlay_masks(image, masks, alpha=alpha)
        ax.imshow(display_img)
        ax.axis("off")

        num_items = len(masks)
        colors = get_distinct_colors(max(num_items, 1))

        if show_boxes and boxes:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box[:4]
                w, h = x2 - x1, y2 - y1
                color = colors[i % len(colors)]

                rect = plt.Rectangle(
                    (x1, y1),
                    w,
                    h,
                    fill=False,
                    edgecolor=color,
                    linewidth=2,
                )
                ax.add_patch(rect)

                if show_labels:
                    parts = []
                    if i < len(object_ids):
                        parts.append(f"ID:{object_ids[i]}")
                    if i < len(labels) and labels[i]:
                        parts.append(str(labels[i]))
                    if i < len(scores) and scores[i] is not None:
                        parts.append(f"{scores[i]:.2f}")

                    caption = " | ".join(parts)
                    ax.text(
                        x1,
                        max(y1 - 4, 0),
                        caption,
                        fontsize=9,
                        color="white",
                        bbox=dict(facecolor=color, alpha=0.8, edgecolor="none", pad=2),
                    )

        default_title = title or f"Detections: {len(masks)} objects"
        ax.set_title(default_title, fontsize=12, fontweight="bold")
        return ax

    @classmethod
    def visualise_batch(
        cls,
        results: List[Result],
        images: List[Image.Image],
        cols: int = 2,
        alpha: float = 0.5,
        figsize_per_img: tuple[int, int] = (7, 6),
    ) -> plt.Figure:
        """Renders multiple image results in a grid layout."""
        num_items = len(results)
        if num_items == 0:
            raise ValueError("No results to visualize.")

        rows = (num_items + cols - 1) // cols
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(figsize_per_img[0] * cols, figsize_per_img[1] * rows),
            squeeze=False,
        )

        for i, (res, img) in enumerate(zip(results, images)):
            r, c = divmod(i, cols)
            cls.visualise(
                result=res,
                image=img,
                alpha=alpha,
                ax=axes[r, c],
                title=f"Frame / Image {i} ({len(res['masks'])} objects)",
            )

        for j in range(num_items, rows * cols):
            r, c = divmod(j, cols)
            axes[r, c].axis("off")

        plt.tight_layout()
        return fig
    
    @staticmethod
    def show_image_with_grid(
        image_input: Image.Image,
        points: Optional[list[tuple[int, int]]] = None,
        major_step: int = 100,
        minor_step: int = 20,
        figsize: tuple = (12, 10),
        title: str = "",
        point_color: str = "#FF3366",
        point_size: int = 60,
    ):
        """
        Renders an image overlaid with fine graph-paper grids for coordinate estimation,
        with optional point plotting.

        Args:
            image_input: PIL.Image instance.
            points: Optional list of (x, y) absolute coordinate tuples.
            major_step: Pixel spacing for prominent grid lines and numerical labels.
            minor_step: Pixel spacing for faint sub-grid lines. Set to None to disable.
            figsize: Figure dimensions for matplotlib display.
            title: Optional plot title showing resolution or metadata.
            point_color: Hex or color name for the plotted points.
            point_size: Marker size for plotted points.
        """
        img = image_input.convert("RGB")
        width, height = img.size

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img)

        # Configure major ticks (labeled, prominent lines)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(major_step))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(major_step))

        # Configure minor ticks (sub-grid lines)
        if minor_step:
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(minor_step))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(minor_step))
            ax.grid(which="minor", color="cyan", linestyle=":", linewidth=0.6, alpha=0.35)

        ax.grid(which="major", color="yellow", linestyle="--", linewidth=1.0, alpha=0.75)

        # Keep ticks visible on top and right edges
        ax.tick_params(
            which="both",
            top=True,
            bottom=True,
            left=True,
            right=True,
            labeltop=True,
            labelbottom=True,
            labelleft=True,
            labelright=True,
            labelsize=8,
            colors="white",
        )

        # Explicit X/Y axis labels clarifying coordinate orientation
        ax.set_xlabel("X (Width / Horizontal Pixel Index)", color="white", fontsize=10, labelpad=8)
        ax.set_ylabel("Y (Height / Vertical Pixel Index)", color="white", fontsize=10, labelpad=8)

        # Plot points if provided: points = [(x1, y1), (x2, y2), ...]
        if points:
            pts_x = [p[0] for p in points]
            pts_y = [p[1] for p in points]

            # Render crosshairs and center circles for precision targeting
            ax.scatter(pts_x, pts_y, s=point_size, c=point_color, marker="o", edgecolors="white", linewidths=1.5, zorder=5)
            ax.scatter(pts_x, pts_y, s=point_size * 2, c=point_color, marker="+", linewidths=1.5, zorder=6)

            for idx, (px, py) in enumerate(points):
                ax.annotate(
                    f"P{idx}: ({px}, {py})",
                    xy=(px, py),
                    xytext=(px + 8, py - 8),
                    color="white",
                    fontsize=9,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#1e1e1e", edgecolor=point_color, alpha=0.85),
                    zorder=7,
                )

        # Pixel boundaries (top-left is (0, 0))
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)

        # Dark theme frame
        fig.patch.set_facecolor("#1e1e1e")
        ax.set_facecolor("#1e1e1e")

        display_title = title or f"Image: {width}x{height} px | Top-Left is (X=0, Y=0)"
        ax.set_title(display_title, color="white", fontsize=11, pad=20)

        plt.tight_layout()
        plt.show()