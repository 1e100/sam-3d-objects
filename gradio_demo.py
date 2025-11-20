#!/usr/bin/env python3

from typing import List, Tuple, Optional

import gradio as gr
import numpy as np
import PIL.Image
import PIL.ImageDraw
import torch
import transformers


# ---------------------------------------------------------------------------
# SAM 2 setup (via Hugging Face Transformers)
# ---------------------------------------------------------------------------

# Detect a suitable device (GPU/CPU/MPS).
_SAM2_DEVICE = transformers.infer_device()

# Choose a SAM 2 checkpoint from HF Hub.
_SAM2_MODEL_ID = "facebook/sam2.1-hiera-large"

# Load model and processor once at startup.
_SAM2_MODEL = transformers.Sam2Model.from_pretrained(
    _SAM2_MODEL_ID
).to(_SAM2_DEVICE)
_SAM2_PROCESSOR = transformers.Sam2Processor.from_pretrained(_SAM2_MODEL_ID)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def run_sam2_segmentation(
    image: PIL.Image.Image,
    points: List[Tuple[float, float]],
) -> Optional[np.ndarray]:
    """Run SAM 2 to get a single mask from a set of positive points."""
    # Return early if there are no points to use as prompts.
    if not points:
        return None

    # Build input_points with shape [batch=1, objects=1, num_points, 2].
    single_object_points: List[List[float]] = []
    for x, y in points:
        single_object_points.append([float(x), float(y)])
    input_points = [[[value for value in single_object_points]]]

    # Build input_labels (all positive) with shape [1, 1, num_points].
    labels_for_points: List[int] = [1 for _ in points]
    input_labels = [[[value for value in labels_for_points]]]

    # Prepare batched model inputs for the image and point prompts.
    inputs = _SAM2_PROCESSOR(
        images=image,
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt",
    )
    inputs = inputs.to(_SAM2_DEVICE)

    # Run the SAM 2 model forward pass with no gradients.
    with torch.no_grad():
        outputs = _SAM2_MODEL(
            **inputs,
            multimask_output=False,
        )

    # Post-process masks to the original image size.
    all_masks = _SAM2_PROCESSOR.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"],
    )

    # Extract masks for the single image in the batch.
    masks_tensor = all_masks[0]

    # Handle possible shapes and pick a single mask.
    if masks_tensor.ndim == 4:
        # Shape: [num_objects, num_masks, H, W].
        mask_tensor = masks_tensor[0, 0]
    elif masks_tensor.ndim == 3:
        # Shape: [num_objects, H, W] (if multimask collapsed).
        mask_tensor = masks_tensor[0]
    elif masks_tensor.ndim == 2:
        # Shape: [H, W].
        mask_tensor = masks_tensor
    else:
        # Unexpected shape: do not return a mask.
        return None

    # Convert the mask to a NumPy boolean array for visualization.
    mask = mask_tensor.numpy() > 0.5
    return mask


def overlay_mask_and_points(
    image: PIL.Image.Image,
    mask: Optional[np.ndarray],
    points: List[Tuple[float, float]],
) -> PIL.Image.Image:
    """Overlay a segmentation mask and clicked points on top of the image."""
    # Convert base image to RGBA so we can alpha-blend overlays.
    base_image = image.convert("RGBA")

    # Create an empty transparent overlay for mask and points.
    overlay = PIL.Image.new("RGBA", base_image.size, (0, 0, 0, 0))

    # If a mask is present, render it as a semi-transparent colored region.
    if mask is not None:
        mask_image = PIL.Image.new("RGBA", base_image.size, (0, 0, 0, 0))
        mask_alpha = 120
        mask_color = (0, 255, 0, mask_alpha)

        mask_pixels = mask_image.load()
        height, width = mask.shape
        for y in range(height):
            for x in range(width):
                if mask[y, x]:
                    mask_pixels[x, y] = mask_color

        overlay = PIL.Image.alpha_composite(overlay, mask_image)

    # Draw the clicked points on their own transparent layer.
    points_layer = PIL.Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    draw = PIL.ImageDraw.Draw(points_layer)

    radius = 5
    for x, y in points:
        bbox = [x - radius, y - radius, x + radius, y + radius]
        draw.ellipse(bbox, outline=(255, 0, 0, 255), width=2)

    # Merge mask overlay and points overlay together.
    overlay = PIL.Image.alpha_composite(overlay, points_layer)

    # Composite the combined overlay onto the original image.
    result = PIL.Image.alpha_composite(base_image, overlay)
    return result


def remove_nearest_point(
    points: List[Tuple[float, float]],
    x: float,
    y: float,
    max_dist: float = 10.0,
) -> List[Tuple[float, float]]:
    """Remove the point nearest to (x, y) if it is within max_dist."""
    # If there are no points, there is nothing to remove.
    if not points:
        return points

    # Compute squared distance from click location to each existing point.
    distances: List[Tuple[int, float]] = []
    for index, (px, py) in enumerate(points):
        squared_distance = (px - x) ** 2 + (py - y) ** 2
        distances.append((index, squared_distance))

    # Find the point with the smallest squared distance.
    closest_index, min_squared_distance = min(
        distances,
        key=lambda item: item[1],
    )

    # If the closest point is too far away, do not modify the list.
    if min_squared_distance > max_dist ** 2:
        return points

    # Remove the closest point from a copy of the list.
    new_points = list(points)
    new_points.pop(closest_index)
    return new_points


# ---------------------------------------------------------------------------
# Gradio event handlers
# ---------------------------------------------------------------------------

def on_image_upload(
    image: PIL.Image.Image,
) -> Tuple[PIL.Image.Image, List[Tuple[float, float]]]:
    """Handle a new image upload and reset the point state."""
    # Reset all prompts when a new image is uploaded.
    points: List[Tuple[float, float]] = []

    # Initial visualization: the clean image with no mask or points.
    result_image = overlay_mask_and_points(image, None, points)
    return result_image, points


def on_image_click(
    image: PIL.Image.Image,
    event: gr.SelectData,
    mode: str,
    points: List[Tuple[float, float]],
) -> Tuple[PIL.Image.Image, List[Tuple[float, float]]]:
    """Handle a click to add or remove points and update SAM 2 mask."""
    # Read the pixel coordinates from the click event.
    x = float(event.index[0])
    y = float(event.index[1])

    # Update the list of points based on the selected interaction mode.
    if mode == "Add point":
        updated_points = list(points)
        updated_points.append((x, y))
    elif mode == "Remove point":
        updated_points = remove_nearest_point(points, x, y)
    else:
        updated_points = points

    # Run SAM 2 with the new set of positive points.
    mask = run_sam2_segmentation(image, updated_points)

    # Render the resulting mask and points on top of the image.
    result_image = overlay_mask_and_points(image, mask, updated_points)
    return result_image, updated_points


def on_clear_points(
    image: PIL.Image.Image,
) -> Tuple[PIL.Image.Image, List[Tuple[float, float]]]:
    """Clear all points and the associated mask for the image."""
    # Reset all prompts to an empty list.
    points: List[Tuple[float, float]] = []

    # Show the original image with no mask or points.
    result_image = overlay_mask_and_points(image, None, points)
    return result_image, points


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks() as demo:
    gr.Markdown(
        "# Interactive SAM 2 Segmentation (Point Prompts)\n"
        "Upload an image, click to add positive points for a single object, "
        "and use **Remove point** mode to delete points that cause unwanted mask regions."
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Input image (click to add/remove points)",
                type="pil",
                interactive=True,
            )
            mode_radio = gr.Radio(
                ["Add point", "Remove point"],
                value="Add point",
                label="Click mode",
            )
            clear_button = gr.Button("Clear all points")
        with gr.Column():
            result_image = gr.Image(
                label="Segmentation result",
                type="pil",
                interactive=False,
            )

    # Keep the current list of clicked points as Gradio state.
    points_state = gr.State([])

    # Reset points when a new image is uploaded.
    input_image.upload(
        fn=on_image_upload,
        inputs=input_image,
        outputs=[result_image, points_state],
    )

    # Update points and mask when the image is clicked.
    input_image.select(
        fn=on_image_click,
        inputs=[input_image, mode_radio, points_state],
        outputs=[result_image, points_state],
    )

    # Clear all points and mask when the button is pressed.
    clear_button.click(
        fn=on_clear_points,
        inputs=[input_image],
        outputs=[result_image, points_state],
    )

if __name__ == "__main__":
    demo.launch(share=True)

