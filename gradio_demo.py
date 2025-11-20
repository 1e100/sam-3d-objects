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

# Select device for running SAM2.
_SAM2_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Use a SAM2 checkpoint from Hugging Face.
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
    # If there are no prompt points, skip segmentation.
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

    # Prepare SAM2 processor inputs for the image and point prompts.
    inputs = _SAM2_PROCESSOR(
        images=image,
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt",
    )
    inputs = inputs.to(_SAM2_DEVICE)

    # Run the SAM 2 model forward pass without gradients.
    with torch.no_grad():
        outputs = _SAM2_MODEL(
            **inputs,
            multimask_output=False,
        )

    # Post-process the predicted masks back to the original image size.
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
        # Shape: [num_objects, H, W].
        mask_tensor = masks_tensor[0]
    elif masks_tensor.ndim == 2:
        # Shape: [H, W].
        mask_tensor = masks_tensor
    else:
        return None

    # Convert the mask to a NumPy boolean array for drawing.
    mask = mask_tensor.numpy() > 0.5
    return mask


def overlay_mask_and_points(
    image: PIL.Image.Image,
    mask: Optional[np.ndarray],
    points: List[Tuple[float, float]],
) -> PIL.Image.Image:
    """Overlay segmentation mask and clicked points directly on the image."""
    # Convert base image to RGBA so we can alpha-blend overlays.
    base_image = image.convert("RGBA")

    # Create an empty transparent overlay for mask and points.
    overlay = PIL.Image.new("RGBA", base_image.size, (0, 0, 0, 0))

    # If a mask is present, draw it as a semi-transparent colored region.
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

    # Draw the clicked points on a separate transparent layer.
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
) -> Tuple[PIL.Image.Image, List[Tuple[float, float]], PIL.Image.Image]:
    """Handle a new image upload, reset points, and show clean image."""
    # Reset all prompt points when a new image is uploaded.
    points: List[Tuple[float, float]] = []

    # Initial visualization is just the clean original image.
    display_image = overlay_mask_and_points(image, None, points)

    # Return what will be shown, the empty points list, and the stored original.
    return display_image, points, image


def on_image_click(
    original_image: PIL.Image.Image,
    event: gr.SelectData,
    mode: str,
    points: List[Tuple[float, float]],
) -> Tuple[PIL.Image.Image, List[Tuple[float, float]]]:
    """Handle a click to add/remove points and update overlay on original."""
    # If there is no original image stored yet, ignore the click.
    if original_image is None:
        return None, points  # Gradio will usually not call this before upload.

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

    # Run SAM 2 on the original image with the updated set of positive points.
    mask = run_sam2_segmentation(original_image, updated_points)

    # Render the overlay directly on the original image for display/clicking.
    display_image = overlay_mask_and_points(original_image, mask, updated_points)
    return display_image, updated_points


def on_clear_points(
    original_image: PIL.Image.Image,
) -> Tuple[PIL.Image.Image, List[Tuple[float, float]]]:
    """Clear all points and mask, returning the clean original image."""
    # If there is no stored original image, return unchanged content.
    if original_image is None:
        return original_image, []

    # Reset all prompts to an empty list.
    points: List[Tuple[float, float]] = []

    # Show the original image with no mask or points.
    display_image = overlay_mask_and_points(original_image, None, points)
    return display_image, points


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks() as demo:
    gr.Markdown(
        "# Interactive SAM 2 Segmentation (Single Object, Point Prompts)\n"
        "Upload an image, click to add positive points on the object, and use "
        "**Remove point** mode to delete points that cause unwanted mask regions. "
        "The mask is overlaid directly on the same image you are clicking."
    )

    with gr.Row():
        with gr.Column():
            image_component = gr.Image(
                label="Image (mask + points overlaid; click to edit)",
                type="pil",
                interactive=True,
            )
            mode_radio = gr.Radio(
                ["Add point", "Remove point"],
                value="Add point",
                label="Click mode",
            )
            clear_button = gr.Button("Clear all points")
        # No separate result image: we always draw on top of this one.

    # State: current list of points and original (unmodified) image.
    points_state = gr.State([])
    original_image_state = gr.State(None)

    # When a new image is uploaded, store the original and reset overlay/points.
    image_component.upload(
        fn=on_image_upload,
        inputs=image_component,
        outputs=[image_component, points_state, original_image_state],
    )

    # When the image is clicked, operate on the stored original image,
    # but update the displayed image (with mask + points) in-place.
    image_component.select(
        fn=on_image_click,
        inputs=[original_image_state, mode_radio, points_state],
        outputs=[image_component, points_state],
    )

    # When clearing, drop all points and show the clean original again.
    clear_button.click(
        fn=on_clear_points,
        inputs=[original_image_state],
        outputs=[image_component, points_state],
    )

if __name__ == "__main__":
    demo.launch(share=True)

