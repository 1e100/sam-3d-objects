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

# Select device for running SAM 2.
_SAM2_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Use a SAM 2 checkpoint from Hugging Face.
_SAM2_MODEL_ID = "facebook/sam2.1-hiera-large"

# Load model and processor once at startup.
_SAM2_MODEL = transformers.Sam2Model.from_pretrained(_SAM2_MODEL_ID).to(_SAM2_DEVICE)
_SAM2_PROCESSOR = transformers.Sam2Processor.from_pretrained(_SAM2_MODEL_ID)


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------


def resize_to_max_side(
    image: PIL.Image.Image,
    max_side: int = 1024,
) -> PIL.Image.Image:
    """Resize image so that its longest side is at most max_side, keeping
    aspect ratio."""
    # Read original width and height.
    width, height = image.size
    longest_side = max(width, height)

    # If already within size limit, return image unchanged.
    if longest_side <= max_side:
        return image

    # Compute scale factor to bring longest side down to max_side.
    scale = max_side / float(longest_side)

    # Compute new dimensions with rounding.
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    # Resize with high-quality resampling.
    resized = image.resize((new_width, new_height), PIL.Image.LANCZOS)
    return resized


# ---------------------------------------------------------------------------
# Core SAM 2 helpers
# ---------------------------------------------------------------------------


def run_sam2_segmentation(
    image: PIL.Image.Image,
    objects: List[List[Tuple[float, float]]],
) -> Optional[np.ndarray]:
    """Run SAM 2 to get a combined mask from multiple objects with positive
    points."""
    # Skip segmentation if there are no objects or no points at all.
    if not objects or not any(len(obj_points) > 0 for obj_points in objects):
        return None

    # Collect prompt points for all non-empty objects.
    prompt_points: List[List[List[float]]] = []
    prompt_labels: List[List[List[int]]] = []

    # Fill prompt lists from object point sets.
    for obj_points in objects:
        if not obj_points:
            continue
        obj_point_list: List[List[float]] = []
        obj_label_list: List[int] = []
        for x, y in obj_points:
            obj_point_list.append([float(x), float(y)])
            obj_label_list.append(1)
        prompt_points.append(obj_point_list)
        prompt_labels.append(obj_label_list)

    # If everything was empty, do not run segmentation.
    if not prompt_points:
        return None

    # Wrap prompts for a single-image batch.
    input_points = [prompt_points]
    input_labels = [prompt_labels]

    # Prepare processor inputs with points and labels.
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

    # Post-process predicted masks back to the original image size.
    all_masks = _SAM2_PROCESSOR.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"],
    )

    # Take masks for our single image.
    masks_tensor = all_masks[0]
    masks_array = masks_tensor.numpy()

    # Combine masks from all objects into a single boolean mask.
    if masks_array.ndim == 4:
        # Shape: [num_objects, num_masks, H, W].
        combined = masks_array > 0.5
        combined = combined.any(axis=1).any(axis=0)
    elif masks_array.ndim == 3:
        # Shape: [num_objects, H, W].
        combined = (masks_array > 0.5).any(axis=0)
    elif masks_array.ndim == 2:
        # Shape: [H, W].
        combined = masks_array > 0.5
    else:
        # Unexpected shape: do not return a mask.
        return None

    # Return combined mask as boolean NumPy array.
    return combined


def overlay_mask_and_points(
    image: PIL.Image.Image,
    mask: Optional[np.ndarray],
    objects: List[List[Tuple[float, float]]],
) -> PIL.Image.Image:
    """Overlay segmentation mask and all object points directly on the
    image."""
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
        # Fill pixels where mask is True.
        for y in range(height):
            for x in range(width):
                if mask[y, x]:
                    mask_pixels[x, y] = mask_color

        # Merge mask overlay into the main overlay.
        overlay = PIL.Image.alpha_composite(overlay, mask_image)

    # Draw the clicked points for all objects on a separate layer.
    points_layer = PIL.Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    draw = PIL.ImageDraw.Draw(points_layer)

    # Flatten all points across objects for visualization.
    all_points: List[Tuple[float, float]] = []
    for obj_points in objects:
        for x, y in obj_points:
            all_points.append((x, y))

    # Draw a red circle for each point.
    radius = 5
    for x, y in all_points:
        bbox = [x - radius, y - radius, x + radius, y + radius]
        draw.ellipse(bbox, outline=(255, 0, 0, 255), width=2)

    # Merge the points overlay into the main overlay.
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
    if min_squared_distance > max_dist**2:
        return points

    # Remove the closest point from a copy of the list.
    new_points = list(points)
    new_points.pop(closest_index)
    return new_points


def _object_label(index: int) -> str:
    """Build a user-visible label for an object index."""
    return f"Object {index + 1}"


# ---------------------------------------------------------------------------
# Gradio event handlers
# ---------------------------------------------------------------------------


def on_image_upload(
    image: PIL.Image.Image,
) -> Tuple[
    PIL.Image.Image, List[List[Tuple[float, float]]], int, PIL.Image.Image, object
]:
    """Handle a new image upload, resize, reset objects, and show clean
    image."""
    # If no image is provided, propagate empty content.
    if image is None:
        return image, [], 0, image, gr.update(choices=[], value=None)

    # Resize the uploaded image so that its longest side is at most 1024.
    resized = resize_to_max_side(image, max_side=1024)

    # Initialize with one empty object for the new image.
    objects: List[List[Tuple[float, float]]] = [[]]

    # Set the active object index to the first object.
    active_index = 0

    # Initial visualization: resized clean image without mask or points.
    display_image = overlay_mask_and_points(resized, None, objects)

    # Prepare dropdown choices for the single active object.
    choices = [_object_label(0)]
    dropdown_update = gr.update(choices=choices, value=_object_label(0))

    # Return displayed image, objects state, active index, stored original, and
    # dropdown update.
    return display_image, objects, active_index, resized, dropdown_update


def on_image_click(
    original_image: PIL.Image.Image,
    event: gr.SelectData,
    mode: str,
    objects: List[List[Tuple[float, float]]],
    active_object_index: int,
) -> Tuple[PIL.Image.Image, List[List[Tuple[float, float]]]]:
    """Handle click to add/remove points on active object and update
    overlay."""
    # If there is no stored original image, ignore the click.
    if original_image is None:
        return original_image, objects

    # Ensure there is at least one object list in the state.
    if not objects:
        objects = [[]]
        active_object_index = 0

    # Clamp active object index to valid range.
    if active_object_index < 0:
        active_object_index = 0
    if active_object_index >= len(objects):
        active_object_index = len(objects) - 1

    # Extract click coordinates from the event.
    x = float(event.index[0])
    y = float(event.index[1])

    # Copy current objects list so we can modify it safely.
    new_objects: List[List[Tuple[float, float]]] = []
    for obj_points in objects:
        new_objects.append(list(obj_points))

    # Get points for the currently active object.
    current_points = new_objects[active_object_index]

    # Update the active object's points based on the current mode.
    if mode == "Add point":
        updated_points = list(current_points)
        updated_points.append((x, y))
    elif mode == "Remove point":
        updated_points = remove_nearest_point(current_points, x, y)
    else:
        updated_points = current_points

    # Store updated points back into the objects list.
    new_objects[active_object_index] = updated_points

    # Run SAM 2 with the updated object prompts to obtain a segmentation mask.
    mask = run_sam2_segmentation(original_image, new_objects)

    # Render overlay (mask + all points) directly on the original image.
    display_image = overlay_mask_and_points(original_image, mask, new_objects)
    return display_image, new_objects


def on_clear_points(
    original_image: PIL.Image.Image,
) -> Tuple[PIL.Image.Image, List[List[Tuple[float, float]]], int, object]:
    """Clear all objects and points, returning the clean original image."""
    # If there is no stored original image, propagate empty content.
    if original_image is None:
        return original_image, [], 0, gr.update(choices=[], value=None)

    # Reset to a single empty object.
    objects: List[List[Tuple[float, float]]] = [[]]
    active_index = 0

    # Show the original image with no mask or points.
    display_image = overlay_mask_and_points(original_image, None, objects)

    # Update dropdown to a single active object.
    choices = [_object_label(0)]
    dropdown_update = gr.update(choices=choices, value=_object_label(0))

    # Return updated display, object state, active index, and dropdown update.
    return display_image, objects, active_index, dropdown_update


def on_create_new_object(
    original_image: PIL.Image.Image,
    objects: List[List[Tuple[float, float]]],
) -> Tuple[PIL.Image.Image, List[List[Tuple[float, float]]], int, object]:
    """Create a new empty object, set it active, and refresh the overlay."""
    # If there is no stored original image, do nothing useful.
    if original_image is None:
        return original_image, objects, 0, gr.update()

    # Start from current objects list, or an empty list if none.
    if objects is None:
        objects = []

    # Append a new empty object to the list.
    new_objects: List[List[Tuple[float, float]]] = []
    for obj_points in objects:
        new_objects.append(list(obj_points))
    new_objects.append([])

    # Set the newly created object as active.
    active_index = len(new_objects) - 1

    # Run SAM 2 with existing non-empty objects to get a mask (if any).
    mask = run_sam2_segmentation(original_image, new_objects)

    # Overlay current masks and all points on the original image.
    display_image = overlay_mask_and_points(original_image, mask, new_objects)

    # Build updated dropdown choices and select the new object.
    choices = []
    for idx in range(len(new_objects)):
        choices.append(_object_label(idx))
    dropdown_update = gr.update(
        choices=choices,
        value=_object_label(active_index),
    )

    # Return updated display, objects, active index, and dropdown update.
    return display_image, new_objects, active_index, dropdown_update


def on_change_active_object(
    original_image: PIL.Image.Image,
    objects: List[List[Tuple[float, float]]],
    active_label: str,
) -> Tuple[PIL.Image.Image, int]:
    """Handle change of active object via dropdown and refresh the overlay."""
    # If there is no image or no objects, keep existing state.
    if original_image is None or not objects:
        return original_image, 0

    # Parse the object index from the label like "Object 3".
    index = 0
    if isinstance(active_label, str) and active_label.startswith("Object "):
        parts = active_label.split()
        if len(parts) == 2:
            try:
                parsed = int(parts[1]) - 1
                index = parsed
            except ValueError:
                index = 0

    # Clamp to valid range.
    if index < 0:
        index = 0
    if index >= len(objects):
        index = len(objects) - 1

    # Recompute mask with current prompts.
    mask = run_sam2_segmentation(original_image, objects)

    # Overlay mask and all points on the original image.
    display_image = overlay_mask_and_points(original_image, mask, objects)
    return display_image, index


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks() as demo:
    gr.Markdown(
        "# Interactive SAM 2 Segmentation (Multiple Objects, Point Prompts)\n"
        "Upload an image (resized to max 1024px on the longest side), click "
        "to add positive points on the active object, use **Remove point** "
        "mode to delete points, and use **Create new object** to start "
        "annotating another object. All masks are overlaid on the "
        "same image for easier refinement."
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
            active_object_dropdown = gr.Dropdown(
                label="Active object",
                choices=[],
                value=None,
            )
            create_object_button = gr.Button("Create new object")
            clear_button = gr.Button("Clear all points and objects")
        # Single image pane: we always draw overlays directly on this image.

    # State: list of objects (each object is a list of points) and active
    # object index.
    objects_state = gr.State([])
    active_object_state = gr.State(0)
    original_image_state = gr.State(None)

    # When a new image is uploaded:
    # - Resize it
    # - Initialize objects state
    # - Set active object
    # - Store resized original
    # - Update dropdown
    image_component.upload(
        fn=on_image_upload,
        inputs=image_component,
        outputs=[
            image_component,
            objects_state,
            active_object_state,
            original_image_state,
            active_object_dropdown,
        ],
    )

    # When the image is clicked:
    # - Modify points for the active object
    # - Recompute combined mask
    # - Update overlaid image and objects state
    image_component.select(
        fn=on_image_click,
        inputs=[
            original_image_state,
            mode_radio,
            objects_state,
            active_object_state,
        ],
        outputs=[
            image_component,
            objects_state,
        ],
    )

    # When the active object is changed in the dropdown:
    # - Update active object index
    # - Recompute and redraw overlay
    active_object_dropdown.change(
        fn=on_change_active_object,
        inputs=[
            original_image_state,
            objects_state,
            active_object_dropdown,
        ],
        outputs=[
            image_component,
            active_object_state,
        ],
    )

    # When "Create new object" is pressed:
    # - Append a new empty object
    # - Set it active
    # - Recompute and redraw overlay
    # - Update dropdown choices and active value
    create_object_button.click(
        fn=on_create_new_object,
        inputs=[
            original_image_state,
            objects_state,
        ],
        outputs=[
            image_component,
            objects_state,
            active_object_state,
            active_object_dropdown,
        ],
    )

    # When "Clear all points and objects" is pressed:
    # - Reset to a single empty object
    # - Clear all masks
    # - Redraw clean original image
    # - Update dropdown
    clear_button.click(
        fn=on_clear_points,
        inputs=[original_image_state],
        outputs=[
            image_component,
            objects_state,
            active_object_state,
            active_object_dropdown,
        ],
    )

if __name__ == "__main__":
    demo.launch(share=True)
