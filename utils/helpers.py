import cv2
import numpy as np


def calculate_deforestation_metrics(mask, threshold=0.5):
    """
    Calculate deforestation metrics from the predicted mask

    Args:
        mask: Predicted mask (numpy array)
        threshold: Threshold to binarize the mask (default is 0.5)

    Returns:
        Dictionary containing deforestation metrics
    """
    # Binarize the mask
    binary_mask = (mask > threshold).astype(np.uint8)

    # Calculate pixel counts
    total_pixels = binary_mask.size
    forest_pixels = np.sum(binary_mask)
    deforested_pixels = total_pixels - forest_pixels

    # Calculate percentages
    forest_percentage = (forest_pixels / total_pixels) * 100
    deforested_percentage = (deforested_pixels / total_pixels) * 100

    # Determine deforestation level
    if deforested_percentage < 20:
        level = "Low"
    elif deforested_percentage < 50:
        level = "Medium"
    else:
        level = "High"

    return {
        "forest_pixels": forest_pixels,
        "deforested_pixels": deforested_pixels,
        "forest_percentage": forest_percentage,
        "deforested_percentage": deforested_percentage,
        "deforestation_level": level,
    }


def create_overlay(original_image, mask, threshold=0.5, alpha=0.5):
    """
    Create a visualization by overlaying the mask on the original image

    Args:
        original_image: Original RGB image
        mask: Predicted mask
        threshold: Threshold to binarize the mask
        alpha: Opacity of the overlay

    Returns:
        Overlay image
    """
    # Resize mask to match original image if needed
    if original_image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))

    # Create binary mask
    binary_mask = (mask > threshold).astype(np.uint8) * 255

    # Create a colored mask (green for forest, red for deforested)
    colored_mask = np.zeros_like(original_image)
    colored_mask[binary_mask == 255] = [0, 255, 0]  # Green for forest
    colored_mask[binary_mask == 0] = [150, 75, 0]  # Brown for deforested

    # Create overlay
    overlay = cv2.addWeighted(original_image, 1 - alpha, colored_mask, alpha, 0)

    return overlay
