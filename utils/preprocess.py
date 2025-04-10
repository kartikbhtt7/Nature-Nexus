import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


def preprocess_image(image, img_size=256):
    """
    Preprocess the input image for model prediction

    Args:
        image: Input image (numpy array)
        img_size: Size to resize image to (model was trained on 256x256)

    Returns:
        Preprocessed image tensor
    """
    # Define the transformation
    transform = A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    # Apply the transformation
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented = transform(image=image)
    image_tensor = augmented["image"]

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor
