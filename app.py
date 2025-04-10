import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import torch
from predict import load_onnx_model
from utils.helpers import calculate_deforestation_metrics, create_overlay

torch.classes.__path__ = []

# Set page config
st.set_page_config(page_title="Deforestation Detection", page_icon="🌳", layout="wide")

# Set constants
MODEL_INPUT_SIZE = 256  # The size our model expects

# Load ONNX model
@st.cache_resource
def load_cached_onnx_model():
    model_path = "models/deforestation_model.onnx"
    return load_onnx_model(model_path, input_size=MODEL_INPUT_SIZE)

def process_image(model, image):
    """Process a single image and return results"""
    # Save original image dimensions for display
    orig_height, orig_width = image.shape[:2]

    # Make prediction
    mask = model.predict(image)

    # Resize mask back to original dimensions for display
    display_mask = cv2.resize(mask, (orig_width, orig_height))

    # Create binary mask for visualization
    binary_mask = (display_mask > 0.5).astype(np.uint8) * 255

    # Create colored overlay
    overlay = create_overlay(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), display_mask)

    # Calculate metrics
    metrics = calculate_deforestation_metrics(mask)

    return binary_mask, overlay, metrics

def main():
    # App title and description
    st.title("🌳 Deforestation Detection")
    st.markdown(
        """
    This app detects areas of deforestation in satellite or aerial images of forests.
    Upload an image to get started!
    """
    )

    # Model info
    st.info(
        f"⚙️ Model optimized for {MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE} pixel images using ONNX runtime"
    )

    # Load model
    try:
        model = load_cached_onnx_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info(
            "Make sure you have converted your PyTorch model to ONNX format using the utils/onnx_converter.py script."
        )
        st.code(
            "python -m utils.onnx_converter models/best_model_100.pth models/deforestation_model.onnx"
        )
        return

    # File uploader for images
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Display original image
        st.subheader("Original Image")
        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            caption="Uploaded Image",
            use_container_width=True,
        )

        # Add a spinner while processing
        with st.spinner("Processing..."):
            try:
                # Process image
                binary_mask, overlay, metrics = process_image(model, image)

                # Display results in columns
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Segmentation Result")
                    st.image(
                        binary_mask,
                        caption="Forest Areas (White)",
                        use_container_width=True,
                    )

                with col2:
                    st.subheader("Overlay Visualization")
                    st.image(
                        overlay,
                        caption="Green: Forest, Brown: Deforested",
                        use_container_width=True,
                    )

                # Display metrics
                st.subheader("Deforestation Analysis")

                # Create metrics cards
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

                with metrics_col1:
                    st.metric(
                        label="Forest Coverage",
                        value=f"{metrics['forest_percentage']:.1f}%",
                    )

                with metrics_col2:
                    st.metric(
                        label="Deforested Area",
                        value=f"{metrics['deforested_percentage']:.1f}%",
                    )

                with metrics_col3:
                    st.metric(
                        label="Deforestation Level",
                        value=metrics["deforestation_level"],
                    )

            except Exception as e:
                st.error(f"Error during processing: {e}")

if __name__ == "__main__":
    main()