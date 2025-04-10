import torch
import numpy as np
import cv2
import onnxruntime as ort
from utils.preprocess import preprocess_image
from utils.model import load_model


class PredictionEngine:
    def __init__(self, model_path=None, use_onnx=True, input_size=256):
        """
        Initialize the prediction engine

        Args:
            model_path: Path to the model file (PyTorch or ONNX)
            use_onnx: Whether to use ONNX runtime for inference
            input_size: Input size for the model (default is 256)
        """
        self.use_onnx = use_onnx
        self.input_size = input_size

        if model_path:
            if use_onnx:
                self.model = self._load_onnx_model(model_path)
            else:
                self.model = load_model(model_path)
        else:
            self.model = None

    def _load_onnx_model(self, model_path):
        """
        Load an ONNX model

        Args:
            model_path: Path to the ONNX model

        Returns:
            ONNX Runtime InferenceSession
        """
        # Try with CUDA first, fall back to CPU if needed
        try:
            session = ort.InferenceSession(
                model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            print("ONNX model loaded with CUDA support")
            return session
        except Exception as e:
            print(f"Could not load ONNX model with CUDA, falling back to CPU: {e}")
            session = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"]
            )
            print("ONNX model loaded with CPU support")
            return session

    def preprocess(self, image):
        """
        Preprocess an image for prediction

        Args:
            image: Input image (numpy array)

        Returns:
            Processed image suitable for the model
        """
        # Keep the original image for reference
        self.original_shape = image.shape[:2]

        # Preprocess image
        if self.use_onnx:
            # For ONNX, we need to ensure the input is exactly the expected size
            tensor = preprocess_image(image, img_size=self.input_size)
            return tensor.numpy()
        else:
            # For PyTorch
            return preprocess_image(image, img_size=self.input_size)

    def predict(self, image):
        """
        Make a prediction on an image

        Args:
            image: Input image (numpy array)

        Returns:
            Predicted mask
        """
        if self.model is None:
            raise ValueError("Model not loaded. Initialize with a valid model path.")

        # Preprocess the image
        processed_input = self.preprocess(image)

        # Run inference
        if self.use_onnx:
            # Get input and output names
            input_name = self.model.get_inputs()[0].name
            output_name = self.model.get_outputs()[0].name

            # Run ONNX inference
            outputs = self.model.run([output_name], {input_name: processed_input})

            # Apply sigmoid to output
            mask = 1 / (1 + np.exp(-outputs[0].squeeze()))
        else:
            # PyTorch inference
            with torch.no_grad():
                # Move to device
                device = next(self.model.parameters()).device
                processed_input = processed_input.to(device)

                # Forward pass
                output = self.model(processed_input)
                output = torch.sigmoid(output)

                # Convert to numpy
                mask = output.cpu().numpy().squeeze()

        return mask


def load_pytorch_model(model_path):
    """
    Load the PyTorch model for prediction

    Args:
        model_path: Path to the PyTorch model

    Returns:
        PredictionEngine instance
    """
    return PredictionEngine(model_path, use_onnx=False)


def load_onnx_model(model_path, input_size=256):
    """
    Load the ONNX model for prediction

    Args:
        model_path: Path to the ONNX model
        input_size: Input size for the model

    Returns:
        PredictionEngine instance
    """
    return PredictionEngine(model_path, use_onnx=True, input_size=input_size)


# For backwards compatibility
def predict(model, image):
    """
    Legacy function for prediction

    Args:
        model: Model instance
        image: Input image

    Returns:
        Predicted mask
    """
    if isinstance(model, PredictionEngine):
        return model.predict(image)

    engine = PredictionEngine(use_onnx=True)
    engine.model = model
    return engine.predict(image)