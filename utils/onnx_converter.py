import torch
import sys
import numpy as np
from utils.model import AttentionUNet
import onnx
import onnxruntime as ort


def convert_to_onnx(pytorch_model_path, onnx_output_path, input_size=256):
    """
    Convert a PyTorch model to ONNX format

    Args:
        pytorch_model_path: Path to the PyTorch model
        onnx_output_path: Path to save the ONNX model
        input_size: Input size for the model (default is 256x256)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used for conversion: {device}")

    model = AttentionUNet(in_channels=3, out_channels=1)
    model.to(device)
    model.load_state_dict(torch.load(pytorch_model_path, map_location=device))

    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size, device=device)

    # Export the model
    torch.onnx.export(
        model,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        onnx_output_path,  # where to save the model
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=12,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},  # variable length axes
            "output": {0: "batch_size", 2: "height", 3: "width"},
        },
    )

    print(f"Model converted and saved to {onnx_output_path}")

    verify_onnx_model(onnx_output_path, input_size)


def verify_onnx_model(onnx_model_path, input_size=256):
    """
    Verify the ONNX model to ensure it was exported correctly

    Args:
        onnx_model_path: Path to the ONNX model
        input_size: Input size used during export
    """
    try:
        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid")
    except Exception as e:
        print(f"ONNX model validation failed: {e}")
        return False

    try:
        session = ort.InferenceSession(
            onnx_model_path, providers=["CPUExecutionProvider"]
        )

        input_data = np.random.rand(1, 3, input_size, input_size).astype(np.float32)

        # Get input and output names
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # Run inference
        outputs = session.run([output_name], {input_name: input_data})

        print(f"ONNX model inference test passed. Output shape: {outputs[0].shape}")
        return True
    except Exception as e:
        print(f"ONNX model inference test failed: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python -m utils.onnx_converter <pytorch_model_path> <onnx_output_path> [input_size]"
        )
        sys.exit(1)

    pytorch_model_path = sys.argv[1]
    onnx_output_path = sys.argv[2]
    input_size = int(sys.argv[3]) if len(sys.argv) > 3 else 256

    convert_to_onnx(pytorch_model_path, onnx_output_path, input_size)
