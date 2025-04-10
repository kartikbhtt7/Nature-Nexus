# Forest Surveillance System

## Directory Sturcture
```
Nature-Nexus/
├── app.py                     # Main Streamlit application
├── prediction_engine.py       # Unified prediction engine
├── models/                    # Model files
│   ├── deforestation_model.onnx
│   ├── fire_detection_model.onnx
│   └── audio_model.pth
├── utils/
│   ├── model.py              # Common model architectures
│   ├── process_img.py        # Image processing utilities
│   ├── process_video.py      # Video processing utilities
│   ├── process_audio.py      # Audio processing utilities
│   ├── onnx_converter.py     # ONNX conversion utility
│   ├── deforested_segmentor/ # Deforestation-specific utilities
│   │   ├── __init__.py
│   │   ├── model.py          # Deforestation model loader
│   │   └── predict.py        # Deforestation prediction functions
│   ├── fire_detection/       # Fire detection-specific utilities
│   │   ├── __init__.py
│   │   ├── model.py          # Fire detection model loader
│   │   └── predict.py        # Fire detection prediction functions
│   └── audio_processing/     # Audio processing-specific utilities
│       ├── __init__.py
│       ├── model.py          # Audio model loader
│       └── predict.py        # Audio prediction functions
```