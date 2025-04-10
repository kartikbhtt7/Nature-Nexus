import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import torch

# Import deforestation modules
from prediction_engine import load_onnx_model
from utils.helpers import calculate_deforestation_metrics, create_overlay

# Import audio classification modules
from utils.audio_processing import preprocess_audio
from utils.audio_model import load_audio_model, predict_audio, class_names

# Import YOLO detection modules
from utils.onnx_inference import YOLOv11

# Ensure torch classes path is initialized to avoid warnings
torch.classes.__path__ = []

# Set page config
st.set_page_config(
    page_title="Nature Nexus - Forest Surveillance",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DEFOREST_MODEL_INPUT_SIZE = 256
AUDIO_MODEL_PATH = "models/best_model.pth"
YOLO_MODEL_PATH = "models/best_model.onnx"

# Initialize session state for navigation
if 'current_service' not in st.session_state:
    st.session_state.current_service = 'deforestation'
if 'audio_input_method' not in st.session_state:
    st.session_state.audio_input_method = 'upload'
if 'detection_input_method' not in st.session_state:
    st.session_state.detection_input_method = 'image'

# Sidebar for navigation
with st.sidebar:
    st.title("Nature Nexus")
    st.subheader("Forest Surveillance System")
    
    selected_service = st.radio(
        "Select Service:",
        ["Deforestation Detection", "Forest Audio Surveillance", "Object Detection"]
    )
    
    if selected_service == "Deforestation Detection":
        st.session_state.current_service = 'deforestation'
    elif selected_service == "Forest Audio Surveillance":
        st.session_state.current_service = 'audio'
    else:
        st.session_state.current_service = 'detection'
    
    st.markdown("---")
    
    # Service-specific sidebar content
    if st.session_state.current_service == 'deforestation':
        st.info(
            """
            **Deforestation Detection**
            
            Upload satellite or aerial images to detect areas of deforestation.
            """
        )
    elif st.session_state.current_service == 'audio':
        st.info(
            """
            **Forest Audio Surveillance**
            
            Detect unusual human-related sounds in forested regions.
            """
        )
        
        # Audio service specific controls
        st.subheader("Audio Configuration")
        audio_input_method = st.radio(
            "Select Input Method:",
            ("Upload Audio", "Record Audio"),
            index=0 if st.session_state.audio_input_method == 'upload' else 1
        )
        st.session_state.audio_input_method = 'upload' if audio_input_method == "Upload Audio" else 'record'
        
        # Audio class information
        st.markdown("**Detection Classes:**")
        
        # Group classes by category
        human_sounds = ['footsteps', 'coughing', 'laughing', 'breathing', 
                       'drinking_sipping', 'snoring', 'sneezing']
        tool_sounds = ['chainsaw', 'hand_saw']
        vehicle_sounds = ['car_horn', 'engine', 'siren']
        other_sounds = ['crackling_fire', 'fireworks']
        
        st.markdown("👤 **Human Sounds:** " + ", ".join([s.capitalize() for s in human_sounds]))
        st.markdown("🔨 **Tool Sounds:** " + ", ".join([s.capitalize() for s in tool_sounds]))
        st.markdown("🚗 **Vehicle Sounds:** " + ", ".join([s.capitalize() for s in vehicle_sounds]))
        st.markdown("💥 **Other Sounds:** " + ", ".join([s.capitalize() for s in other_sounds]))
    else:  # Object Detection
        st.info(
            """
            **Object Detection**
            
            Detect trespassers, vehicles, fires, and other objects in forest surveillance footage.
            """
        )
        
        # Detection service specific controls
        st.subheader("Detection Configuration")
        detection_input_method = st.radio(
            "Select Input Method:",
            ("Image", "Video", "Camera"),
            index=0 if st.session_state.detection_input_method == 'image' else 
                  (1 if st.session_state.detection_input_method == 'video' else 2)
        )
        
        if detection_input_method == "Image":
            st.session_state.detection_input_method = 'image'
        elif detection_input_method == "Video":
            st.session_state.detection_input_method = 'video'
        else:
            st.session_state.detection_input_method = 'camera'
            
        # Detection threshold controls
        st.subheader("Detection Settings")
        confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
        iou_thres = st.slider("IoU Threshold", 0.0, 1.0, 0.5)
        
        # Detection class information
        st.markdown("**Detection Classes:**")
        st.markdown("🚴 **Bike/Bicycle**")
        st.markdown("🚚 **Bus/Truck**")
        st.markdown("🚗 **Car**")
        st.markdown("🔥 **Fire**")
        st.markdown("👤 **Human**")
        st.markdown("💨 **Smoke**")

# Load deforestation model
@st.cache_resource
def load_cached_deforestation_model():
    model_path = "models/deforestation_model.onnx"
    return load_onnx_model(model_path, input_size=DEFOREST_MODEL_INPUT_SIZE)

# Load audio model
@st.cache_resource
def load_cached_audio_model():
    return load_audio_model(AUDIO_MODEL_PATH)

@st.cache_resource
def load_cached_yolo_model():
    return YOLOv11(YOLO_MODEL_PATH)

# Process image for deforestation detection
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

# Visualize audio for audio classification
def visualize_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    duration = len(y) / sr
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    
    # Waveform plot
    librosa.display.waveshow(y, sr=sr, ax=ax[0])
    ax[0].set_title('Audio Waveform')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Amplitude')
    
    # Spectrogram plot
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax[1])
    fig.colorbar(img, ax=ax[1], format='%+2.0f dB')
    ax[1].set_title('Mel Spectrogram')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    return y, sr, duration

# Process audio for classification
def process_audio(audio_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_file.read() if hasattr(audio_file, 'read') else audio_file)
        audio_path = tmp_file.name
    
    try:
        # Load audio model
        audio_model = load_cached_audio_model()
        
        # Visualize audio
        with st.spinner('Analyzing audio...'):
            y, sr, duration = visualize_audio(audio_path)
            st.caption(f"Audio duration: {duration:.2f} seconds")
        
        # Make prediction
        with st.spinner('Making prediction...'):
            class_name, confidence = predict_audio(audio_path, audio_model)
        
        # Display results
        st.subheader("Detection Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Detected Sound", class_name.replace('_', ' ').title())
        with col2:
            st.metric("Confidence", f"{confidence*100:.2f}%")
        
        # Show alerts based on class
        human_sounds = ['footsteps', 'coughing', 'laughing', 'breathing', 
                      'drinking_sipping', 'snoring', 'sneezing']
        tool_sounds = ['chainsaw', 'hand_saw']
        
        if class_name in human_sounds:
            st.warning("""
            ⚠️ **Human Activity Detected!**
            Potential human presence in the monitored area.
            """)
        elif class_name in tool_sounds:
            st.error("""
            🚨 **ALERT: Human Tool Detected!**
            Potential illegal logging or activity detected. Consider immediate verification.
            """)
        elif class_name in ['car_horn', 'engine', 'siren']:
            st.warning("""
            ⚠️ **Vehicle Detected!**
            Vehicle sounds detected in the monitored area.
            """)
        elif class_name == 'fireworks':
            st.error("""
            🚨 **ALERT: Fireworks Detected!**
            Potential fire hazard and disturbance to wildlife. Immediate verification required.
            """)
        elif class_name == 'crackling_fire':
            st.error("""
            🚨 **ALERT: Fire Detected!**
            Potential wildfire detected. Immediate verification required.
            """)
        else:
            st.success("✅ Environmental sound detected - no immediate threat")
            
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        st.exception(e)
    finally:
        # Clean up temp file
        try:
            os.unlink(audio_path)
        except:
            pass

# Deforestation detection UI
def show_deforestation_detection():
    # App title and description
    st.title("🌳 Deforestation Detection")
    st.markdown(
        """
        This service detects areas of deforestation in satellite or aerial images of forests.
        Upload an image to get started!
        """
    )

    # Model info
    st.info(
        f"⚙️ Model optimized for {DEFOREST_MODEL_INPUT_SIZE}x{DEFOREST_MODEL_INPUT_SIZE} pixel images using ONNX runtime"
    )

    # Load model
    try:
        model = load_cached_deforestation_model()
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

# Audio classification UI
def show_audio_classification():
    # App title and description
    st.title("🎧 Forest Audio Surveillance")
    st.markdown("""
    Detect unusual human-related sounds in forested regions to prevent illegal activities.
    Supported sounds: {}
    """.format(", ".join(class_names)))
    
    if st.session_state.audio_input_method == 'upload':
        st.header("Upload Audio File")
        
        sample_col, upload_col = st.columns(2)
        with sample_col:
            st.info("Upload a WAV, MP3 or OGG file with forest sounds")
            st.markdown("""
            **Tips for best results:**
            - Use audio with minimal background noise
            - Ensure the sound of interest is clear
            - 2-3 second clips work best
            """)
        
        with upload_col:
            audio_file = st.file_uploader(
                "Choose an audio file",
                type=["wav", "mp3", "ogg"],
                help="Supported formats: WAV, MP3, OGG"
            )
            
        if audio_file:
            st.success("File uploaded successfully!")
            with st.expander("Audio Preview", expanded=True):
                st.audio(audio_file)
            process_audio(audio_file)

    else:  # Record mode
        st.header("Record Live Audio")
        
        st.info("""
        Click the microphone button below to record a sound for analysis.  
        **Note:** Please ensure your browser has permission to access your microphone.  
        When prompted, click "Allow" to enable recording.
        """)
        
        recorded_audio = st.audio_input(
            label="Record a sound",
            key="audio_recorder",
            help="Click to record forest sounds for analysis",
            label_visibility="visible"
        )
        
        if recorded_audio:
            st.success("Audio recorded successfully!")
            with st.expander("Recorded Audio", expanded=True):
                st.audio(recorded_audio)
            process_audio(recorded_audio)
        else:
            st.write("Waiting for recording...")

# Object Detection UI
def show_object_detection():
    # App title and description
    st.title("🔍 Forest Object Detection")
    st.markdown(
        """
        Detect trespassers, vehicles, fires, and other objects in forest surveillance footage.
        Choose an input method to begin detection.
        """
    )

    # Model info
    st.info("⚙️ Object detection model optimized with ONNX runtime for faster inference")

    # Load model
    try:
        model = load_cached_yolo_model()
        # Update model confidence and IoU thresholds from sidebar
        confidence = st.session_state.get('confidence', 0.5)
        iou_thres = st.session_state.get('iou_thres', 0.5)
        model.conf_thres = confidence
        model.iou_thres = iou_thres
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info(
            "Make sure you have the YOLO ONNX model file available at models/best_model.onnx"
        )
        return

    # Input method based selection
    if st.session_state.detection_input_method == 'image':
        # Image upload
        img_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if img_file is not None:
            # Load image
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is not None:
                # Display original image
                st.subheader("Original Image")
                st.image(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    caption="Uploaded Image",
                    use_container_width=True,
                )
                
                # Process with detection model
                with st.spinner("Processing image..."):
                    try:
                        detections = model.detect(image)
                        result_image = model.draw_detections(image.copy(), detections)
                        
                        # Display results
                        st.subheader("Detection Results")
                        st.image(
                            cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB),
                            caption="Detected Objects",
                            use_container_width=True,
                        )
                        
                        # Display detection statistics
                        st.subheader("Detection Statistics")
                        
                        # Count detections by class
                        class_counts = {}
                        for det in detections:
                            class_name = det['class']
                            if class_name in class_counts:
                                class_counts[class_name] += 1
                            else:
                                class_counts[class_name] = 1
                        
                        # Display counts with emojis
                        cols = st.columns(3)
                        col_idx = 0
                        
                        for class_name, count in class_counts.items():
                            emoji = "👤" if class_name == "human" else (
                                   "🔥" if class_name == "fire" else (
                                   "💨" if class_name == "smoke" else (
                                   "🚗" if class_name == "car" else (
                                   "🚴" if class_name == "bike-bicycle" else "🚚"))))
                            
                            with cols[col_idx % 3]:
                                st.metric(f"{emoji} {class_name.capitalize()}", count)
                            col_idx += 1
                        
                        # Check for priority threats
                        if "fire" in class_counts or "smoke" in class_counts:
                            st.error("🚨 **ALERT: Fire Detected!** Potential forest fire detected. Immediate action required.")
                        
                        if "human" in class_counts or "car" in class_counts or "bike-bicycle" in class_counts or "bus-truck" in class_counts:
                            st.warning("⚠️ **Trespassers Detected!** Unauthorized entry detected in monitored area.")
                            
                    except Exception as e:
                        st.error(f"Error during detection: {e}")
                        st.exception(e)
    
    elif st.session_state.detection_input_method == 'video':
        # Video upload
        video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
        if video_file is not None:
            # Save uploaded video to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(video_file.read())
                temp_video_path = tfile.name
            
            # Display video upload success
            st.success("Video uploaded successfully!")
            
            # Process video button
            if st.button("Process Video"):
                with st.spinner("Processing video... This may take a while."):
                    try:
                        # Open video file
                        cap = cv2.VideoCapture(temp_video_path)
                        
                        # Create video writer for output
                        output_path = "output_video.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                        
                        # Create placeholder for video frames
                        video_placeholder = st.empty()
                        status_text = st.empty()
                        
                        # Process frames
                        frame_count = 0
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                                
                            # Process every 5th frame for speed
                            if frame_count % 5 == 0:
                                detections = model.detect(frame)
                                result_frame = model.draw_detections(frame.copy(), detections)
                                
                                # Update preview
                                if frame_count % 15 == 0:  # Update display less frequently
                                    video_placeholder.image(
                                        cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB),
                                        caption="Processing Video",
                                        use_container_width=True
                                    )
                                    progress = min(100, int((frame_count / total_frames) * 100))
                                    status_text.text(f"Processing: {progress}% complete")
                            else:
                                result_frame = frame  # Skip detection on some frames
                                
                            # Write frame to output video
                            out.write(result_frame)
                            frame_count += 1
                            
                        # Release resources
                        cap.release()
                        out.release()
                        
                        # Display completion message
                        st.success("Video processing complete!")
                        
                        # Provide download button for processed video
                        with open(output_path, "rb") as file:
                            st.download_button(
                                label="Download Processed Video",
                                data=file,
                                file_name="forest_surveillance_results.mp4",
                                mime="video/mp4"
                            )
                            
                    except Exception as e:
                        st.error(f"Error processing video: {e}")
                        st.exception(e)
                    finally:
                        # Clean up temp file
                        try:
                            os.unlink(temp_video_path)
                        except:
                            pass
    
    else:  # Camera mode
        # Live camera feed
        st.subheader("Live Camera Detection")
        st.info("Use your webcam to detect objects in real-time")
        
        cam = st.camera_input("Camera Feed")
        
        if cam:
            # Process camera input
            with st.spinner("Processing image..."):
                try:
                    # Convert image
                    image = cv2.imdecode(np.frombuffer(cam.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                    
                    # Run detection
                    detections = model.detect(image)
                    result_image = model.draw_detections(image.copy(), detections)
                    
                    # Display results
                    st.image(
                        cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB),
                        caption="Detection Results",
                        use_container_width=True
                    )
                    
                    # Show detection summary
                    if detections:
                        # Count detections by class
                        class_counts = {}
                        for det in detections:
                            class_name = det['class']
                            if class_name in class_counts:
                                class_counts[class_name] += 1
                            else:
                                class_counts[class_name] = 1
                        
                        # Display as metrics
                        st.subheader("Detection Summary")
                        cols = st.columns(3)
                        for i, (class_name, count) in enumerate(class_counts.items()):
                            with cols[i % 3]:
                                st.metric(class_name.capitalize(), count)
                        
                        # Check for priority threats
                        if "fire" in class_counts or "smoke" in class_counts:
                            st.error("🚨 **ALERT: Fire Detected!** Potential forest fire detected.")
                        
                        if "human" in class_counts:
                            st.warning("⚠️ **Trespasser Detected!** Human presence detected.")
                    else:
                        st.info("No objects detected in frame")
                        
                except Exception as e:
                    st.error(f"Error processing camera feed: {e}")

# Main function
def main():
    # Check which service is selected and render appropriate UI
    if st.session_state.current_service == 'deforestation':
        show_deforestation_detection()
    elif st.session_state.current_service == 'audio':
        show_audio_classification()
    else:  # 'detection'
        show_object_detection()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 10px;">
        <p>Nature Nexus - Forest Surveillance System | 🌳 Protect Natural Ecosystems</p>
        <p><small>Built with Streamlit and PyTorch</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()