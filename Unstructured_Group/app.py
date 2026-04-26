import streamlit as st
import cv2
import math
import numpy as np
import pandas as pd
from roboflow import Roboflow
import os
import shutil
import subprocess
import sys
import tempfile
from PIL import Image

# Configure the Streamlit app page
st.set_page_config(page_title="Roboflow Football Player Tracker", layout="wide")
st.title("Roboflow Football Player Tracker")

# Roboflow project configuration constants
ROBOFLOW_WORKSPACE = "tommys-workspace-vmucs"
ROBOFLOW_PROJECT = "rishi-fohsb-fdsoi"
ROBOFLOW_VERSION = 1
ROBOFLOW_MODEL_ID = f"{ROBOFLOW_PROJECT}/{ROBOFLOW_VERSION}"
DEFAULT_ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "YYrmkzwSLYTs1DUgRxed")

# Tracking configuration constants
MAX_DETECTION_FRAMES = 90  # Maximum frames to scan for QB detection
MIN_BBOX_AREA = 500  # Minimum bounding box area to avoid noise detections
MAX_PLAUSIBLE_SPEED_MPH = 30.0  # Maximum realistic speed for filtering unreliable estimates
MAX_TRACKER_CENTER_JUMP_FRAC = 0.10  # Reject jumps larger than 10% of frame diagonal
MIN_TRACKER_AREA_RATIO = 0.35  # Reject boxes that shrink too abruptly
MAX_TRACKER_AREA_RATIO = 2.75  # Reject boxes that grow too abruptly
MAX_TRACKER_OFF_FRAME_FRAC = 0.25  # Reject boxes that are mostly off-screen

# Sidebar configuration section
st.sidebar.header("Configuration")

# API key setup - uses saved key by default
api_key = "YYrmkzwSLYTs1DUgRxed"
st.sidebar.success("Using saved Roboflow API key")
if st.sidebar.checkbox("Use a different Roboflow API key"):
    api_key = st.sidebar.text_input("Roboflow API Key", type="password")

# Tracking mode selection
track_mode = st.sidebar.radio("Track mode", ["Track QB", "Track Specific ID"])
track_id = ""
if track_mode == "Track Specific ID":
    # Get preview detections from session state to populate dropdown
    preview_detections = st.session_state.get("preview_detections", [])
    preview_ids = [str(det["id"]) for det in preview_detections]
    if preview_ids:
        track_id = st.sidebar.selectbox("Target ID", preview_ids)
    else:
        track_id = st.sidebar.text_input("Target ID (show ID preview first)")

# Video source selection
video_source = st.sidebar.radio("Video source", ["Choose sideline video", "Upload video"])


def find_sideline_videos_dir():
    """
    Find the directory containing sideline videos by checking multiple possible paths.
    This handles different environments (local, Colab, etc.) and directory structures.
    """
    app_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    candidates = [
        os.path.join(app_dir, "data", "train"),
        os.path.join(cwd, "data", "train"),
        os.path.join(cwd, "Unstructured_Group", "data", "train"),
        os.path.join(cwd, "Unstructured Group", "data", "train"),
        "/content/drive/MyDrive/Unstructured_Group/data/train",
        "/content/drive/MyDrive/Unstructured Group/data/train",
        "/content/drive/MyDrive/Unstructured Group Folder/NFL 1st and Future - Impact Detection - Data/nfl-impact-detection (Unzipped Files)/train",
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return None


sideline_videos_dir = find_sideline_videos_dir()

if video_source == "Choose sideline video":
    if sideline_videos_dir:
        video_files = sorted(
            f for f in os.listdir(sideline_videos_dir)
            if f.lower().endswith(".mp4") and "sideline" in f.lower()
        )
        if video_files:
            video_labels = {
                f"Sample {i + 1}: {name.replace('_Sideline.mp4', ' Sideline')}" : name
                for i, name in enumerate(video_files)
            }
            selected_label = st.sidebar.selectbox("Select sample video", list(video_labels.keys()))
            selected_video = video_labels[selected_label]
            video_path = os.path.join(sideline_videos_dir, selected_video)
        else:
            st.sidebar.error("No sideline videos found in the directory.")
            video_path = None
    else:
        st.sidebar.error("Sideline videos directory not found.")
        video_path = None
else:
    uploaded_file = st.sidebar.file_uploader("Upload video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
    else:
        video_path = None

confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5)
known_calibration_yards = st.sidebar.number_input(
    "Known calibration distance (yards)",
    min_value=1.0,
    max_value=100.0,
    value=5.0,
    step=1.0,
)
pixels_per_yard = st.session_state.get("pixels_per_yard")
if pixels_per_yard:
    st.sidebar.success(f"Calibrated scale: {pixels_per_yard:.1f} px/yard")
else:
    st.sidebar.info("Calibrate on the video frame to estimate yards and MPH.")
show_frame = st.sidebar.button("Show labeled preview")
run_tracking = st.sidebar.button("Run video tracking")


def create_tracker():
    """
    Create and configure an OpenCV CSRT tracker for object tracking.
    CSRT (Channel and Spatial Reliability Tracking) is robust for tracking objects
    in video sequences, especially in sports analytics where objects may change
    appearance due to motion, lighting, or occlusion.
    """
    try:
        return cv2.TrackerCSRT_create()
    except AttributeError:
        return cv2.legacy.TrackerCSRT_create()


def get_frame(video_path, frame_index=0):
    """
    Extract a specific frame from a video file.
    This is used to get the first frame for preview and detection purposes.

    Args:
        video_path (str): Path to the video file
        frame_index (int): Frame number to extract (0-based)

    Returns:
        numpy.ndarray: The extracted frame as a BGR image, or None if failed
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > frame_index:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def get_image_coordinates_component():
    """
    Dynamically import and install the streamlit-image-coordinates component.
    This component allows users to click on images to get pixel coordinates,
    which is used for field calibration to estimate yards and MPH.

    Returns:
        function: The streamlit_image_coordinates function, or None if unavailable
    """
    try:
        from streamlit_image_coordinates import streamlit_image_coordinates
        return streamlit_image_coordinates
    except Exception:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "streamlit-image-coordinates"],
                check=False,
                capture_output=True,
                text=True,
            )
            from streamlit_image_coordinates import streamlit_image_coordinates
            return streamlit_image_coordinates
        except Exception:
            return None


def show_field_calibration(video_path, known_yards):
    """
    Display an interactive field calibration interface.
    Users click two points on the field that are a known distance apart
    (like yard lines) to establish a pixel-to-yard conversion ratio.
    This enables estimating yards traveled and MPH calculations.

    Args:
        video_path (str): Path to the video file for calibration
        known_yards (float): Known distance in yards between calibration points
    """
    with st.expander("Field Calibration for Estimated Yards/MPH", expanded=False):
        st.caption(
            "Click two points on the frame that are a known distance apart, such as adjacent yard lines. "
            "The resulting yards and MPH are still estimates because camera perspective changes across the field."
        )

        if not video_path:
            st.info("Select or upload a video before calibrating.")
            return

        frame = get_frame(video_path, frame_index=0)
        if frame is None:
            st.warning("Could not load a calibration frame from this video.")
            return

        component = get_image_coordinates_component()
        if component is None:
            st.warning("Install streamlit-image-coordinates to click calibration points.")
            st.code("pip install streamlit-image-coordinates")
            return

        if st.button("Reset calibration points"):
            st.session_state["calibration_points"] = []
            st.session_state.pop("pixels_per_yard", None)
            st.session_state.pop("last_calibration_click", None)

        points = st.session_state.get("calibration_points", [])
        display_width = min(900, frame.shape[1])
        scale = display_width / frame.shape[1]
        display_height = int(frame.shape[0] * scale)
        display_frame = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_AREA)

        for point in points:
            x_display = int(point[0] * scale)
            y_display = int(point[1] * scale)
            cv2.circle(display_frame, (x_display, y_display), 7, (0, 255, 255), -1)

        if len(points) == 2:
            p1 = (int(points[0][0] * scale), int(points[0][1] * scale))
            p2 = (int(points[1][0] * scale), int(points[1][1] * scale))
            cv2.line(display_frame, p1, p2, (0, 255, 255), 2)

        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        click = component(Image.fromarray(rgb_frame), key="field_calibration")
        if click:
            original_x = click["x"] / scale
            original_y = click["y"] / scale
            rounded_click = (round(original_x, 1), round(original_y, 1))
            if st.session_state.get("last_calibration_click") != rounded_click and len(points) < 2:
                points.append(rounded_click)
                st.session_state["calibration_points"] = points
                st.session_state["last_calibration_click"] = rounded_click
                st.rerun()

        if len(points) < 2:
            st.info(f"Calibration points selected: {len(points)}/2")
            return

        pixel_distance = math.dist(points[0], points[1])
        if known_yards <= 0:
            st.warning("Known calibration distance must be greater than zero.")
            return

        st.session_state["pixels_per_yard"] = pixel_distance / known_yards
        st.success(
            f"Scale calibrated from {pixel_distance:.1f} px over {known_yards:.1f} yd: "
            f"{st.session_state['pixels_per_yard']:.1f} px/yard"
        )


def get_class_colors(detections):
    """
    Generate consistent colors for different detection classes.
    Each unique class gets assigned a color from a predefined palette
    to make visualizations consistent across frames.

    Args:
        detections (list): List of detection dictionaries with 'class' keys

    Returns:
        dict: Mapping of class names to BGR color tuples
    """
    class_colors = {}
    palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 165, 0), (128, 0, 128), (0, 255, 255)
    ]
    for det in detections:
        cls = det["class"]
        if cls not in class_colors:
            class_colors[cls] = palette[len(class_colors) % len(palette)]
    return class_colors


def annotate_detections(frame, detections):
    """
    Draw bounding boxes and labels on a frame for detected objects.
    Each detection gets a colored rectangle and text label showing
    the class, ID, and confidence score.

    Args:
        frame (numpy.ndarray): The image frame to annotate
        detections (list): List of detection dictionaries with bbox, class, id, conf

    Returns:
        tuple: (annotated_frame, class_colors_dict) - the annotated frame and color mapping
    """
    display = frame.copy()
    class_colors = get_class_colors(detections)
    for det in detections:
        cls = det["class"]
        color = class_colors[cls]
        x, y, w, h = det["bbox"]
        cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
        label = f"ID:{det['id']} {cls} {det['conf']:.2f}"
        cv2.putText(display, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return display, class_colors


def image_to_png_bytes(rgb_image):
    """Encode an RGB image array as PNG bytes for download."""
    success, encoded = cv2.imencode(".png", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    return encoded.tobytes() if success else None


def build_detections(results, min_conf):
    """
    Parse Roboflow inference results into a standardized detection format.
    Filters detections by confidence and minimum bounding box area.

    Args:
        results (dict): Raw inference results from Roboflow API
        min_conf (float): Minimum confidence threshold (0-1)

    Returns:
        list: List of detection dictionaries with standardized format
    """
    detections = []
    for i, pred in enumerate(results.get("predictions", [])):
        confidence = float(pred.get("confidence", 0))
        if confidence < min_conf:
            continue
        width = float(pred.get("width", 0))
        height = float(pred.get("height", 0))
        x = max(0, int(float(pred.get("x", 0)) - width / 2))
        y = max(0, int(float(pred.get("y", 0)) - height / 2))
        w = int(width)
        h = int(height)
        if w * h < MIN_BBOX_AREA:
            continue
        detections.append({
            "id": i,
            "class": pred.get("class", pred.get("class_name", "unknown")),
            "conf": confidence,
            "bbox": (x, y, w, h),
        })
    return detections


def load_model(api_key):
    """
    Load the Roboflow model for player detection.
    Initializes the Roboflow API client and loads the specified model version.

    Args:
        api_key (str): Roboflow API key for authentication

    Returns:
        Roboflow model object: Ready-to-use inference model
    """
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    model = project.version(ROBOFLOW_VERSION).model
    return model


def predict_frame(model, frame, confidence_threshold):
    """
    Run inference on a single frame using the Roboflow model.
    Temporarily saves the frame as an image file for the API call.

    Args:
        model: Roboflow model object
        frame (numpy.ndarray): OpenCV frame to analyze
        confidence_threshold (float): Minimum confidence for detections (0-1)

    Returns:
        dict: JSON response from Roboflow API with predictions
    """
    frame_path = tempfile.mktemp(suffix=".jpg")
    cv2.imwrite(frame_path, frame)
    try:
        return model.predict(
            frame_path,
            confidence=int(confidence_threshold * 100),
            overlap=30,
        ).json()
    finally:
        if os.path.exists(frame_path):
            os.remove(frame_path)


def select_target(detections, track_mode, track_id):
    """
    Select which detected object to track based on the tracking mode.
    For QB tracking, selects the highest confidence QB detection.
    For specific ID tracking, finds the detection with the matching ID.

    Args:
        detections (list): List of detection dictionaries
        track_mode (str): Either "Track QB" or "Track Specific ID"
        track_id (str): Target ID for specific tracking mode

    Returns:
        dict or None: The selected detection to track, or None if not found
    """
    if track_mode == "Track QB":
        qb_candidates = [d for d in detections if d["class"].lower() == "qb"]
        return max(qb_candidates, key=lambda d: d["conf"]) if qb_candidates else None

    if track_id is None or track_id == "":
        return None
    match = next((d for d in detections if str(d["id"]) == str(track_id)), None)
    return match


def show_preview_for_choice(model, video_path, track_mode, confidence_threshold):
    """
    Display a preview of the first frame with detections to help users choose tracking targets.
    Shows annotated image and detection table, with different displays based on tracking mode.

    Args:
        model: Roboflow model for inference
        video_path (str): Path to the video file
        track_mode (str): Current tracking mode ("Track QB" or "Track Specific ID")
        confidence_threshold (float): Minimum confidence for detections
    """
    frame = get_frame(video_path, frame_index=0)
    if frame is None:
        st.error("Could not open video file or read frame.")
        return None
    results = predict_frame(model, frame, confidence_threshold)
    detections = build_detections(results, confidence_threshold)
    show_ids = track_mode == "Track Specific ID"
    annotated_bgr, class_colors = annotate_detections(frame, detections)
    annotated = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    caption = "ID preview of the first frame" if show_ids else "Labeled position preview of the first frame"
    st.image(annotated, channels="RGB", caption=caption)
    png_bytes = image_to_png_bytes(annotated)
    if png_bytes:
        st.download_button(
            "Save labeled image",
            data=png_bytes,
            file_name="labeled_preview.png",
            mime="image/png",
        )
    show_detected_table(detections, show_ids=show_ids)
    if show_ids:
        st.write("Use the ID list above to select a Target ID for specific tracking.")
    st.session_state["preview_detections"] = detections
    st.session_state["class_colors"] = class_colors
    return detections


def show_detected_table(detections, show_ids=False):
    """
    Display a table of detected objects with their properties.
    Shows different information based on whether IDs are needed for selection.

    Args:
        detections (list): List of detection dictionaries
        show_ids (bool): Whether to show IDs for selection (affects display format)
    """
    if not detections:
        st.warning("No detections found on the preview frame.")
        return
    if show_ids:
        df = [
            {
                "ID": det["id"],
                "Class": det["class"],
                "Confidence": round(det["conf"], 3),
                "BBox": det["bbox"],
            }
            for det in detections
        ]
        st.write("### Detected players")
        st.code("\\n".join(
            f"ID {det['id']:2d} — {det['class']:15s} conf: {det['conf']:.2f}  bbox: {det['bbox']}"
            for det in detections
        ))
    else:
        df = [
            {
                "Position": det["class"],
                "Confidence": round(det["conf"], 3),
                "BBox": det["bbox"],
            }
            for det in detections
        ]
        st.write("### Labeled positions on preview frame")
    st.table(df)


def draw_target(frame, target, bbox=None, class_colors=None):
    """
    Draw a highlighted bounding box around the tracking target.
    Uses thicker lines and larger text to distinguish the tracked object.

    Args:
        frame (numpy.ndarray): Frame to draw on
        target (dict): Target detection dictionary
        bbox (tuple): Optional custom bounding box coordinates
        class_colors (dict): Optional color mapping for classes

    Returns:
        numpy.ndarray: Frame with target highlighted
    """
    x, y, w, h = [int(v) for v in (bbox or target["bbox"])]
    if class_colors is None:
        class_colors = get_class_colors([target])
    color = class_colors.get(target["class"], (255, 0, 0))
    label = f"ID:{target['id']} {target['class']}"
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
    cv2.putText(frame, label, (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame


def bbox_center(bbox):
    """Return center point for a bounding box."""
    x, y, w, h = bbox
    return (x + w / 2, y + h / 2)


def bbox_area(bbox):
    """Return nonzero area for a bounding box."""
    return max(1, bbox[2] * bbox[3])


def off_frame_fraction(bbox, frame_shape):
    """Calculate what fraction of a bbox is outside the visible frame."""
    frame_h, frame_w = frame_shape[:2]
    x, y, w, h = bbox
    x2 = x + w
    y2 = y + h
    inside_w = max(0, min(x2, frame_w) - max(x, 0))
    inside_h = max(0, min(y2, frame_h) - max(y, 0))
    inside_area = inside_w * inside_h
    return 1 - (inside_area / bbox_area(bbox))


def validate_tracker_bbox(bbox, previous_bbox, frame_shape):
    """
    Reject tracker updates that are technically successful but visually implausible.
    This prevents the app from drawing a confident box after the tracker drifts.
    """
    if bbox is None:
        return False, "tracker_failed"

    x, y, w, h = bbox
    if w <= 0 or h <= 0 or w * h < MIN_BBOX_AREA:
        return False, "invalid_box_size"

    frame_h, frame_w = frame_shape[:2]
    frame_diag = math.sqrt(frame_w ** 2 + frame_h ** 2)
    max_center_jump = max(80, frame_diag * MAX_TRACKER_CENTER_JUMP_FRAC)
    center_jump = math.dist(bbox_center(bbox), bbox_center(previous_bbox))
    if center_jump > max_center_jump:
        return False, f"large_jump_{center_jump:.0f}px"

    area_ratio = bbox_area(bbox) / bbox_area(previous_bbox)
    if area_ratio < MIN_TRACKER_AREA_RATIO or area_ratio > MAX_TRACKER_AREA_RATIO:
        return False, f"area_change_{area_ratio:.2f}x"

    offscreen = off_frame_fraction(bbox, frame_shape)
    if offscreen > MAX_TRACKER_OFF_FRAME_FRAC:
        return False, f"off_screen_{offscreen:.0%}"

    return True, "tracking"


def get_ffmpeg_exe():
    """
    Find the FFmpeg executable for video processing.
    Tries system PATH first, then imageio-ffmpeg package.

    Returns:
        str or None: Path to FFmpeg executable, or None if not found
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        try:
            subprocess.run(
                [os.sys.executable, "-m", "pip", "install", "imageio-ffmpeg"],
                check=False,
                capture_output=True,
                text=True,
            )
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None


def browser_ready_video(input_path):
    """
    Convert video to browser-compatible MP4 format for inline playback.
    Uses FFmpeg to re-encode with H.264 codec and faststart flags for web streaming.

    Args:
        input_path (str): Path to input video file

    Returns:
        str: Path to converted video file, or original path if conversion fails
    """
    ffmpeg = get_ffmpeg_exe()
    if not ffmpeg:
        return input_path
    output_path = tempfile.mktemp(suffix=".mp4")
    cmd = [
        ffmpeg,
        "-y",
        "-i", input_path,
        "-vcodec", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path
    st.warning("The browser-ready video conversion failed, so the raw MP4 may not play inline.")
    if result.stderr:
        st.caption(result.stderr[-800:])
    return input_path


def build_tracking_metadata(positions, tracking_lost_frames, total_frames, fps, target, pixels_per_yard, tracking_lost_reasons=None):
    """
    Calculate and compile tracking analytics and metadata.
    Computes distance traveled, speed metrics, and converts to real-world units if calibrated.

    Args:
        positions (list): List of (frame, x, y) position tuples
        tracking_lost_frames (list): Frames where tracking was lost
        total_frames (int): Total number of frames in video
        fps (float): Frames per second of the video
        target (dict): Target detection information
        pixels_per_yard (float): Pixels per yard conversion factor, or None

    Returns:
        dict: Dictionary containing tracking statistics and metadata
    """
    df = pd.DataFrame(positions)
    tracking_lost_reasons = tracking_lost_reasons or []
    target_label = f"ID:{target['id']} {target['class']}"
    has_calibration = pixels_per_yard is not None and pixels_per_yard > 0
    if df.empty:
        summary = {
            "Tracked target": target_label,
            "Real-world values": "Unavailable until field calibration is completed",
            "Total frames": total_frames,
            "Frames tracked": 0,
            "Frames lost": len(tracking_lost_frames),
            "Total distance": "0.0 px",
            "Max speed": "0.0 px/sec",
            "Avg speed": "0.0 px/sec",
            "First position": "N/A",
            "Last position": "N/A",
        }
        if has_calibration:
            summary["Pixels per yard calibration"] = f"{pixels_per_yard:.1f}"
            summary["Estimated total distance"] = "0.0 yd"
            summary["Estimated max speed"] = "0.0 mph"
            summary["Estimated avg speed"] = "0.0 mph"
        return df, summary

    df["distance"] = 0.0
    for i in range(1, len(df)):
        dx = df.loc[i, "center_x"] - df.loc[i - 1, "center_x"]
        dy = df.loc[i, "center_y"] - df.loc[i - 1, "center_y"]
        df.loc[i, "distance"] = math.sqrt(dx ** 2 + dy ** 2)

    df["speed_px_per_sec"] = df["distance"] * fps

    total_distance = df["distance"].sum()
    max_speed = df["speed_px_per_sec"].max()
    avg_speed = df["speed_px_per_sec"].mean()

    summary = {
        "Tracked target": target_label,
        "Real-world values": "Unavailable until field calibration is completed",
        "Total frames": int(total_frames),
        "Frames tracked": int(len(df)),
        "Frames lost": int(len(tracking_lost_frames)),
        "Total distance": f"{total_distance:.1f} px",
        "Max speed": f"{max_speed:.1f} px/sec",
        "Avg speed": f"{avg_speed:.1f} px/sec",
        "First position": f"({df.iloc[0]['center_x']:.0f}, {df.iloc[0]['center_y']:.0f})",
        "Last position": f"({df.iloc[-1]['center_x']:.0f}, {df.iloc[-1]['center_y']:.0f})",
    }

    if has_calibration:
        df["estimated_distance_yards"] = df["distance"] / pixels_per_yard
        df["estimated_speed_yards_per_sec"] = df["speed_px_per_sec"] / pixels_per_yard
        df["estimated_speed_mph"] = df["estimated_speed_yards_per_sec"] * 2.04545
        df["estimated_speed_flag"] = np.where(
            df["estimated_speed_mph"] > MAX_PLAUSIBLE_SPEED_MPH,
            "unreliable",
            "ok",
        )
        reliable_speed_df = df[df["estimated_speed_flag"] == "ok"]
        max_estimated_speed = df["estimated_speed_mph"].max()
        avg_estimated_speed = reliable_speed_df["estimated_speed_mph"].mean() if not reliable_speed_df.empty else np.nan

        if max_estimated_speed > MAX_PLAUSIBLE_SPEED_MPH:
            summary["Real-world values"] = (
                "Calibration produced implausible speed spikes; MPH should be treated as unreliable"
            )
        else:
            summary["Real-world values"] = "Estimates from clicked field calibration"
        summary["Pixels per yard calibration"] = f"{pixels_per_yard:.1f}"
        summary["Estimated total distance"] = f"{df['estimated_distance_yards'].sum():.1f} yd"
        summary["Estimated max speed"] = (
            f"Unreliable ({max_estimated_speed:.1f} mph spike)"
            if max_estimated_speed > MAX_PLAUSIBLE_SPEED_MPH
            else f"{max_estimated_speed:.1f} mph"
        )
        summary["Estimated avg speed"] = (
            "Unreliable"
            if np.isnan(avg_estimated_speed)
            else f"{avg_estimated_speed:.1f} mph from plausible frames"
        )

    return df, summary


def build_lost_tracking_table(tracking_lost_reasons):
    """Build a table of frames where tracking was rejected or failed."""
    if not tracking_lost_reasons:
        return pd.DataFrame(columns=["frame", "reason"])
    return pd.DataFrame(tracking_lost_reasons)


show_field_calibration(video_path, known_calibration_yards)
pixels_per_yard = st.session_state.get("pixels_per_yard")


if api_key and video_path:
    model = load_model(api_key)

    if show_frame:
        frame = get_frame(video_path, frame_index=0)
        if frame is None:
            st.error("Could not open video file or read frame.")
        else:
            results = predict_frame(model, frame, confidence_threshold)
            detections = build_detections(results, confidence_threshold)
            show_ids = track_mode == "Track Specific ID"
            annotated_bgr, class_colors = annotate_detections(frame, detections)
            annotated = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            caption = "ID preview of the first frame" if show_ids else "Labeled position preview of the first frame"
            st.image(annotated, channels="RGB", caption=caption)
            png_bytes = image_to_png_bytes(annotated)
            if png_bytes:
                st.download_button(
                    "Save labeled image",
                    data=png_bytes,
                    file_name="labeled_preview.png",
                    mime="image/png",
                )
            show_detected_table(detections, show_ids=show_ids)
            if show_ids:
                st.write("Use the ID list above to select a Target ID for specific tracking.")
            st.session_state["preview_detections"] = detections
            st.session_state["class_colors"] = class_colors

    if run_tracking:
        if track_mode == "Track Specific ID" and not track_id:
            st.error("Enter a Target ID before running specific ID tracking.")
            show_preview_for_choice(model, video_path, track_mode, confidence_threshold)
        else:
            frame = get_frame(video_path, frame_index=0)
            if frame is None:
                st.error("Could not open video file or read frame.")
            else:
                results = predict_frame(model, frame, confidence_threshold)
                detections = build_detections(results, confidence_threshold)
                class_colors = get_class_colors(detections)
                target = select_target(detections, track_mode, track_id)
                if target is None:
                    st.error("Could not identify the requested target in the preview frame.")
                else:
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        st.error("Could not open video file.")
                    else:
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS) or 30
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        if width <= 0 or height <= 0:
                            cap.release()
                            st.error("Could not read the video dimensions.")
                            st.stop()
                        raw_output_path = tempfile.mktemp(suffix='.mp4')
                        out = cv2.VideoWriter(raw_output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

                        ret, first_frame = cap.read()
                        if not ret:
                            st.error("Could not read video frames.")
                            cap.release()
                        else:
                            if not out.isOpened():
                                st.error("Could not create output video file.")
                                cap.release()
                            else:
                                tracker = create_tracker()
                                tracker.init(first_frame, target["bbox"])
                                last_good_bbox = target["bbox"]
                                tracked_positions = []
                                tracking_lost_frames = []
                                tracking_lost_reasons = []

                                x, y, w, h = target["bbox"]
                                tracked_positions.append({
                                    "frame": 0,
                                    "target_id": target["id"],
                                    "target_class": target["class"],
                                    "center_x": x + w / 2,
                                    "center_y": y + h / 2,
                                })
                                out.write(draw_target(first_frame.copy(), target, class_colors=class_colors))

                                frame_num = 1
                                while True:
                                    ret, frame = cap.read()
                                    if not ret:
                                        break
                                    success, bbox = tracker.update(frame)
                                    if success:
                                        x, y, w, h = [int(v) for v in bbox]
                                        validated_bbox = (x, y, w, h)
                                        bbox_is_valid, lost_reason = validate_tracker_bbox(
                                            validated_bbox,
                                            last_good_bbox,
                                            frame.shape,
                                        )
                                        if bbox_is_valid:
                                            last_good_bbox = validated_bbox
                                            tracked_positions.append({
                                                "frame": frame_num,
                                                "target_id": target["id"],
                                                "target_class": target["class"],
                                                "center_x": x + w / 2,
                                                "center_y": y + h / 2,
                                            })
                                            draw_target(frame, target, validated_bbox, class_colors=class_colors)
                                        else:
                                            tracking_lost_frames.append(frame_num)
                                            tracking_lost_reasons.append({
                                                "frame": frame_num,
                                                "reason": lost_reason,
                                            })
                                            cv2.putText(
                                                frame,
                                                f"Tracking lost: {lost_reason}",
                                                (50, 50),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.9,
                                                (0, 0, 255),
                                                2,
                                            )
                                    else:
                                        tracking_lost_frames.append(frame_num)
                                        tracking_lost_reasons.append({
                                            "frame": frame_num,
                                            "reason": "tracker_failed",
                                        })
                                        cv2.putText(frame, "Tracking lost: tracker_failed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                                    out.write(frame)
                                    frame_num += 1

                                cap.release()
                                out.release()
                                if not os.path.exists(raw_output_path) or os.path.getsize(raw_output_path) == 0:
                                    st.error("The tracked video was created but has no playable frames.")
                                    st.stop()
                                output_path = browser_ready_video(raw_output_path)
                                st.subheader("Tracked Video")
                                st.caption(f"Output file: {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.2f} MB)")
                                with open(output_path, "rb") as video_file:
                                    video_bytes = video_file.read()
                                st.video(output_path)
                                st.download_button(
                                    "Save tracked video",
                                    data=video_bytes,
                                    file_name="tracked_video.mp4",
                                    mime="video/mp4",
                                )

                                metadata_df, metadata_summary = build_tracking_metadata(
                                    tracked_positions,
                                    tracking_lost_frames,
                                    total_frames,
                                    fps,
                                    target,
                                    pixels_per_yard,
                                    tracking_lost_reasons,
                                )
                                lost_tracking_df = build_lost_tracking_table(tracking_lost_reasons)
                                st.subheader(f"Tracking Data: ID:{target['id']} {target['class']}")
                                if pixels_per_yard:
                                    st.warning(
                                        "Yards and MPH are rough estimates from the clicked field calibration. "
                                        "They are not exact player tracking measurements. "
                                        "Speed spikes above 30 mph are marked unreliable."
                                    )
                                else:
                                    st.info("Complete field calibration above to add estimated yards and MPH.")
                                st.table(
                                    pd.DataFrame(
                                        [{"Metric": key, "Value": value} for key, value in metadata_summary.items()]
                                    )
                                )
                                st.dataframe(metadata_df, use_container_width=True)
                                if not lost_tracking_df.empty:
                                    st.write("### Rejected / Lost Tracking Frames")
                                    st.dataframe(lost_tracking_df, use_container_width=True)
                                st.download_button(
                                    "Save tracking data CSV",
                                    data=metadata_df.to_csv(index=False).encode("utf-8"),
                                    file_name=f"tracking_id_{target['id']}_metadata.csv",
                                    mime="text/csv",
                                )
                                if not lost_tracking_df.empty:
                                    st.download_button(
                                        "Save lost tracking frames CSV",
                                        data=lost_tracking_df.to_csv(index=False).encode("utf-8"),
                                        file_name=f"tracking_id_{target['id']}_lost_frames.csv",
                                        mime="text/csv",
                                    )
                                st.success("Video tracking completed!")
else:
    if run_tracking or show_frame:
        st.error("Please select or upload a video.")

st.markdown("""
## Instructions
1. The app uses a saved Roboflow API key by default.
2. Upload a video or choose one from sideline videos.
3. Click "Show labeled preview" to see position labels and ID labels.
4. Select "Track QB" or "Track Specific ID".
5. If using specific ID, type the ID shown in the Detected players list.
6. Use field calibration if you want rough yards/MPH estimates.
7. Click "Run video tracking" to generate the tracked video.

**Testing note:** This app is still being tested and improved. The tracking, calibration, and speed estimates are experimental and still have a long way to go.
""")
