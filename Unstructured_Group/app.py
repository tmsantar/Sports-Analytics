import streamlit as st
import cv2
import numpy as np
from roboflow import Roboflow
import os
import shutil
import subprocess
import tempfile

st.set_page_config(page_title="Roboflow Football Tracker", layout="wide")
st.title("Tommy Roboflow Football Tracker")

ROBOFLOW_WORKSPACE = "tommys-workspace-vmucs"
ROBOFLOW_PROJECT = "rishi-fohsb-fdsoi"
ROBOFLOW_VERSION = 1
ROBOFLOW_MODEL_ID = f"{ROBOFLOW_PROJECT}/{ROBOFLOW_VERSION}"
DEFAULT_ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "YYrmkzwSLYTs1DUgRxed")
MAX_DETECTION_FRAMES = 90
MIN_BBOX_AREA = 500


st.sidebar.header("Configuration")
api_key = "YYrmkzwSLYTs1DUgRxed"
st.sidebar.success("Using saved Roboflow API key")
if st.sidebar.checkbox("Use a different Roboflow API key"):
    api_key = st.sidebar.text_input("Roboflow API Key", type="password")
track_mode = st.sidebar.radio("Track mode", ["Track QB", "Track Specific ID"])
track_id = ""
if track_mode == "Track Specific ID":
    preview_detections = st.session_state.get("preview_detections", [])
    preview_ids = [str(det["id"]) for det in preview_detections]
    if preview_ids:
        track_id = st.sidebar.selectbox("Target ID", preview_ids)
    else:
        track_id = st.sidebar.text_input("Target ID (show ID preview first)")

video_source = st.sidebar.radio("Video source", ["Upload video", "Choose sideline video"])
sideline_videos_dir = "/content/drive/MyDrive/Unstructured Group Folder/NFL 1st and Future - Impact Detection - Data/nfl-impact-detection (Unzipped Files)/train/"

if video_source == "Choose sideline video":
    if os.path.exists(sideline_videos_dir):
        video_files = [f for f in os.listdir(sideline_videos_dir) if f.endswith('.mp4') and 'Sideline' in f]
        if video_files:
            selected_video = st.sidebar.selectbox("Select sideline video", video_files)
            video_path = os.path.join(sideline_videos_dir, selected_video)
        else:
            st.sidebar.error("No sideline videos found in the directory.")
            video_path = None
    else:
        st.sidebar.error("Sideline videos directory not found. Please check the path.")
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
show_frame = st.sidebar.button("Show labeled preview")
run_tracking = st.sidebar.button("Run video tracking")


def create_tracker():
    try:
        return cv2.TrackerCSRT_create()
    except AttributeError:
        return cv2.legacy.TrackerCSRT_create()


def get_frame(video_path, frame_index=0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > frame_index:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def get_class_colors(detections):
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


def build_detections(results, min_conf):
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
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    model = project.version(ROBOFLOW_VERSION).model
    return model


def predict_frame(model, frame, confidence_threshold):
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
    if track_mode == "Track QB":
        qb_candidates = [d for d in detections if d["class"].lower() == "qb"]
        return max(qb_candidates, key=lambda d: d["conf"]) if qb_candidates else None

    if track_id is None or track_id == "":
        return None
    match = next((d for d in detections if str(d["id"]) == str(track_id)), None)
    return match


def show_preview_for_choice(model, video_path, track_mode, confidence_threshold):
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
    show_detected_table(detections, show_ids=show_ids)
    if show_ids:
        st.write("Use the ID list above to select a Target ID for specific tracking.")
    st.session_state["preview_detections"] = detections
    st.session_state["class_colors"] = class_colors
    return detections


def show_detected_table(detections, show_ids=False):
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
    x, y, w, h = [int(v) for v in (bbox or target["bbox"])]
    if class_colors is None:
        class_colors = get_class_colors([target])
    color = class_colors.get(target["class"], (255, 0, 0))
    label = f"ID:{target['id']} {target['class']}"
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
    cv2.putText(frame, label, (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame


def make_video_writer(width, height, fps):
    fps = fps if fps and fps > 0 else 30
    if shutil.which("ffmpeg"):
        raw_output_path = tempfile.mktemp(suffix=".avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    else:
        raw_output_path = tempfile.mktemp(suffix=".mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))
    return raw_output_path, out


def browser_ready_video(input_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return input_path
    output_path = tempfile.mktemp(suffix=".mp4")
    cmd = [
        ffmpeg,
        "-y",
        "-i", input_path,
        "-vcodec", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path
    st.warning("Could not convert the video to browser-ready MP4. Showing the raw output instead.")
    if result.stderr:
        st.caption(result.stderr[-800:])
    return input_path


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
                        if width <= 0 or height <= 0:
                            cap.release()
                            st.error("Could not read the video dimensions.")
                            st.stop()
                        raw_output_path, out = make_video_writer(width, height, fps)

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
                                out.write(draw_target(first_frame.copy(), target, class_colors=class_colors))

                                while True:
                                    ret, frame = cap.read()
                                    if not ret:
                                        break
                                    success, bbox = tracker.update(frame)
                                    if success:
                                        draw_target(frame, target, bbox, class_colors=class_colors)
                                    else:
                                        cv2.putText(frame, "Tracking lost", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                                    out.write(frame)

                                cap.release()
                                out.release()
                                if not os.path.exists(raw_output_path) or os.path.getsize(raw_output_path) == 0:
                                    st.error("The tracked video was created but has no playable frames.")
                                    st.stop()
                                output_path = browser_ready_video(raw_output_path)
                                st.subheader("Tracked Video")
                                with open(output_path, "rb") as video_file:
                                    st.video(video_file.read(), format="video/mp4")
                                st.success("Video tracking completed!")
else:
    if run_tracking or show_frame:
        st.error("Please select or upload a video.")

st.markdown("""
## Instructions
1. The app uses Tommy's saved Roboflow API key by default.
2. Upload a video or choose one from sideline videos.
3. Click "Show labeled preview" to see position labels for QB mode or ID labels for specific ID mode.
4. Select "Track QB" or "Track Specific ID".
5. If using specific ID, type the ID shown in the Detected players list.
6. Click "Run video tracking" to generate the tracked video.
""")
