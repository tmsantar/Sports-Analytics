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

st.set_page_config(page_title="Roboflow Football Tracker", layout="wide")
st.title("Roboflow Football Tracker")

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

video_source = st.sidebar.radio("Video source", ["Choose sideline video", "Upload video"])


def find_sideline_videos_dir():
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


def get_image_coordinates_component():
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


def get_ffmpeg_exe():
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


def build_tracking_metadata(positions, tracking_lost_frames, total_frames, fps, target, pixels_per_yard):
    df = pd.DataFrame(positions)
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
        summary["Real-world values"] = "Estimates from clicked field calibration"
        summary["Pixels per yard calibration"] = f"{pixels_per_yard:.1f}"
        summary["Estimated total distance"] = f"{df['estimated_distance_yards'].sum():.1f} yd"
        summary["Estimated max speed"] = f"{df['estimated_speed_mph'].max():.1f} mph"
        summary["Estimated avg speed"] = f"{df['estimated_speed_mph'].mean():.1f} mph"

    return df, summary


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
                                tracked_positions = []
                                tracking_lost_frames = []

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
                                        tracked_positions.append({
                                            "frame": frame_num,
                                            "target_id": target["id"],
                                            "target_class": target["class"],
                                            "center_x": x + w / 2,
                                            "center_y": y + h / 2,
                                        })
                                        draw_target(frame, target, (x, y, w, h), class_colors=class_colors)
                                    else:
                                        tracking_lost_frames.append(frame_num)
                                        cv2.putText(frame, "Tracking lost", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
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
                                    file_name="tommy_tracked_video.mp4",
                                    mime="video/mp4",
                                )

                                metadata_df, metadata_summary = build_tracking_metadata(
                                    tracked_positions,
                                    tracking_lost_frames,
                                    total_frames,
                                    fps,
                                    target,
                                    pixels_per_yard,
                                )
                                st.subheader(f"Tracking Data: ID:{target['id']} {target['class']}")
                                if pixels_per_yard:
                                    st.warning(
                                        "Yards and MPH are rough estimates from the clicked field calibration. "
                                        "They are not exact player tracking measurements."
                                    )
                                else:
                                    st.info("Complete field calibration above to add estimated yards and MPH.")
                                st.table(
                                    pd.DataFrame(
                                        [{"Metric": key, "Value": value} for key, value in metadata_summary.items()]
                                    )
                                )
                                st.dataframe(metadata_df, use_container_width=True)
                                st.download_button(
                                    "Save tracking data CSV",
                                    data=metadata_df.to_csv(index=False).encode("utf-8"),
                                    file_name=f"tommy_tracking_id_{target['id']}_metadata.csv",
                                    mime="text/csv",
                                )
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
6. Use field calibration if you want rough yards/MPH estimates.
7. Click "Run video tracking" to generate the tracked video.
""")
