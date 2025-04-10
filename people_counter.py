import cv2
import threading
import time
import numpy as np
from ultralytics import YOLO
from flask import redirect, url_for
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

# Global vars
VIDEO_OPTIONS = {
    "Video 1 (RPTRA A)": "sample/test-1.mp4",
    "Video 2 (RPTRA B)": "sample/test-22.mp4",
    "Webcam": 0
}
MODEL_OPTIONS = {
    "YOLOv8n": "yolov8n.pt"
}

selected_video = list(VIDEO_OPTIONS.values())[0]
selected_model = list(MODEL_OPTIONS.values())[0]
line_orientation = "Horizontal"
line_position = 200

frame_lock = threading.Lock()
latest_frame = None

up_count = 0
down_count = 0
alert_50_sent = False
alert_75_sent = False
inference_started = False
running = False
video_thread = None

model_yolo = YOLO("yolov8n.pt")
tracked_objects = {}
next_id = 1
frame_count = 0
distance_threshold = 80

def init_people_counter():
    global inference_started, running
    inference_started = False
    running = False

def handle_people_post(request):
    global selected_video, selected_model, line_orientation, line_position
    global up_count, down_count, alert_50_sent, alert_75_sent
    global tracked_objects, next_id, running, video_thread

    selected_video = VIDEO_OPTIONS.get(request.form['video_source'], selected_video)
    selected_model = MODEL_OPTIONS.get(request.form['model_choice'], selected_model)
    new_orientation = request.form['line_orientation']
    new_position = int(request.form['line_position'])

    if new_orientation != line_orientation or new_position != line_position:
        line_orientation = new_orientation
        line_position = new_position
        tracked_objects.clear()
        next_id = 1
    else:
        line_orientation = new_orientation
        line_position = new_position

    up_count = 0
    down_count = 0
    alert_50_sent = False
    alert_75_sent = False

    if inference_started:
        running = False
        time.sleep(1)
        video_thread = threading.Thread(target=process_video, daemon=True)
        video_thread.start()

    return redirect(url_for('dashboard'))

def process_video():
    global latest_frame, up_count, down_count, tracked_objects, next_id
    global alert_50_sent, alert_75_sent, frame_count, running

    cap = cv2.VideoCapture(selected_video)
    running = True

    while running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1
        (h, w) = frame.shape[:2]
        detected_centroids = []

        results = model_yolo(frame, iou=0.7)
        for result in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = result.tolist()
            if int(cls) == 0 and conf > 0.5:
                centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                detected_centroids.append(centroid)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        existing_ids = list(tracked_objects.keys())
        prev_centroids = [tracked_objects[obj_id][0] for obj_id in existing_ids]

        cost_matrix = np.zeros((len(prev_centroids), len(detected_centroids)))
        for i, prev_c in enumerate(prev_centroids):
            for j, new_c in enumerate(detected_centroids):
                cost_matrix[i, j] = distance.euclidean(prev_c, new_c)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        new_tracked_objects = {}

        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < distance_threshold:
                obj_id = existing_ids[i]
                prev_centroid, last_seen, entry_frame = tracked_objects[obj_id]

                if line_orientation == "Horizontal":
                    if prev_centroid[1] < line_position and detected_centroids[j][1] > line_position:
                        down_count += 1
                    elif prev_centroid[1] > line_position and detected_centroids[j][1] < line_position:
                        up_count += 1
                else:
                    if prev_centroid[0] < line_position and detected_centroids[j][0] > line_position:
                        down_count += 1
                    elif prev_centroid[0] > line_position and detected_centroids[j][0] < line_position:
                        up_count += 1

                new_tracked_objects[obj_id] = (detected_centroids[j], frame_count, entry_frame)

        for j, new_c in enumerate(detected_centroids):
            if j not in col_ind:
                new_tracked_objects[next_id] = (new_c, frame_count, frame_count)
                next_id += 1

        tracked_objects = new_tracked_objects

        # Draw line
        font = cv2.FONT_HERSHEY_SIMPLEX
        if line_orientation == "Horizontal":
            cv2.line(frame, (0, line_position), (w, line_position), (0, 255, 255), 2)
            cv2.putText(frame, "Masuk", (10, line_position + 25), font, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, "Keluar", (10, line_position - 10), font, 0.6, (0, 0, 255), 2)
        else:
            cv2.line(frame, (line_position, 0), (line_position, h), (0, 255, 255), 2)
            cv2.putText(frame, "Masuk", (line_position + 10, 30), font, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, "Keluar", (line_position - 70, 30), font, 0.6, (0, 0, 255), 2)


        with frame_lock:
            latest_frame = cv2.imencode('.jpg', frame)[1].tobytes()

        time.sleep(0.03)

    cap.release()

def generate_people_feed():
    while True:
        with frame_lock:
            if latest_frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
        time.sleep(0.03)

def get_people_stats():
    return {
        "video_options": VIDEO_OPTIONS,
        "model_options": MODEL_OPTIONS,
        "current_video": [k for k, v in VIDEO_OPTIONS.items() if v == selected_video][0],
        "current_model": [k for k, v in MODEL_OPTIONS.items() if v == selected_model][0],
        "line_orientation": line_orientation,
        "line_position": line_position,
        "up_count": up_count,
        "down_count": down_count,
        "total_visitor": down_count - up_count
    }

def start_people_thread():
    global inference_started, video_thread
    if not inference_started:
        video_thread = threading.Thread(target=process_video, daemon=True)
        video_thread.start()
        inference_started = True

def stop_people_thread():
    global running
    running = False

