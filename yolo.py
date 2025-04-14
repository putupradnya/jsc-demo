import cv2
import threading
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import LineString, Polygon
import math
import numpy as np
import os
import requests
import time
from dotenv import load_dotenv
from flask import redirect, url_for
from utils.alert import send_whatsapp_alert

load_dotenv()

# === Konfigurasi Garis dan Posisi (default)
first_x, first_y = 1094, 231
second_x, second_y = 1083, 403
pixelsInAMeter = 15
tipHeight = 15
warningLevel = 10

# === Opsi video & model
FLOOD_VIDEO_OPTIONS = {
    "Sample Video": "sample/sample1.mp4",
    "Webcam": 0
}
FLOOD_MODEL_OPTIONS = {
    "Segmentasi Banjir": "sample/best.pt"
}

selected_flood_video = list(FLOOD_VIDEO_OPTIONS.values())[0]
selected_flood_model = list(FLOOD_MODEL_OPTIONS.values())[0]

# State
latest_flood_frame = None
flood_status = "Not Running"
flood_level = 0.0
running = False
flood_thread = None
last_alert_time = None


def load_font(size=45):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except:
        return ImageFont.load_default()


def send_telegram_alert(distance, warningLevel, image):
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

    message = f"⚠️ ALERT! Ketinggian air mencapai {distance:.2f} meter, melebihi batas {warningLevel} meter!"
    url_msg = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    url_img = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"

    requests.get(url_msg, params={"chat_id": CHAT_ID, "text": message})
    img_bytes = cv2.imencode(".jpg", np.array(image))[1].tobytes()
    requests.post(url_img, data={"chat_id": CHAT_ID}, files={"photo": img_bytes})


def draw_percentage_markers(draw, start, end, color, width=3, interval=0.2):
    x1, y1 = start
    x2, y2 = end
    for i in range(1, 5):
        fraction = i * interval
        px = int(x1 + fraction * (x2 - x1))
        py = int(y1 + fraction * (y2 - y1))
        draw.line([(px - 5, py), (px + 5, py)], fill=color, width=width)
        draw.text((px + 10, py - 10), f"{int((1 - fraction) * 100)}%", fill=color, font=load_font(20))


def calculateDistance(x1, y1, x2, y2):
    return float("{:.2f}".format(tipHeight - (math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)) / pixelsInAMeter))


def find_intersection(polygon, line):
    intersection = polygon.intersection(line)
    return [(intersection.xy)] if not intersection.is_empty else None


def run_flood_detection_stream():
    global latest_flood_frame, flood_status, flood_level, running, last_alert_time

    model = YOLO(selected_flood_model)
    cap = cv2.VideoCapture(selected_flood_video)

    running = True
    last_alert_time = None

    while running and cap.isOpened():
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        results = model(frame, conf=0.4)
        segments = getattr(getattr(results[0], 'masks'), 'xy', [])

        if not segments or len(segments[0]) == 0:
            flood_status = "NO WATER DETECTED"
            time.sleep(0.03)
            continue

        polygon_vertices = [(int(x), int(y)) for x, y in segments[0]]

        annotated_frame = results[0].plot()
        pil_image = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        line_coords = [(first_x, first_y), (second_x, second_y)]
        draw.line(line_coords, fill=(0, 0, 255), width=3)

        polygon = Polygon(polygon_vertices)
        line = LineString(line_coords)

        if find_intersection(polygon, line):
            intersection_points = find_intersection(polygon, line)
            intersection_x = intersection_points[0][0][0]
            intersection_y = intersection_points[0][1][0]
            dist = calculateDistance(intersection_x, intersection_y, first_x, first_y)
            flood_level = dist
            flood_status = "WARNING" if dist >= warningLevel else "SAFE"

            draw.line([(first_x, first_y), (intersection_x, intersection_y)], fill=(0, 255, 0), width=3)
            draw.text((1013, 134), str(dist), font=load_font(40), fill=(0, 0, 0))

            if dist >= warningLevel:
                draw.text((936, 98), "WARNING!!!", font=load_font(30), fill=(255, 0, 0))
                now = time.time()
                if last_alert_time is None or (now - last_alert_time) >= 10:
                    send_telegram_alert(dist, warningLevel, pil_image)

                    # message to whatsapp
                    message_body = f"""⚠️ *ALERT!*
                    Ketinggian air mencapai *{dist:.2f} meter*
                    Status: *WARNING 🚨*"""
                    send_whatsapp_alert(message_body)
                    last_alert_time = now
        else:
            flood_status = "SAFE"
            draw.text((936, 98), "SAFE", font=load_font(30), fill=(0, 255, 0))

        draw_percentage_markers(draw, (first_x, first_y), (second_x, second_y), color=(255, 255, 255), width=3)
        latest_flood_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        time.sleep(0.03)

    cap.release()


def generate_flood_feed():
    global latest_flood_frame
    while True:
        if latest_flood_frame is not None:
            ret, buffer = cv2.imencode('.jpg', latest_flood_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)


def get_flood_status():
    return {
        "flood_level": flood_level,
        "flood_status": flood_status,
        "video_options": FLOOD_VIDEO_OPTIONS,
        "model_options": FLOOD_MODEL_OPTIONS,
        "current_video": [k for k, v in FLOOD_VIDEO_OPTIONS.items() if v == selected_flood_video][0],
        "current_model": [k for k, v in FLOOD_MODEL_OPTIONS.items() if v == selected_flood_model][0]
    }


def init_flood_detection():
    global flood_status, flood_level, running
    flood_status = "Not Running"
    flood_level = 0.0
    running = False


def start_flood_thread():
    global flood_thread
    flood_thread = threading.Thread(target=run_flood_detection_stream, daemon=True)
    flood_thread.start()


def handle_flood_post(request):
    global selected_flood_video, selected_flood_model, running, flood_thread

    selected_flood_video = FLOOD_VIDEO_OPTIONS.get(request.form['video_source'], selected_flood_video)
    selected_flood_model = FLOOD_MODEL_OPTIONS.get(request.form['model_choice'], selected_flood_model)

    if running:
        running = False
        time.sleep(1)
        flood_thread = threading.Thread(target=run_flood_detection_stream, daemon=True)
        flood_thread.start()

    return redirect(url_for("flood_detection"))

def stop_flood_thread():
    global running
    running = False

