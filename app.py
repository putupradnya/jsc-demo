from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
from people_counter import (
    init_people_counter,
    handle_people_post,
    generate_people_feed,
    get_people_stats,
    start_people_thread,
)
from yolo import (
    init_flood_detection,
    handle_flood_post,
    generate_flood_feed,
    get_flood_status,
    start_flood_thread
)

app = Flask(__name__)
init_people_counter()
init_flood_detection()

@app.route('/')
def home():
    return redirect(url_for('dashboard'))

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if request.method == 'POST':
        return handle_people_post(request)
    return render_template("dashboard.html", **get_people_stats())

@app.route('/video_feed')
def video_feed():
    return Response(generate_people_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start_people():
    start_people_thread()
    return redirect(url_for('dashboard', alert_message="People Counter started."))

@app.route('/reset', methods=['POST'])
def reset_people():
    return redirect(url_for('dashboard', alert_message="People Counter reset."))

@app.route('/stats')
def stats():
    return jsonify(get_people_stats())

# === Flood Detection ===
@app.route('/flood-detection', methods=['GET', 'POST'])
def flood_detection():
    if request.method == 'POST':
        return handle_flood_post(request)
    return render_template("flood_detection.html", **get_flood_status())

@app.route('/flood-feed')
def flood_feed():
    return Response(generate_flood_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/flood-start', methods=['POST'])
def start_flood():
    start_flood_thread()
    return redirect(url_for('flood_detection', alert_message="Flood Detection started."))

@app.route('/flood-stats')
def flood_stats():
    from yolo import flood_status, flood_level
    return jsonify({
        "status": flood_status,
        "level": flood_level
    })

@app.route('/stop', methods=['POST'])
def stop_people():
    from people_counter import stop_people_thread
    stop_people_thread()
    return redirect(url_for('dashboard', alert_message="People Counter stopped."))

@app.route('/flood-stop', methods=['POST'])
def stop_flood():
    from yolo import stop_flood_thread
    stop_flood_thread()
    return redirect(url_for('flood_detection', alert_message="Flood Detection stopped."))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
