"""
AST20707 Creative Electronics Project

Overview:
- Uses Camera Module 3 and YOLOv8 pose estimation to monitor an elderly person in a room.
- PIR motion sensor acts as the first trigger so heavy AI processing runs only when motion exists.
- Calculates the torso angle from shoulder and hip keypoints to estimate standing vs lying posture.
- If a lying posture (low torso angle) persists longer than a time threshold, the system:
  - Turns on a local LED and buzzer alarm.
  - Captures a snapshot image.
  - Sends a Telegram alert message (and optional photo) to caregivers.
- Provides a Flask web dashboard with live video stream and status for remote monitoring.
"""

from flask import Flask, Response, render_template, jsonify
import cv2
import math
import os
import time
import requests
from gpiozero import MotionSensor, LED, Buzzer
from picamera2 import Picamera2
from ultralytics import YOLO

# Create the Flask web app
app = Flask(__name__)

# GPIO pin assignments
PIR_PIN = 23
LED_PIN = 27
BUZZER_PIN = 22

# Telegram bot settings
BOT_TOKEN = ""
CHAT_ID = ""

# Detection and camera settings
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MIN_CONF = 0.4
FALL_ANGLE_THRESHOLD = 30
FALL_CONFIRM_SECS = 8
ALERT_COOLDOWN_SECS = 30
FRAME_SKIP = 2
SAVE_DIR = "captures"
os.makedirs(SAVE_DIR, exist_ok=True)

# Set up hardware devices
pir = MotionSensor(PIR_PIN)
led = LED(LED_PIN)
buzzer = Buzzer(BUZZER_PIN)

# Set up the Pi camera.
# RGB888 gives frames in RGB format, which are then converted for OpenCV
picam2 = Picamera2()
picam2.preview_configuration.main.size = (FRAME_WIDTH, FRAME_HEIGHT)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load the YOLO pose model.
model = YOLO("yolov8n-pose.pt")

# Global state used while the app is running
fall_start_time = None
alert_sent = False
last_alert_time = 0
frame_counter = 0
last_annotated = None
system_status = "Starting"
last_angle = None


def send_telegram_message(text):
    # Skip sending if Telegram credentials were not filled in.
    if not BOT_TOKEN or not CHAT_ID:
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=10)
    except Exception:
        # Ignore Telegram errors so the app can keep running.
        pass


def send_telegram_photo(photo_path, caption="Possible fall detected"):
    # Skip sending if Telegram credentials were not filled in
    if not BOT_TOKEN or not CHAT_ID:
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    try:
        with open(photo_path, "rb") as photo:
            requests.post(
                url,
                data={"chat_id": CHAT_ID, "caption": caption},
                files={"photo": photo},
                timeout=20,
            )
    except Exception:
        # Ignore Telegram errors so the app can keep running.
        pass


def get_torso_angle_and_frame(frame):
    # Run YOLO pose detection on the frame.
    results = model(frame, verbose=False, conf=MIN_CONF)

    # Process each detection result
    for r in results:
        # If no keypoints were detected, skip
        if r.keypoints is None or len(r.keypoints.xy) == 0:
            continue

        # Use the first detected person's keypoints
        kps = r.keypoints.xy[0]

        # Need at least shoulder and hip keypoints
        if len(kps) < 13:
            continue

        # COCO keypoint indexes:
        # 5 = left shoulder, 6 = right shoulder, 11 = left hip, 12 = right hip
        ls, rs, lh, rh = kps[5], kps[6], kps[11], kps[12]

        # Skip if keypoints are missing or invalid.
        if (ls[0] == 0 and ls[1] == 0) or (rs[0] == 0 and rs[1] == 0):
            continue
        if (lh[0] == 0 and lh[1] == 0) or (rh[0] == 0 and rh[1] == 0):
            continue

        # Find the midpoint between both shoulders and both hips.
        shoulder_mid_x = float((ls[0] + rs[0]) / 2)
        shoulder_mid_y = float((ls[1] + rs[1]) / 2)
        hip_mid_x = float((lh[0] + rh[0]) / 2)
        hip_mid_y = float((lh[1] + rh[1]) / 2)

        # Create a torso vector from hip midpoint to shoulder midpoint.
        dx = shoulder_mid_x - hip_mid_x
        dy = shoulder_mid_y - hip_mid_y

        # Convert the vector into an angle in degrees
        angle = abs(math.degrees(math.atan2(dy, dx)))

        # Normalize the result to 0-90 degrees.
        # Smaller angles mean the body is more horizontal.
        if angle > 90:
            angle = 180 - angle

        # Draw YOLO keypoints and skeleton on the frame.
        annotated = r.plot()
        return angle, annotated

    # No valid person found.
    return None, frame


def put_overlay(frame, text, y, color):
    # Draw status text on the video frame.
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def process_frame():
    global fall_start_time, alert_sent, last_alert_time, frame_counter
    global last_annotated, system_status, last_angle

    # Capture one frame from the Pi camera.
    raw = picam2.capture_array()

    # Convert RGB to BGR because OpenCV uses BGR by default.
    frame = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)

    # If no motion is detected by the PIR sensor,
    # reset fall-related state and do not run YOLO.
    if not pir.motion_detected:
        fall_start_time = None
        alert_sent = False
        last_angle = None
        led.off()
        buzzer.off()
        system_status = "No motion detected"
        put_overlay(frame, system_status, 30, (180, 180, 180))
        last_annotated = frame
        return frame

    system_status = "Motion detected"
    frame_counter += 1

    # Skip some frames to reduce CPU load.
    # When a frame is skipped, return the last processed frame.
    if last_annotated is not None and frame_counter % FRAME_SKIP != 0:
        return last_annotated

    # Detect pose and calculate torso angle.
    angle, annotated = get_torso_angle_and_frame(frame)
    last_angle = angle

    if angle is None:
        # Motion exists, but no person was detected clearly.
        fall_start_time = None
        alert_sent = False
        led.off()
        buzzer.off()
        system_status = "No person detected"
        put_overlay(annotated, system_status, 30, (0, 165, 255))
    else:
        # Show the current torso angle.
        put_overlay(annotated, f"Torso angle: {angle:.1f} deg", 30, (255, 255, 0))

        # If the torso angle is below the threshold, assume a lying posture
        if angle < FALL_ANGLE_THRESHOLD:
            if fall_start_time is None:
                # Start counting how long the person stays in that posture.
                fall_start_time = time.time()

            elapsed = time.time() - fall_start_time
            system_status = f"Lying detected {elapsed:.1f}s"
            put_overlay(annotated, system_status, 60, (0, 0, 255))

            # Confirm the fall only after the lying posture lasts long enough.
            if elapsed >= FALL_CONFIRM_SECS:
                led.on()
                buzzer.on()
                now = time.time()

                # Send alert only once, or again after cooldown expires.
                if (not alert_sent) or (now - last_alert_time > ALERT_COOLDOWN_SECS):
                    filename = os.path.join(SAVE_DIR, f"fall_{int(now)}.jpg")
                    cv2.imwrite(filename, annotated)
                    send_telegram_message(
                        "Possible fall detected. Please check immediately."
                    )
                    send_telegram_photo(filename, "SafeStep Watch alert")
                    alert_sent = True
                    last_alert_time = now
        else:
            # Person is upright again, so reset fall tracking and alarms
            fall_start_time = None
            alert_sent = False
            led.off()
            buzzer.off()
            system_status = "Standing or upright"
            put_overlay(annotated, system_status, 60, (0, 255, 0))

    # Save the latest processed frame for streaming and skipped-frame reuse.
    last_annotated = annotated
    return annotated


def mjpeg_generator():
    # Continuously generate JPEG frames for the browser video stream.
    while True:
        frame = process_frame()
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue

        # Send each frame in MJPEG format.
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )
        time.sleep(0.03)


@app.route("/")
def index():
    # Serve the main dashboard page.
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    # Stream the processed camera frames to the browser.
    return Response(
        mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/status")
def status():
    # Return live system information as JSON for the frontend.
    return jsonify(
        {
            "system_status": system_status,
            "pir_detected": bool(pir.motion_detected),
            "last_angle": last_angle,
            "alarm_on": bool(led.is_lit or buzzer.is_active),
        }
    )


if __name__ == "__main__":
    try:
        # Start the Flask server so other devices on the same network can access it.
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        # Make sure alarm outputs are turned off if the app stops.
        led.off()
        buzzer.off()
