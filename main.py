
import cv2
import math
import time
import winsound
import threading
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model

options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)
landmarker = vision.FaceLandmarker.create_from_options(options)
model = load_model("drowsiness_model.h5")
CLASSES = {0: "Closed", 1: "Open", 2: "no_yawn", 3: "yawn"}
IMG_SIZE = 160

RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308]
POSE_IDX = [1, 152, 33, 263, 61, 291]

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
], dtype=np.float64)

MAR_THRESH = 0.5
CLOSED_FRAMES_LIMIT = 15
YAWN_FRAMES_LIMIT = 15
YAW_THRESH = 20
DISTRACTED_FRAMES_LIMIT = 20
CALIBRATION_SECONDS = 3
CNN_EVERY_N = 10
ESCALATE_AFTER = 5

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def eye_ratio(pts, idx):
    p1, p2, p3, p4, p5, p6 = [pts[i] for i in idx]
    return (dist(p2, p6) + dist(p3, p5)) / (2 * dist(p1, p4))

def mouth_ratio(pts, idx):
    top, bottom, left, right = [pts[i] for i in idx]
    return dist(top, bottom) / dist(left, right)

def get_yaw(pts, w, h):
    image_points = np.array([pts[i] for i in POSE_IDX], dtype=np.float64)
    cam_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)
    ok, rot_vec, _ = cv2.solvePnP(MODEL_POINTS, image_points, cam_matrix, np.zeros((4,1)))
    rot_mat, _ = cv2.Rodrigues(rot_vec)
    sy = math.sqrt(rot_mat[0,0]**2 + rot_mat[1,0]**2)
    return math.degrees(math.atan2(-rot_mat[2,0], sy))

def crop_region(frame, pts, idxs, pad=20):
    xs = [pts[i][0] for i in idxs]
    ys = [pts[i][1] for i in idxs]
    x1, x2 = max(min(xs) - pad, 0), max(xs) + pad
    y1, y2 = max(min(ys) - pad, 0), max(ys) + pad
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
    return np.expand_dims(crop, axis=0)

def classify(frame, pts, idxs):
    crop = crop_region(frame, pts, idxs)
    if crop is None:
        return None
    pred = model.predict(crop, verbose=0)[0]
    return CLASSES[np.argmax(pred)]

def beep(freq=1000, dur=300):
    winsound.Beep(freq, dur)

def draw_panel(frame, lines, status, color):
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (330, 40 + 32 * len(lines)), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (10, 10), (330, 40 + 32 * len(lines)), color, 2)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (22, 38 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(frame, (300, 25), 8, color, -1)
    (tw, th), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
    cv2.putText(frame, status, (frame.shape[1] - tw - 20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)

cap = cv2.VideoCapture(0)
frame_idx = 0
closed_counter = 0
yawn_counter = 0
distracted_counter = 0
was_drowsy = False
was_distracted = False
drowsy_start = None
last_escalation = 0
eye_label_cnn = None
mouth_label_cnn = None

calib_start = time.time()
calib_values = []
baseline_ear = None
EAR_THRESH = None

while True:
    ok, frame = cap.read()
    if not ok:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, frame_idx)

    status = "ALERT"
    color = (0, 200, 0)

    if result.face_landmarks:
        h, w, _ = frame.shape
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in result.face_landmarks[0]]

        ear = (eye_ratio(pts, RIGHT_EYE) + eye_ratio(pts, LEFT_EYE)) / 2
        mar = mouth_ratio(pts, MOUTH)
        yaw = get_yaw(pts, w, h)

        if baseline_ear is None:
            elapsed = time.time() - calib_start
            calib_values.append(ear)
            cv2.putText(frame, f"Calibrating... look at camera ({int(CALIBRATION_SECONDS - elapsed)+1}s)",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if elapsed >= CALIBRATION_SECONDS:
                baseline_ear = sum(calib_values) / len(calib_values)
                EAR_THRESH = baseline_ear * 0.80
            cv2.imshow("Drowsiness Detection", frame)
            frame_idx += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        if frame_idx % CNN_EVERY_N == 0:
            eye_label_cnn = classify(frame, pts, RIGHT_EYE)
            mouth_label_cnn = classify(frame, pts, MOUTH)

        eye_closed = (ear < EAR_THRESH) and (eye_label_cnn != "Open")
        is_yawning = (mar > MAR_THRESH) and (mouth_label_cnn != "no_yawn")

        closed_counter = closed_counter + 1 if eye_closed else 0
        yawn_counter = yawn_counter + 1 if is_yawning else 0
        distracted_counter = distracted_counter + 1 if abs(yaw) > YAW_THRESH else 0

        is_drowsy_now = closed_counter >= CLOSED_FRAMES_LIMIT
        is_distracted_now = distracted_counter >= DISTRACTED_FRAMES_LIMIT

        if is_drowsy_now:
            status = "DROWSY!"
            color = (0, 0, 255)
            if not was_drowsy:
                drowsy_start = time.time()
                threading.Thread(target=beep, args=(1000, 300), daemon=True).start()
            else:
                duration = time.time() - drowsy_start
                if duration >= ESCALATE_AFTER and time.time() - last_escalation > 1.5:
                    threading.Thread(target=beep, args=(1800, 500), daemon=True).start()
                    last_escalation = time.time()
        elif is_distracted_now:
            status = "DISTRACTED!"
            color = (0, 140, 255)
            if not was_distracted:
                threading.Thread(target=beep, args=(1000, 300), daemon=True).start()
        elif yawn_counter >= YAWN_FRAMES_LIMIT:
            status = "YAWNING"
            color = (0, 200, 255)

        was_drowsy = is_drowsy_now
        was_distracted = is_distracted_now

        lines = [f"Eye (EAR/CNN): {ear:.2f} / {eye_label_cnn}",
                 f"Mouth (MAR/CNN): {mar:.2f} / {mouth_label_cnn}",
                 f"Head Yaw: {yaw:.1f} deg"]
        draw_panel(frame, lines, status, color)

    cv2.imshow("Drowsiness Detection", frame)
    frame_idx += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()