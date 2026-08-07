
import av
import cv2
import math
import time
import numpy as np
import streamlit as st
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ---------------- Streamlit ---------------- #

st.set_page_config(
    page_title="Driver Drowsiness Detection",
    layout="centered"
)

st.title("🚗 Driver Drowsiness & Distraction Detector")
st.caption(
    "Allow camera access below. Look straight ahead for 3 seconds to calibrate."
)

# ---------------- MediaPipe ---------------- #

options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path="face_landmarker.task"
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

RIGHT_EYE = [33,160,158,133,153,144]
LEFT_EYE  = [362,385,387,263,373,380]
MOUTH      = [13,14,78,308]
POSE_IDX   = [1,152,33,263,61,291]

MODEL_POINTS = np.array([
    (0,0,0),
    (0,-330,-65),
    (-225,170,-135),
    (225,170,-135),
    (-150,-150,-125),
    (150,-150,-125)
],dtype=np.float64)

MAR_THRESH = 0.50
YAW_THRESH = 15

CLOSED_LIMIT = 15
YAWN_LIMIT = 15
DISTRACT_LIMIT = 8

CALIBRATION_TIME = 3


def dist(a,b):
    return math.hypot(a[0]-b[0],a[1]-b[1])


def eye_ratio(pts,idx):
    p1,p2,p3,p4,p5,p6=[pts[i] for i in idx]
    return (
        dist(p2,p6)+dist(p3,p5)
    )/(2*dist(p1,p4))


def mouth_ratio(pts,idx):
    t,b,l,r=[pts[i] for i in idx]
    return dist(t,b)/dist(l,r)


def get_yaw(pts,w,h):

    img=np.array([pts[i] for i in POSE_IDX],dtype=np.float64)

    cam=np.array([
        [w,0,w/2],
        [0,w,h/2],
        [0,0,1]
    ],dtype=np.float64)

    _,rv,_=cv2.solvePnP(
        MODEL_POINTS,
        img,
        cam,
        np.zeros((4,1))
    )

    rot,_=cv2.Rodrigues(rv)

    sy=math.sqrt(rot[0,0]**2+rot[1,0]**2)

    return math.degrees(
        math.atan2(-rot[2,0],sy)
    )


def draw_panel(frame,lines,status,color):

    overlay=frame.copy()

    cv2.rectangle(
        overlay,
        (10,10),
        (330,40+32*len(lines)),
        (30,30,30),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.55,
        frame,
        0.45,
        0,
        frame
    )

    cv2.rectangle(
        frame,
        (10,10),
        (330,40+32*len(lines)),
        color,
        2
    )

    for i,line in enumerate(lines):

        cv2.putText(
            frame,
            line,
            (22,38+32*i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            1,
            cv2.LINE_AA
        )

    (tw,_),_=cv2.getTextSize(
        status,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        3
    )

    cv2.putText(
        frame,
        status,
        (frame.shape[1]-tw-20,45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        color,
        3,
        cv2.LINE_AA
    )
class Processor(VideoProcessorBase):

    def __init__(self):

        self.landmarker = vision.FaceLandmarker.create_from_options(options)

        self.frame_id = 0

        self.closed_counter = 0
        self.yawn_counter = 0
        self.distracted_counter = 0
        self.face_missing_counter = 0

        self.calib_start = time.time()
        self.calib_values = []

        self.baseline_ear = None
        self.ear_thresh = None

        self.drowsy_start = None

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = int(time.time() * 1000)

        result = self.landmarker.detect_for_video(
            mp_image,
            timestamp
        )

        status = "ALERT"
        color = (0,200,0)

        flash = False

        # -------------------------------------------------
        # FACE FOUND
        # -------------------------------------------------

        if result.face_landmarks:

            self.face_missing_counter = 0

            h,w,_ = img.shape

            pts = [
                (int(lm.x*w), int(lm.y*h))
                for lm in result.face_landmarks[0]
            ]

            ear = (
                eye_ratio(pts,RIGHT_EYE)
                +
                eye_ratio(pts,LEFT_EYE)
            )/2

            mar = mouth_ratio(pts,MOUTH)

            yaw = get_yaw(pts,w,h)

            # ---------------- Calibration ---------------- #

            if self.baseline_ear is None:

                elapsed = time.time() - self.calib_start

                self.calib_values.append(ear)

                cv2.putText(
                    img,
                    f"Calibrating... {int(CALIBRATION_TIME-elapsed)+1}s",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,255),
                    2
                )

                if elapsed >= CALIBRATION_TIME:

                    self.baseline_ear = np.mean(self.calib_values)

                    self.ear_thresh = self.baseline_ear * 0.80

                return av.VideoFrame.from_ndarray(
                    img,
                    format="bgr24"
                )

            # ---------------- Drowsiness ---------------- #

            if ear < self.ear_thresh:
                self.closed_counter += 1
            else:
                self.closed_counter = 0

            # ---------------- Yawn ---------------- #

            if mar > MAR_THRESH:
                self.yawn_counter += 1
            else:
                self.yawn_counter = 0

            # ---------------- Head Turn ---------------- #

            if abs(yaw) > YAW_THRESH:
                self.distracted_counter += 1
            else:
                self.distracted_counter = max(
                    0,
                    self.distracted_counter-1
                )

            is_drowsy = (
                self.closed_counter >= CLOSED_LIMIT
            )

            is_yawn = (
                self.yawn_counter >= YAWN_LIMIT
            )

            is_distracted = (
                self.distracted_counter >= DISTRACT_LIMIT
            )

            # ---------------- Alert Logic ---------------- #

            if is_drowsy:

                if self.drowsy_start is None:
                    self.drowsy_start = time.time()

                if time.time()-self.drowsy_start >= 5:

                    status = "DROWSY!! WAKE UP"

                    flash = (
                        int(time.time()*4)%2==0
                    )

                else:

                    status = "DROWSY!"

                color = (0,0,255)

            else:

                self.drowsy_start = None

                if is_distracted:

                    status = "DISTRACTED!"
                    color = (0,140,255)

                elif is_yawn:

                    status = "YAWNING"
                    color = (0,200,255)

            if flash:

                red = np.full_like(
                    img,
                    (0,0,255)
                )

                img = cv2.addWeighted(
                    img,
                    0.7,
                    red,
                    0.3,
                    0
                )

            draw_panel(

                img,

                [
                    f"EAR : {ear:.2f}",
                    f"MAR : {mar:.2f}",
                    f"Yaw : {yaw:.1f}"
                ],

                status,

                color

            )

        # -------------------------------------------------
        # FACE LOST
        # -------------------------------------------------

        else:

            self.face_missing_counter += 1

            if self.face_missing_counter >= DISTRACT_LIMIT:

                status = "DISTRACTED!"
                color = (0,140,255)

                draw_panel(

                    img,

                    [
                        "Face : Lost",
                        "Please Look Ahead"
                    ],

                    status,

                    color

                )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


webrtc_streamer(
    key="drowsiness",
    video_processor_factory=Processor
)