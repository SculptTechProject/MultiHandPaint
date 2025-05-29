import os
import cv2
import numpy as np
import mediapipe as mp
from absl import logging


# suppress logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'
os.environ['GLOG_logtostderr'] = '1'
logging.set_verbosity(logging.ERROR)

# init hands + face
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5,
                          max_num_hands=2)
mp_face   = mp.solutions.face_mesh
face      = mp_face.FaceMesh(min_detection_confidence=0.5,
                             min_tracking_confidence=0.5)
mp_draw   = mp.solutions.drawing_utils

# helpers (jak poprzednio)
FINGER_TIPS = [mp_hands.HandLandmark.INDEX_FINGER_TIP,
               mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
               mp_hands.HandLandmark.RING_FINGER_TIP,
               mp_hands.HandLandmark.PINKY_TIP]
FINGER_PIPS = [mp_hands.HandLandmark.INDEX_FINGER_PIP,
               mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
               mp_hands.HandLandmark.RING_FINGER_PIP,
               mp_hands.HandLandmark.PINKY_PIP]

def count_fingers(hand_landmarks):
    cnt = 0
    for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            cnt += 1
    return cnt

def pinch_dist(hand_landmarks, w, h):
    x1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * w)
    y1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * h)
    x2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w)
    y2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)
    return np.hypot(x2-x1, y2-y1)

# video + canvas
cap      = cv2.VideoCapture(0)
canvas   = None
prev_p   = None
color_idx= 0
colors   = [(0,0,255),(0,255,0),(255,0,0),(0,255,255)]
MOUTH_THRESH = 30  # piksele, dostosuj wg kamery

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_res = hands.process(rgb)
    face_res = face.process(rgb)

    mode = 'brush'
    # lewa dłoń = kontroler
    if hand_res.multi_hand_landmarks:
        for lm, hd in zip(hand_res.multi_hand_landmarks,
                          hand_res.multi_handedness):
            if hd.classification[0].label == 'Left':
                fingers = count_fingers(lm)
                pd      = pinch_dist(lm, w, h)
                if fingers == 0:      mode = 'eraser'
                elif fingers == 5:    mode = 'brush'
                elif fingers == 2 and pd < 40: mode = 'color'
                cv2.putText(frame, f'Control: {mode}', (10,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # prawa dłoń = malowanie/gumowanie/zmiana koloru
    if hand_res.multi_hand_landmarks:
        for lm, hd in zip(hand_res.multi_hand_landmarks,
                          hand_res.multi_handedness):
            if hd.classification[0].label == 'Right':
                x = int(lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w)
                y = int(lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)
                if mode == 'brush':
                    if prev_p:
                        cv2.line(canvas, prev_p, (x,y), colors[color_idx], 5)
                    prev_p = (x,y)
                elif mode == 'eraser':
                    cv2.circle(canvas, (x,y), 30, (0,0,0), -1)
                    prev_p = None
                elif mode == 'color':
                    color_idx = (color_idx + 1) % len(colors)
                    mode = 'brush'
                    prev_p = None
    else:
        prev_p = None

    if face_res.multi_face_landmarks:
        flm = face_res.multi_face_landmarks[0]
        # punkty 13 = górna warga, 14 = dolna warga
        x13 = int(flm.landmark[13].x * w); y13 = int(flm.landmark[13].y * h)
        x14 = int(flm.landmark[14].x * w); y14 = int(flm.landmark[14].y * h)
        mouth_dist = np.hypot(x14-x13, y14-y13)
        if mouth_dist > MOUTH_THRESH:
            canvas[:] = 0
            cv2.putText(frame, "Buziak wyczyscil!", (10,100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    # overlay
    frame = cv2.addWeighted(frame,1,canvas,0.7,0)
    cv2.putText(frame, "'c' clear, 'q' quit", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow("Air Paint Multi-Hand + Mouth", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        canvas[:] = 0
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
