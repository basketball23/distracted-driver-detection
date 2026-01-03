import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import tensorflow as tf
layers = tf.keras.layers
import numpy as np

IMG_SIZE = 224


# facemesh setup
BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

# options for facemesh
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/face_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    # can use this for expression detection
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    min_tracking_confidence=0.7
)

# facemesh model
landmarker = FaceLandmarker.create_from_options(options)

# set video settings
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

timestamp = 0

# video loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # convert to rgb and wrap in mediapipe object
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect_for_video(mp_image, timestamp)
    timestamp += 1

    if result.face_landmarks:
        H, W, _ = frame.shape
        # for each face detected (1 in this case)
        for face in result.face_landmarks:
            '''
            Preprocessing and model inference
            '''

            # finding and plotting bounding box (face roi [region of interest])
            xs = [lm.x for lm in face]
            ys = [lm.y for lm in face]
            x1 = int(min(xs) * W)
            y1 = int(min(ys) * H)
            x2 = int(max(xs) * W)
            y2 = int(max(ys) * H)

            # apply padding
            pad = 0.25
            dx = int((x2 - x1) * pad)
            dy = int((y2 - y1) * pad)
            x1 = max(0, x1 - dx)
            y1 = max(0, y1 - dy)
            x2 = min(W, x2 + dx)
            y2 = min(H, y2 + dy)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # face roi
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue
            
            # preprocessing
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))

            face_input = face_resized.astype(np.float32) / 127.5 - 1.0
            face_input = np.expand_dims(face_input, axis=0)

            
            # run inference every 3 frames
            if timestamp % 3 != 0:
                cv2.imshow("Face Mesh", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue


    # break
    cv2.imshow("Face Mesh", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()