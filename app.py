import cv2

import torch
import torch.nn as nn
import torchvision.models as models

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import numpy as np

from models.classifiers import DriverActionClassifier, EMASmoother

IMG_SIZE = 224
prediction_classes = [
    "safe driving",
    "texting - right",
    "talking on the phone - right",
    "texting - left",
    "talking on the phone - left",
    "operating the radio",
    "drinking",
    "reaching behind",
    "hair and makeup",
    "talking to passenger",
]

# face detector setup
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# options
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='models/blaze_face_short_range.tflite.task'),
    running_mode=VisionRunningMode.VIDEO,
    min_detection_confidence=0.5,
)

# facemesh model
detector = FaceDetector.create_from_options(options)

# mobilenet as the backbone
mobilenet = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
mobilenet.classifier = torch.nn.Identity()

model = DriverActionClassifier(backbone=mobilenet, num_classes=10)
model.eval()

smoother = EMASmoother(num_classes=10, alpha=0.5)

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

    result = detector.detect_for_video(mp_image, timestamp)
    timestamp += 1

    if result.detections:
        H, W, _ = frame.shape
        # for each face detected
        for face in result.detections:
            '''
            Preprocessing and model inference
            '''

            '''
            Head bounding box
            '''

            # finding and plotting bounding box (face roi [region of interest])
            x1 = face.bounding_box.origin_x
            y1 = face.bounding_box.origin_y
            x2 = x1 + face.bounding_box.width
            y2 = y1 + face.bounding_box.height

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
            face_input = torch.from_numpy(face_input).permute(2, 0, 1).unsqueeze(0)

            '''
            Bottom-right quadrant input (for hands)
            '''

            x3 = H//2
            y3 = W//2
            x4 = H
            y4 = W
            
            cv2.rectangle(frame, (y3, x3), (y4, x4), (0, 255, 0), 2)

            # hand roi
            hand_roi = frame[H//2:H, W//2:W]

            hand_rgb = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB)
            hand_resized = cv2.resize(hand_rgb, (IMG_SIZE, IMG_SIZE))

            hand_input = hand_resized.astype(np.float32) / 127.5 - 1.0
            hand_input = torch.from_numpy(hand_input).permute(2, 0, 1).unsqueeze(0)

            
            # frame preprocessing
            frame_roi = frame[0:H, 0:W]

            frame_rgb = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE))

            frame_input = frame_resized.astype(np.float32) / 127.5 - 1.0
            frame_input = torch.from_numpy(frame_input).permute(2, 0, 1).unsqueeze(0)
            
            # run inference every 3 frames
            if timestamp % 3 != 0:
                cv2.imshow("Face Mesh", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue


            # predictions
            with torch.no_grad():
                logits = model(frame_input, face_input, hand_input)
                smoothed_probs = smoother.smooth(logits)
                pred_class = smoothed_probs.argmax(dim=1)
                
                final_class = prediction_classes[pred_class.item()]
                print("Predicted class:", final_class)


    # break
    cv2.imshow("Face Mesh", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()