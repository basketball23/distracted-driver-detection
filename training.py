import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class DriverActionClassifier(nn.Module):
    def __init__(self, backbone, num_classes=10):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(3 * 576, num_classes)

    def forward(self, image, face, hand):
        im = self.backbone(image).flatten(1)
        f = self.backbone(face).flatten(1)
        ha = self.backbone(hand).flatten(1)
        combined = torch.cat([im, f, ha], dim=1)
        return self.classifier(combined)
  

class DriverDataset(Dataset):
    def __init__(self, root_dir, detector, img_size=224):
        self.img_size = img_size
        self.detector = detector
        self.samples = []

        classes = sorted([c for c in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, c))])
        self.class_map = {c: idx for idx, c in enumerate(classes)}

        for c in classes:
            class_dir = os.path.join(root_dir, c)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(class_dir, fname), self.class_map[c]))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frame = cv2.imread(path)
        if frame is None:
            raise ValueError(f"Failed to read {path}")
        H, W, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)
        if result and result.detections and len(result.detections) > 0:
            face = result.detections[0].bounding_box
            x1 = max(0, int(face.origin_x - 0.25 * face.width))
            y1 = max(0, int(face.origin_y - 0.25 * face.height))
            x2 = min(W, int(face.origin_x + face.width + 0.25 * face.width))
            y2 = min(H, int(face.origin_y + face.height + 0.25 * face.height))
            face_roi = frame[y1:y2, x1:x2]
        else:
            face_roi = frame.copy()
        hand_roi = frame[H//2:H, W//2:W]
        full_roi = frame.copy()
        def preprocess(img):
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            
            img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
            
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=img.dtype).view(3,1,1)
            std  = torch.tensor([0.229, 0.224, 0.225], dtype=img.dtype).view(3,1,1)
            img = (img - mean) / std
            return img

        return (
            preprocess(full_roi),
            preprocess(face_roi),
            preprocess(hand_roi),
            label
        )   
        

BaseOptions = mp.tasks.BaseOptions
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode
detector = mp.tasks.vision.FaceDetector.create_from_options(
    FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path='/Users/rushilmohan/Downloads/blaze_face_short_range.tflite'),
        running_mode=VisionRunningMode.IMAGE,
        min_detection_confidence=0.5
    )
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_ROOT = "/Users/rushilmohan/Google Drive/My Drive/Distracted Driver Detection/state-farm-distracted-driver-detection/imgs/train"

dataset = DriverDataset(
    root_dir=DATASET_ROOT,
    detector=detector
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

full, face, hand, labels = next(iter(train_loader))
print(full.shape, face.shape, hand.shape, labels.shape)

mobilenet = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
mobilenet.classifier = nn.Identity()
model = DriverActionClassifier(backbone=mobilenet, num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
best_val_loss = float('inf')

for epoch in range(20):
    model.train()
    total_loss = 0
    for full, face, hand, labels in train_loader:
        full, face, hand, labels = full.to(device), face.to(device), hand.to(device), labels.to(device)
        logits = model(full, face, hand)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for full, face, hand, labels in val_loader:
            full, face, hand, labels = full.to(device), face.to(device), hand.to(device), labels.to(device)
            logits = model(full, face, hand)
            loss = criterion(logits, labels)
            val_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct / total
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_driver_action_model.pth")
    print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} Val Loss={avg_val_loss:.4f} Val Acc={val_acc:.4f}")
