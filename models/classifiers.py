import torch
import torch.nn as nn
import torchvision.models as models

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
        out = self.classifier(combined)

        return out