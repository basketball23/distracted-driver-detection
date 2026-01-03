import torch
import torch.nn as nn
import torch.nn.functional as F

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
    

class EMASmoother:
    def __init__(self, num_classes, alpha=0.3, device='cpu'):
        """
        num_classes: number of classes in your classifier
        alpha: smoothing factor (0 < alpha <= 1). Higher = more responsive, lower = smoother
        """
        self.alpha = alpha
        self.num_classes = num_classes
        self.device = device
        self.ema_probs = None  # stores running smoothed probabilities

    @torch.no_grad()
    def smooth(self, logits):
        """
        logits: tensor of shape (batch_size, num_classes)
        """
        probs = F.softmax(logits, dim=1)

        if self.ema_probs is None:
            self.ema_probs = probs.clone()
        else:
            self.ema_probs = self.alpha * probs + (1 - self.alpha) * self.ema_probs

        return self.ema_probs