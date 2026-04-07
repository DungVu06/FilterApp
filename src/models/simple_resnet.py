import torch.nn as nn
from torchvision import models

class FaceLandmarkModel(nn.Module):
    def __init__(self, num_landmarks=98):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_landmarks * 2) 

    def forward(self, x):
        return self.backbone(x)