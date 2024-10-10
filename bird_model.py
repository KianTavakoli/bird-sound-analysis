import torch
import torch.nn as nn
from torchvision.models import resnet18

# Model definition using ResNet18, modified for single-channel audio spectrograms
class BirdSoundClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BirdSoundClassifier, self).__init__()
        # Use a pre-trained ResNet18 model
        self.model = resnet18(weights="IMAGENET1K_V1")

        # Modify the input layer to accept a single channel (for spectrograms)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Modify the output layer to match the number of bird classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
