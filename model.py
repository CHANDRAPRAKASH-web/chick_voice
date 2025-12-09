import torch
import torch.nn as nn
from torchvision.models import resnet18


class ChickenResNet(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()

        # base ResNet-18, no pretrained weights
        self.backbone = resnet18(weights=None)

        # change first conv to accept 1-channel input (mel spectrograms)
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # change the final classifier
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def get_model(num_classes: int = 3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChickenResNet(num_classes=num_classes).to(device)
    return model, device


if __name__ == "__main__":
    # quick sanity check
    model, device = get_model(num_classes=3)
    print("Using device:", device)
    dummy = torch.randn(4, 1, 64, 94).to(device)  # batch of 4 fake spectrograms
    out = model(dummy)
    print("Output shape:", out.shape)  # [4, 3]
