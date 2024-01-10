from torchvision.models import vgg16, VGG16_Weights
import torch
import torch.nn as nn

class ShroomClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(ShroomClassifier, self).__init__()
        self.features = vgg16(weights = VGG16_Weights.DEFAULT).features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
if __name__ == '__main__':
    model = ShroomClassifier()
    print(model)
    print(model(torch.randn(1, 3, 224, 224)).shape)