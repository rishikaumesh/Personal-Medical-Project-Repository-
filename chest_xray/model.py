import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedPneumoniaDetectionCNN3(nn.Module):
    """
    CNN Model for Pneumonia Detection with Five Convolutional Blocks
    and Global Average Pooling.
    """
    def __init__(self):
        super(ImprovedPneumoniaDetectionCNN3, self).__init__()
        
        # First block
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # Input channels = 1 for grayscale images
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 112x112 output

        # Second block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 56x56 output

        # Third block
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 output

        # Fourth block
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 output

        # Fifth block
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)  # 7x7 output

        # Global average pooling layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # 1x1 output

        # Fully connected layers
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        # Second block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        # Third block
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)

        # Fourth block
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.pool4(x)

        # Fifth block
        x = F.relu(self.bn9(self.conv9(x)))
        x = self.pool5(x)

        # Global average pooling and flatten
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Sigmoid for binary classification
        return x

def initialize_model():
    """
    Function to initialize the CNN model.
    """
    model = ImprovedPneumoniaDetectionCNN3()
    print(model)
    return model

if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_model().to(device)

    # Test the forward pass with a dummy input
    dummy_input = torch.randn(1, 1, 224, 224).to(device)  # Batch size of 1, 1 channel, 224x224 image
    output = model(dummy_input)
    print("Output shape:", output.shape)
