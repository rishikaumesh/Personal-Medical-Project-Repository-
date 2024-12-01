import torch
import torch.nn as nn
import torch.nn.functional as F

# first model architecture
class PneumoniaDetectionCNN(nn.Module):
    def __init__(self):
        super(PneumoniaDetectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(0.1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout(0.2)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.dropout3 = nn.Dropout(0.2)
        self.bn5 = nn.BatchNorm2d(256)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16384, 128)  # Adjust input size
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.dropout1(self.conv2(x))))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = F.relu(self.bn4(self.dropout2(self.conv4(x))))
        x = self.pool(x)
        x = F.relu(self.bn5(self.dropout3(self.conv5(x))))
        x = self.pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = torch.sigmoid(self.fc2(x))
        return x


#  second model architecture
class PneumoniaDetectionCNN2(nn.Module):
    def __init__(self):
        super(PneumoniaDetectionCNN2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(0.2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout(0.3)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.dropout3 = nn.Dropout(0.4)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout1(x)
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout2(x)
        x = self.pool(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.dropout3(x)
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


# third model architecture
class ImprovedPneumoniaDetectionCNN3(nn.Module):
    def __init__(self):
        super(ImprovedPneumoniaDetectionCNN3, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool3(x)
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


# Function to initialize a model based on user input
def initialize_model(model_name="ImprovedPneumoniaDetectionCNN3"):
    models = {
        "PneumoniaDetectionCNN": PneumoniaDetectionCNN,
        "PneumoniaDetectionCNN2": PneumoniaDetectionCNN2,
        "ImprovedPneumoniaDetectionCNN3": ImprovedPneumoniaDetectionCNN3,
    }
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' is not defined. Available models: {list(models.keys())}")
    model = models[model_name]()
    print(f"Initialized model: {model_name}")
    return model


if __name__ == "__main__":
    # Test model initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "ImprovedPneumoniaDetectionCNN3"  # Change this to test different models
    model = initialize_model(model_name).to(device)

    # Test the forward pass with a dummy input
    dummy_input = torch.randn(1, 1, 224, 224).to(device)
    output = model(dummy_input)
    print("Output shape:", output.shape)
