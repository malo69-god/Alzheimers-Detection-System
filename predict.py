import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

# 🔹 Define the CNN model
class MRI_CNN(nn.Module):
    def __init__(self):
        super(MRI_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 30 * 30, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 classes: AD, Normal

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# 🔹 Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MRI_CNN().to(device)
model.load_state_dict(torch.load("2mri_ad_model_retrained.pth", map_location=device))
model.eval()

# 🔹 Define image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 🔹 Load and preprocess new image
image_path = r"/Users/Hawa/Downloads/PythonProject/data_mri/test/AD/0a0d37fb-adeb-4e0e-8bc8-624cd70fc6e7.jpg"
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

# 🔹 Make prediction
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

# 🔹 Output result
if predicted.item() == 0:
    print("Prediction: Alzheimer's Detected (AD)")
else:
    print(" Prediction: Normal Brain")
