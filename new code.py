import os
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader

print("🚀 Script started successfully")
print("🚀 Script started successfully", flush=True)

# 📌 Define Dataset Path (Modify this to your dataset location)
data_path = "/data_mri"  # Change this path

# 🔹 Data Transformations (Resizing, Normalization)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# 🔹 Load Dataset (Assumes data_path/AD and data_path/Normal folders exist)
train_data = datasets.ImageFolder(
    root=os.path.join(r"C:\Users\Administrator\PycharmProjects\PythonProject\data_mri", "train"),
    transform=transform
)
test_data = datasets.ImageFolder(
    root=os.path.join(r"C:\Users\Administrator\PycharmProjects\PythonProject\data_mri", "test"),
    transform=transform
)


train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

print(f"📂 Train Samples: {len(train_data)}, Test Samples: {len(test_data)}")

# 🔹 Define CNN Model
class MRI_CNN(nn.Module):
    def __init__(self):  # <-- Corrected
        super(MRI_CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 30 * 30, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 classes (AD vs Normal)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)  # Automatically flattens the feature map
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# 🔹 Training the Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MRI_CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):  # 🔥 Train for 5 epochs
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch + 1}/5], Loss: {total_loss / len(train_loader):.4f}")
    print("🚀 trained  successfully")

# 🔹 Evaluate Accuracy
print("🚀now evaluation begun")
# 🔹 Evaluate Accuracy
model.eval()
correct = 0
total = 0

with torch.no_grad():  # Disable gradient computation for evaluation
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # Get the class with the highest score
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

# Compute accuracy
accuracy = (correct / total) * 100
print(f"Test Accuracy: {accuracy:.2f}%")

print("🚀 Script started unsuccessfully")
