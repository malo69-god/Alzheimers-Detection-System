import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader

print("🚀 Script started successfully", flush=True)

# 📌 Define Dataset Path
data_path = r"/Users/Hawa/Downloads/PythonProject"

# 🔹 Data Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # VGG-16 expects 224x224 images
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 🔹 Load Dataset
train_data = datasets.ImageFolder(root=os.path.join(r"/Users/Hawa/Downloads/PythonProject/data_mri", "train"), transform=transform)
test_data = datasets.ImageFolder(root=os.path.join(r"/Users/Hawa/Downloads/PythonProject/data_mri", "test"), transform=transform)

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

print(f"📂 Train Samples: {len(train_data)}, Test Samples: {len(test_data)}")

# 🔹 Load Pretrained VGG-16 Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.vgg16(pretrained=True)  # Load VGG-16 with pretrained weights

# 🔹 Modify Fully Connected Layers for 2-class classification
for param in model.features.parameters():  # Freeze convolutional layers
    param.requires_grad = False

model.classifier[6] = nn.Linear(4096, 2)  # Modify last layer for binary classification
model = model.to(device)

# 🔹 Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)  # Only train FC layers

# 🔹 Training Loop
for epoch in range(5):
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

# 🔹 Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = (correct / total) * 100
print(f"Test Accuracy: {accuracy:.2f}%")

print("🚀 Script finished successfully")
