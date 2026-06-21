import os
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split

print(" Script started successfully")
print(" Script started successfully", flush=True)

# Data Transformations (Added Data Augmentation for training to prevent overfitting)
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),  # Augmentation
    transforms.RandomRotation(10),      # Augmentation
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load Datasets
full_train_data = datasets.ImageFolder(
    root=os.path.join(r"C:\Users\Administrator\PycharmProjects\PythonProject\data_mri", "train"),
    transform=train_transform
)
test_data = datasets.ImageFolder(
    root=os.path.join(r"C:\Users\Administrator\PycharmProjects\PythonProject\data_mri", "test"),
    transform=val_test_transform
)

full_train_data.class_to_idx = {'AD': 0, 'Normal': 1}
test_data.class_to_idx = {'AD': 0, 'Normal': 1}

# Split 20% of training data for Validation
val_size = int(0.2 * len(full_train_data))
train_size = len(full_train_data) - val_size
train_data, val_data = random_split(full_train_data, [train_size, val_size])

# Apply non-augmented transform to validation set
val_data.dataset.transform = val_test_transform

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8, shuffle=False)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

print(f" Train Samples: {len(train_data)}, Val Samples: {len(val_data)}, Test Samples: {len(test_data)}")

# Define CNN Model with Dropout
class MRI_CNN(nn.Module):
    def __init__(self):  
        super(MRI_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Added Dropout layers to prevent co-adaptation of features
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(64 * 30 * 30, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)

# Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MRI_CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4) # Added Weight Decay (L2 Regularization)

# Early Stopping Parameters
patience = 3
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(15):  # Increased max epochs since early stopping will halt it
    # Training Phase
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation Phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"Epoch [{epoch + 1}/15] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    # Early Stopping Logic
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.pth')  # Save best weights
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(" Early stopping triggered. Training stopped.")
            break

# Load the best model weights for final evaluation
model.load_state_dict(torch.load('best_model.pth'))

# Evaluate Accuracy
print("now evaluation begun")
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
print(f"Test Accuracy on Best Model: {accuracy:.2f}%")
print(" Script finished successfully")
