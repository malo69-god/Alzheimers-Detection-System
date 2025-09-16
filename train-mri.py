import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import nibabel as nib
from sklearn.model_selection import train_test_split

# Load MRI Data
def load_mri_data(data_path, use_nifti=False):
    x_data, y_data = [], []
    for label in ["AD", "Normal"]:
        folder = os.path.join(data_path, label)
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            img = nib.load(file_path).get_fdata() if use_nifti else np.load(file_path)
            x_data.append(np.expand_dims(img, axis=0))
            y_data.append(0 if label == "Normal" else 1)
    x_data, y_data = np.array(x_data).astype('float32') / np.max(x_data), np.array(y_data)
    return train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Define Dataset
class MRIDataset(data.Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.images)
    def __getitem__(self, idx): return self.images[idx], self.labels[idx]

# Define 3D CNN Model
class MRI3DCNN(nn.Module):
    def __init__(self):
        super(MRI3DCNN, self).__init__()
        self.conv1, self.conv2 = nn.Conv3d(1, 32, 3, 1, 1), nn.Conv3d(32, 64, 3, 1, 1)
        self.pool, self.dropout = nn.MaxPool3d(2, 2), nn.Dropout(0.5)
        self.fc1, self.fc2 = nn.Linear(64*16*16*16, 128), nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(self.dropout(x))

# Training
data_path, device = "/path/to/mri_dataset/", torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_train, x_test, y_train, y_test = load_mri_data(data_path, use_nifti=False)
train_loader = data.DataLoader(MRIDataset(x_train, y_train), batch_size=8, shuffle=True)
test_loader = data.DataLoader(MRIDataset(x_test, y_test), batch_size=8, shuffle=False)

model, criterion, optimizer = MRI3DCNN().to(device), nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=0.001)
for epoch in range(10):
    model.train()
    total_loss = sum(criterion(model(images.to(device)), labels.to(device)).backward() or optimizer.step() or loss.item() for images, labels in train_loader)
    print(f"Epoch [{epoch+1}/10], Loss: {total_loss/len(train_loader):.4f}")

# Evaluation
model.eval()
correct = sum((torch.max(model(images.to(device)), 1)[1] == labels.to(device)).sum().item() for images, labels in test_loader)
print(f"Test Accuracy: {correct / len(test_loader.dataset) * 100:.2f}%")