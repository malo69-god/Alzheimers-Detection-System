from PIL import Image
import torchvision.transforms as transforms
import os



def load_image(img_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((128, 128)),  # Resize to fixed shape
        transforms.ToTensor(),  # Convert to tensor
    ])
    img = Image.open(img_path).convert("RGB")
    return transform(img)

image_path =
image_tensor = load_image(image_path)