import torch
from torchvision import transforms
from PIL import Image
from model import ResNet50
import io
import torch.nn.functional as F 

# Constants
CLASS_NAMES = ['BrownSpot', 'Healthy', 'Hispa', 'LeafBlast']
MODEL_PATH = "rice_disease_classification.pth"
device = torch.device("cpu")

# Load model
model = ResNet50(n_classes=len(CLASS_NAMES))

# Load checkpoint and extract state_dict
checkpoint = torch.load(MODEL_PATH, map_location=device)
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)  

model.eval()

# Define individual transforms
rotation_transform = transforms.RandomRotation(degrees=25)
shift_transform = transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
shear_transform = transforms.RandomAffine(degrees=0, shear=0.2)
zoom_transform = transforms.RandomAffine(degrees=0, scale=(0.8, 1.2))
flip_transform = transforms.RandomHorizontalFlip(p=0.5)

# Combine all augmentations
image_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    rotation_transform,
    shift_transform, 
    shear_transform,
    zoom_transform,
    flip_transform,
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]) 

# Predict function
def predict(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = image_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        return CLASS_NAMES[predicted.item()], confidence.item()
