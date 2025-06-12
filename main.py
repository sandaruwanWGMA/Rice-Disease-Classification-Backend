# main.py
import io
import torch
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

# --- CHANGE 1: The import is now direct because model.py is in the same folder ---
from model import ResNet50

# --- 1. DEFINE CONSTANTS AND LOAD THE MODEL ---

device = torch.device("cpu")

# --- CHANGE 2: The model path is now in the root directory ---
MODEL_PATH = "rice_disease_classification.pth"

CLASS_NAMES = ['BrownSpot', 'Healthy', 'Hispa', 'LeafBlast']

# Instantiate the model architecture
model = ResNet50(n_classes=len(CLASS_NAMES))

# Load the saved model weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

# Set the model to evaluation mode
model.eval()
model.to(device)

# Define the image transformation pipeline
image_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 2. CREATE THE FASTAPI APP ---
app = FastAPI(title="Rice Disease Classification API")

@app.get("/")
def read_root():
    """A simple endpoint to check if the API is running."""
    return {"message": "Welcome to the Rice Disease Classification API!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Receives an image file and returns the predicted class and confidence."""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_tensor = image_transforms(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class_index = torch.max(probabilities, 1)

        predicted_class_name = CLASS_NAMES[predicted_class_index.item()]
        confidence_score = confidence.item()

        return JSONResponse(content={
            "prediction": predicted_class_name,
            "confidence": f"{confidence_score:.4f}"
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"An error occurred: {str(e)}"})