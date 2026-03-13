import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, render_template, url_for
from PIL import Image
from ultralytics import YOLO
# -------------------------------
# Flask Setup
# -------------------------------
app = Flask(__name__)
# Create the static directory if it doesn't exist
UPLOAD_FOLDER = 'static/uploaded_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# -------------------------------
# Model Paths and Device
# -------------------------------
CLASSIFICATION_MODEL_PATH = "models/classification_model.pt"
DETECTION_MODEL_PATH = "models/detection_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------------------------------
# Load Classes (same order as training)
# -------------------------------
classes = [
    'Alambadi Cow', 'Amritmahal Cow', 'Banni Buffalo', 'Bargur Cow', 'Dangi Cow',
    'Deoni Cow', 'Gir Cow', 'Hallikar Cow', 'Jaffrabadi Buffalo', 'Kangayam Cow',
    'Kankrej Cow', 'Kasaragod Cow', 'Kenkatha Cow', 'Kherigarh Cow',
    'Malnad gidda Cow', 'Mehsana Buffalo', 'Nagori Cow', 'Nagpuri Buffalo',
    'Nili ravi Buffalo', 'Nimari Cow', 'Pulikulam Cow', 'Rathi Cow',
    'Sahiwal Cow', 'Shurti Buffalo', 'Tharparkar Cow', 'Umblachery Cow'
]

# -------------------------------
# Load Models
# -------------------------------
# Load Classification Model
classification_model = models.resnet18(pretrained=False)
classification_model.fc = nn.Linear(classification_model.fc.in_features, len(classes))
classification_model.load_state_dict(torch.load(CLASSIFICATION_MODEL_PATH, map_location=DEVICE))
classification_model = classification_model.to(DEVICE)
classification_model.eval()

# Load YOLOv8 Object Detection Model
try:
    detection_model = YOLO(DETECTION_MODEL_PATH)
except Exception as e:
    print(f"Error loading YOLO model: {e}. Please make sure 'detection_model.pt' is in the models folder.")
    detection_model = None

# -------------------------------
# Image Transforms
# -------------------------------
classification_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_breed(image_path):
    # 1. Object Detection Stage
    if detection_model:
        results = detection_model(image_path)
        
        # Filter for 'cow' or 'buffalo' detections
        detected_animals = []
        for result in results:
            for box in result.boxes:
                # The class name is needed from your detection model training.
                # Assuming class 0 is 'cow' and class 1 is 'buffalo'
                class_id = int(box.cls)
                if class_id in [0, 1]:  # Replace with actual class IDs from your training
                    x1, y1, x2, y2 = [int(val) for val in box.xyxy[0]]
                    detected_animals.append((x1, y1, x2, y2))
        
        if not detected_animals:
            return "No cow or buffalo detected.", 0.0

        # For simplicity, we'll use the first detected animal
        box = detected_animals[0]
        x1, y1, x2, y2 = box

        # Crop the image using the bounding box
        original_image = Image.open(image_path).convert("RGB")
        cropped_image = original_image.crop((x1, y1, x2, y2))
        
        # 2. Breed Classification Stage
        processed_image = classification_transform(cropped_image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = classification_model(processed_image)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        return classes[pred.item()], conf.item() * 100

    else:
        # Fallback if detection model is not found, use original logic
        image = Image.open(image_path).convert("RGB")
        image = classification_transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = classification_model(image)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
        return classes[pred.item()], conf.item() * 100

# -------------------------------
# Routes
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", prediction="No file part")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", prediction="No file selected")

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        breed, confidence = predict_breed(filepath)
        
        # Clean up the uploaded image
        # os.remove(filepath)
        
        if confidence == 0.0:
            return render_template("index.html", prediction="No cattle detected.", image_path=url_for('static', filename=f'uploaded_images/{file.filename}'))
        
        return render_template("index.html", prediction=f"{breed} ({confidence:.2f}%)", image_path=url_for('static', filename=f'uploaded_images/{file.filename}'))

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)