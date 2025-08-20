import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from flask import Flask, render_template, request
from PIL import Image

# -------------------------
# Model Definition (must match training)
# -------------------------
class AgeGenderNet(nn.Module):
    def __init__(self, age_classes=4, gender_classes=2):
        super().__init__()
        backbone = models.mobilenet_v2(weights=None)  # no pretrained
        self.features = backbone.features
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Sequential(
            self.dropout,
            nn.Linear(1280, 256)   # trained model had 1280 -> 256
        )
        self.age_head = nn.Linear(256, age_classes)       # 4 classes (trained this way)
        self.gender_head = nn.Linear(256, gender_classes) # 2 classes

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)  # (N, 1280)
        x = self.fc(x)           # (N, 256)
        age_logits = self.age_head(x)
        gender_logits = self.gender_head(x)
        return age_logits, gender_logits


# -------------------------
# Flask App Init
# -------------------------
app = Flask(__name__)

# -------------------------
# Load Trained Model
# -------------------------
AGE_CLASSES = 4
GENDER_CLASSES = 2

age_labels = ["0-18", "19-30", "31-50", "51+"]   # match training setup
gender_labels = ["Male", "Female"]

model = AgeGenderNet(age_classes=AGE_CLASSES, gender_classes=GENDER_CLASSES)

# NOTE: change file name if needed
STATE_PATH = "best_model.pth"   

state = torch.load(STATE_PATH, map_location="cpu")
model.load_state_dict(state, strict=True)
model.eval()

# -------------------------
# Image Transform
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400

    f = request.files["file"]
    if f.filename == "":
        return "No file selected", 400

    image = Image.open(f.stream).convert("RGB")
    x = transform(image).unsqueeze(0)  # (1,3,224,224)

    with torch.no_grad():
        age_logits, gender_logits = model(x)
        age_idx = age_logits.argmax(dim=1).item()
        gender_idx = gender_logits.argmax(dim=1).item()

    age_text = age_labels[age_idx]
    gender_text = gender_labels[gender_idx]

    return f"Predicted → Gender: {gender_text}, Age: {age_text}"


if __name__ == "__main__":
    app.run(debug=True)
