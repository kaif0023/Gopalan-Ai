import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import gradio as gr

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 15

model = timm.create_model("resnet50", pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load("indian_bovine_breeds_resnet50.pkl", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ======================
# 2. Define Preprocessing
# ======================
val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ======================

# ======================
# 3. Class Names (15 breeds)
# ======================
CLASS_NAMES = [
    "Ayrshire",
    "Banni",
    "Bhadawari",
    "Brown_Swiss",
    "Gir",
    "Guernsey",
    "Holstein_Friesian",
    "Jaffrabadi",
    "Kankrej",
    "Mehsana",
    "Murrah",
    "Nagpuri",
    "Nili_Ravi",
    "Ongole",
    "Tharparkar"
]

# ======================
# 4. Breed Info (Realistic details)
# ======================
BREED_INFO = {
    "Ayrshire": {"cattle_category": "Cow", "average_height_cm": 135, "average_chest_width_cm": 28, "avg_milk_per_lactation": 7500},
    "Banni": {"cattle_category": "Cow", "average_height_cm": 130, "average_chest_width_cm": 27, "avg_milk_per_lactation": 5000},
    "Bhadawari": {"cattle_category": "Buffalo", "average_height_cm": 125, "average_chest_width_cm": 32, "avg_milk_per_lactation": 1800},
    "Brown_Swiss": {"cattle_category": "Cow", "average_height_cm": 140, "average_chest_width_cm": 29, "avg_milk_per_lactation": 9000},
    "Gir": {"cattle_category": "Cow", "average_height_cm": 133, "average_chest_width_cm": 28, "avg_milk_per_lactation": 3500},
    "Guernsey": {"cattle_category": "Cow", "average_height_cm": 135, "average_chest_width_cm": 28, "avg_milk_per_lactation": 6000},
    "Holstein_Friesian": {"cattle_category": "Cow", "average_height_cm": 145, "average_chest_width_cm": 30, "avg_milk_per_lactation": 10000},
    "Jaffrabadi": {"cattle_category": "Buffalo", "average_height_cm": 142, "average_chest_width_cm": 35, "avg_milk_per_lactation": 2500},
    "Kankrej": {"cattle_category": "Cow", "average_height_cm": 140, "average_chest_width_cm": 30, "avg_milk_per_lactation": 3100},
    "Mehsana": {"cattle_category": "Buffalo", "average_height_cm": 130, "average_chest_width_cm": 33, "avg_milk_per_lactation": 2200},
    "Murrah": {"cattle_category": "Buffalo", "average_height_cm": 132, "average_chest_width_cm": 34, "avg_milk_per_lactation": 3000},
    "Nagpuri": {"cattle_category": "Buffalo", "average_height_cm": 125, "average_chest_width_cm": 31, "avg_milk_per_lactation": 1700},
    "Nili_Ravi": {"cattle_category": "Buffalo", "average_height_cm": 135, "average_chest_width_cm": 33, "avg_milk_per_lactation": 3500},
    "Ongole": {"cattle_category": "Cow", "average_height_cm": 150, "average_chest_width_cm": 31, "avg_milk_per_lactation": 2000},
    "Tharparkar": {"cattle_category": "Cow", "average_height_cm": 138, "average_chest_width_cm": 29, "avg_milk_per_lactation": 3100}
}

# ======================
# 5. Prediction Function
# ======================
def predict(img):
    img = val_tfms(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    predicted_label = CLASS_NAMES[pred_idx]
    info = BREED_INFO.get(predicted_label, {})

    # Format details into a clean paragraph
    details_str = (
        f"*Cattle category: {info.get('cattle_category', '-') }* \n"
        f"\n*Predicted Breed: {predicted_label.replace('_',' ')}*\n\n"
        f"This breed belongs to the {info.get('cattle_category', '-') } category. "
        f"On average, it reaches a height of *{info.get('average_height_cm', '-') } cm*, "
        f"with a chest width of about *{info.get('average_chest_width_cm', '-') } cm*. "
        f"It typically produces around *{info.get('avg_milk_per_lactation', '-') } liters of milk per lactation*."
    )

    return details_str


# ======================
# 6. Gradio Interface
# ======================
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Markdown(),
    title="Animal Type Classification",
    description="Upload an image of a bovine breed to classify it and get detailed info."
)

if __name__ == "__main__":
    iface.launch()