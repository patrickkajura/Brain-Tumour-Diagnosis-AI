import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import tempfile
import zipfile

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define class names (must match the order used during training)
class_names = ['glioma', 'meningioma', 'nontumour', 'pituitary']

# Define model architecture (ResNet18 with custom head)
def load_model(model_path, num_classes=4):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    return model

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Function to predict image class
def predict_image(image, model):
    image = image.convert('RGB')  # Ensure 3 channels
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]
        return predicted_class

# Load model
@st.cache_resource
def get_model():
    model_weights_path = 'model_epoch_10.pth'
    return load_model(model_weights_path)

model = get_model()

# UI
st.title("🧠 Brain Tumor MRI Diagnosis Tool")
st.write("Upload a ZIP folder or individual MRI scan images (.jpg, .png)")

uploaded_file = st.file_uploader("Choose a ZIP file or an image...", type=["zip", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    temp_dir = tempfile.TemporaryDirectory()
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

    image_files = []

    if file_extension == ".zip":
        # Extract ZIP
        with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
            zip_ref.extractall(temp_dir.name)
        image_files = [os.path.join(temp_dir.name, f) for f in os.listdir(temp_dir.name)
                       if os.path.splitext(f)[1].lower() in [".jpg", ".jpeg", ".png"]]
    else:
        # Single image
        temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        image_files.append(temp_file_path)

    if len(image_files) == 0:
        st.warning("No valid image files found.")
    else:
        st.write(f"Found {len(image_files)} image(s). Predicting...")

        results = []
        for img_path in image_files:
            try:
                img = Image.open(img_path)
                prediction = predict_image(img, model)
                results.append({"filename": os.path.basename(img_path), "prediction": prediction})
            except Exception as e:
                results.append({"filename": os.path.basename(img_path), "prediction": f"Error: {str(e)}"})

        # Display Results
        st.subheader("🔍 Diagnosis Results:")
        for res in results:
            st.write(f"{res['filename']} ➡️ **{res['prediction']}**")

        # Optional: Save to CSV
        if st.checkbox("Download results as CSV"):
            import pandas as pd

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
