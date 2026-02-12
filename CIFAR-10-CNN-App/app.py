import torch
import streamlit as st
from PIL import Image 
import torchvision.transforms as transforms

from model import SimpleCNN

# Page setup
st.set_page_config(page_title="CIFAR-10 Image Classifier", layout="centered")

st.title("🖼️ CIFAR-10 Image Classifier")
st.write("Upload an image and the model will predict its class.")

# Class names (from your notebook)
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Load model
@st.cache_resource
def load_model():
    model = SimpleCNN(dropout=0.2)
    model.load_state_dict(torch.load("cifar10_cnn_best.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# Image preprocessing (same normalization as training)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    )
])

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        st.subheader("Prediction Result")
        st.write(f"**Class:** {class_names[predicted.item()]}")
        st.write(f"**Confidence:** {confidence.item():.2f}")