import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import load_model  # Importă modelul salvat

# Încărcarea modelului
model = load_model("cifar10_model.pth")

# Configurare Streamlit
st.title("Clasificare CIFAR-10")
st.write("Încarcă o imagine pentru clasificare!")

uploaded_file = st.file_uploader("Alege o imagine...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagine Încărcată", use_column_width=True)

    # Transformare pentru model
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    image_tensor = transform(image).unsqueeze(0)  # Adăugăm batch dimension

    # Clasificare
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.argmax(output, 1).item()

    st.write(f"Predicție: **{prediction}**")
