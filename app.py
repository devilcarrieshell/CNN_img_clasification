import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import load_model  # Importăm modelul

# Încărcăm modelul
model = load_model("cifar10_model.pth")

# Clasele datasetului CIFAR-10
classes = ('avion', 'mașină', 'pasăre', 'pisică', 'cerb', 'câine', 'broască', 'cal', 'navă', 'camion')

# Transformare imagine pentru model
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

st.title("Clasificator de imagini CIFAR-10")
st.write("Încarcă o imagine și modelul va încerca să o clasifice.")

uploaded_file = st.file_uploader("Alege o imagine...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagine Încărcată", use_column_width=True)

    # Pregătirea imaginii pentru model
    image_tensor = transform(image).unsqueeze(0)  # Adăugăm dimensiunea batch

    # Clasificare
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.argmax(output, 1).item()

    st.write(f"Predicție: **{classes[prediction]}**")
