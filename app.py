from flask import Flask, render_template, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import load_model

app = Flask(__name__)

# Încărcăm modelul
model = load_model("cifar10_model.pth")

# Clasele datasetului CIFAR-10
classes = ('avion', 'mașină', 'pasăre', 'pisică', 'cerb', 'câine', 'broască', 'cal', 'navă', 'camion')

# Transformare imagine
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Nicio imagine încărcată'}), 400

    file = request.files['file']
    image = Image.open(file).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Adăugăm batch dimension

    # Clasificare
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.argmax(output, 1).item()

    return jsonify({'class': classes[prediction]})

if __name__ == '__main__':
    app.run(debug=True)
