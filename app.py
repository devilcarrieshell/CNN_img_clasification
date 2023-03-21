from model import ConvModel
import streamlit as st
import torch
import torch.nn as nn
import torchvision
from PIL import Image


# Define the class labels for CIFAR-10 dataset
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Load the trained model
model = ConvModel().to(device)
model = torch.load('cifar10_model.pth', map_location=torch.device('cpu'))
model.eval()

# Define the transformation to apply to input images
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Define the Streamlit app
def app():
    # Define the app title and description
    st.title('CIFAR-10 Image Classifier')
    st.write('This is a simple image classifier for the CIFAR-10 dataset.')

    # Create a file uploader widget
    st.set_option('deprecation.showfileUploaderEncoding', False)
    file = st.file_uploader('Choose an image file', type=['jpg', 'jpeg', 'png'])

    # If the user has uploaded an image, display it and make a prediction
    if file is not None:
        image = Image.open(file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Apply the transformation to the image and add a batch dimension
        image_tensor = transform(image).unsqueeze(0)

        # Make a prediction using the model
        with torch.no_grad():
            output = model(image_tensor)
            prediction = torch.argmax(output, 1).item()

        # Display the predicted class label
        st.write(f'Prediction: {classes[prediction]}')

## use command "streamlit run app.py" in your terminal 
#  
#  This will start the app and open it in your web browser. You can then use the file uploader to select an image file, 
#  and the app will display the uploaded image and predict its class label using the trained model.
# 