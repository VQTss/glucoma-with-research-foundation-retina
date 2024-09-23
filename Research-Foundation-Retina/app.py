import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

# Class labels for prediction
class_names = ['NRG', 'RG']

# Load the model for testing
MODEL = './model_train2_1.pth'
model = torch.load(MODEL, map_location=torch.device('cpu'))  # Loading the model for CPU
model.eval()

# Preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Enable GPU mode if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Streamlit app title and description
st.title("Fundus Image Classification")
st.write("Upload an image to classify it as NRG or RG")

# File uploader for a single image
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

# If an image is uploaded, process and display it
if uploaded_file:
    # Load and preprocess the image
    img = Image.open(uploaded_file).convert('RGB')
    inputs = preprocess(img).unsqueeze(0).to(device)

    # Make a prediction
    with torch.no_grad():
        outputs = model(inputs)
        probabilities = F.softmax(outputs, dim=1)  # Get softmax probabilities
        confidence, preds = torch.max(probabilities, 1)
        label = class_names[preds]
        confidence_percentage = confidence.item() * 100  # Convert to percentage
    
    # Clear previous image
    st.empty()

    # Display the uploaded image
    st.image(img, width=500)

    # Display the prediction result
    if label == 'RG':
        st.success(f'The Eye is Glaucomatous ({confidence_percentage:.4f}% confidence)')
    else:
        st.error(f'The Eye is not Glaucomatous ({confidence_percentage:.4f}% confidence)')
