import torch
import torchvision.transforms as T
import torch.nn.functional as F 
from PIL import Image

# ================================================================= #

def process(image):
    # Class labels for prediction
    class_names = ['NRG', 'RG']

    # Load the model for testing
    MODEL = 'Research-Foundation-Retina/model_train2_1.pth'
    model = torch.load(MODEL, map_location=torch.device('cpu'))  # Loading the model for CPU

    # Preprocessing transformations
    preprocess = T.Compose([
        T.Resize(size=256),
        T.CenterCrop(size=224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    # Enable GPU mode if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)
    model.to(device)

    model.eval()

    with torch.no_grad():
        img = Image.fromarray(image)
        inputs = preprocess(img).unsqueeze(0).to(device)
        outputs = model(inputs)
        probabilities = F.softmax(outputs, dim=1)  # Get softmax probabilities
        confidence, preds = torch.max(probabilities, 1)
        label = class_names[preds]
        confidence_percentage = confidence.item() * 100  # Convert to percentage

    return label, confidence_percentage
