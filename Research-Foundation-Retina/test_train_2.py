import numpy as np
import sys, random
import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

# Paths for image directory and model
IMDIR=sys.argv[1]
MODEL='./model_train2_1.pth'

# Load the model for testing
model = torch.load(MODEL)
model.eval()

# Class labels for prediction
class_names= ['NRG', 'RG']

# Retreive 9 random images from directory
files=Path(IMDIR).resolve().glob('*.*')
images=random.sample(list(files), 20)

# Configure plots
fig = plt.figure(figsize=(9,9))
rows,cols = 3,3
max_images_per_figure = rows * cols
# Preprocessing transformations
preprocess=transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

# Enable gpu mode, if cuda available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Perform prediction and plot results
# with torch.no_grad():
#     for num,img in enumerate(images):
#          img=Image.open(img).convert('RGB')
#          inputs=preprocess(img).unsqueeze(0).to(device)
#          outputs = model(inputs)
#          _, preds = torch.max(outputs, 1)    
#          label=class_names[preds]
#          img_np = np.array(img)
#          plt.subplot(rows,cols,num+1)
#          plt.title("Pred: "+label)
#          plt.axis('off')
#          plt.imshow(img_np)
#          output_path = f'./images/predicted_image_{num+1}_{label}.png'
#          plt.imsave(output_path, img_np)


with torch.no_grad():
    for i in range(0, len(images), max_images_per_figure):
        plt.figure(figsize=(10, 10))
        for num, img_path in enumerate(images[i:i + max_images_per_figure]):
            img = Image.open(img_path).convert('RGB')
            inputs = preprocess(img).unsqueeze(0).to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            label = class_names[preds]

            img_np = np.array(img)
            plt.subplot(rows, cols, num + 1)
            plt.title("Pred: " + label)
            plt.axis('off')
            plt.imshow(img_np)

            # Save the image with the prediction label in the filename
            output_path = f'./images/predicted_image_{i + num + 1}_{label}.png'
            plt.imsave(output_path, img_np)

        plt.tight_layout()
        plt.show()

'''
Sample run: python test.py test
'''