BASE_LR = 0.01
EPOCH_DECAY = 30 # number of epochs after which the Learning rate is decayed exponentially.
DECAY_WEIGHT = 0.1 # factor by which the learning rate is reduced.


# DATASET INFO
NUM_CLASSES = 2 # set the number of classes in your dataset
DATA_DIR = './datasets/eyepac-light-v2-512-jpg' # to run with the sample dataset, just set to 'hymenoptera_data'

# DATALOADER PROPERTIES
BATCH_SIZE = 20 # Set as high as possible. If you keep it too high, you'll get an out of memory error.


### GPU SETTINGS
CUDA_DEVICE = 0 # Enter device ID of your gpu if you want to run on gpu. Otherwise neglect.
GPU_MODE = 1 # set to 1 if want to run on gpu.


# SETTINGS FOR DISPLAYING ON TENSORBOARD
USE_TENSORBOARD = 0 #if you want to use tensorboard set this to 1.
TENSORBOARD_SERVER = "http://localhost" # If you set.
EXP_NAME = "fine_tuning_experiment" # if using tensorboard, enter name of experiment you want it to be displayed as.

### Section 1 - First, let's import everything we will be needing.

# from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchmetrics import Precision, Recall, F1Score
from torch.utils.tensorboard import SummaryWriter

## If you want to use the GPU, set GPU_MODE TO 1 in config file
use_gpu = GPU_MODE
if use_gpu:
    torch.cuda.set_device(CUDA_DEVICE)

### SECTION 2 - data loading and shuffling/augmentation/normalization : all handled by torch automatically.

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = DATA_DIR
dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
         for x in ['train', 'val']}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=25)
                for x in ['train', 'val']}
dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
dset_classes = dsets['train'].classes


### SECTION 3 : Writing the functions that do training and validation phase.
device = torch.device(f'cuda:{CUDA_DEVICE}' if use_gpu else 'cpu')
# Initialize metrics
precision = Precision(task="binary", num_classes=NUM_CLASSES, average='macro').to(device)
recall = Recall(task="binary", num_classes=NUM_CLASSES, average='macro').to(device)
f1 = F1Score(task="binary", num_classes=NUM_CLASSES, average='macro').to(device)

# Initialize TensorBoard writer
writer = SummaryWriter()

# Train the model
def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=100):
    since = time.time()

    best_model = model
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Reset metric states
        precision.reset()
        recall.reset()
        f1.reset()

        for phase in ['train', 'val']:
            if phase == 'train':
                mode = 'train'
                optimizer = lr_scheduler(optimizer, epoch)
                model.train()  # Set model to training mode
            else:
                model.eval()
                mode = 'val'

            running_loss = 0.0
            running_corrects = 0

            counter = 0
            for data in dset_loaders[phase]:
                inputs, labels = data

                # Move to GPU if available
                if use_gpu:
                    inputs, labels = Variable(inputs.float().cuda()), Variable(labels.long().cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)

                # Compute loss
                loss = criterion(outputs, labels)

                # Backward pass and optimization in the training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # Update metrics
                precision.update(preds, labels)
                recall.update(preds, labels)
                f1.update(preds, labels)

                # Statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)

                counter += 1

            # Compute final epoch metrics
            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects.item() / float(dset_sizes[phase])
            epoch_precision = precision.compute().item()
            epoch_recall = recall.compute().item()
            epoch_f1 = f1.compute().item()

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Precision: {epoch_precision:.4f} Recall: {epoch_recall:.4f} F1: {epoch_f1:.4f}')

            # Log metrics to TensorBoard
            writer.add_scalar(f'{phase}/Loss', epoch_loss, epoch)
            writer.add_scalar(f'{phase}/Accuracy', epoch_acc, epoch)
            writer.add_scalar(f'{phase}/Precision', epoch_precision, epoch)
            writer.add_scalar(f'{phase}/Recall', epoch_recall, epoch)
            writer.add_scalar(f'{phase}/F1_Score', epoch_f1, epoch)

            # Deep copy the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    writer.close()  # Close the TensorBoard writer
    return best_model

# This function changes the learning rate over the training model.
def exp_lr_scheduler(optimizer, epoch, init_lr=BASE_LR, lr_decay_epoch=EPOCH_DECAY):
    lr = init_lr * (DECAY_WEIGHT ** (epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print(f'LR is set to {lr}')

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

### SECTION 4 : DEFINING MODEL ARCHITECTURE.
checkpoint_pretrain= './checkpoints/resnet50_512_ImageNet+Fundus.pth'
# We use Resnet18 here.
model_ft = models.resnet50(pretrained=checkpoint_pretrain)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)

criterion = nn.CrossEntropyLoss()

if use_gpu:
    criterion.cuda()
    model_ft.cuda()

optimizer_ft = optim.RMSprop(model_ft.parameters(), lr=0.0001)

# Train the model and save the best model
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=100)

# Save model
torch.save(model_ft.state_dict(), 'fine_tuned_best_model.pt')
