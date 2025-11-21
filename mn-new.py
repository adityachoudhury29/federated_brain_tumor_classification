import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import datasets, transforms
import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix, classification_report

# ========================
# 1. MobileNetV2 from scratch
# ========================
class InvertedResidual(nn.Module):
    def __init__(self, inp, outp, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == outp
        layers = []
        if expand_ratio != 1:
            # Pointwise
            layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        # Depthwise & project
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, outp, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outp)
        ])
        self.conv = nn.Sequential(*layers)
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=4, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        interverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        features = [nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(input_channel),
                    nn.ReLU6(inplace=True)]
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, t))
                input_channel = output_channel
        features += [
            nn.Conv2d(input_channel, self.last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.last_channel),
            nn.ReLU6(inplace=True),
        ]
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.last_channel, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

# ========================
# 2. Dataset and Loader
# ========================
# Directories (edit if needed)
data_dir = '/home/aditya/Desktop/Everything/federated_learning/dataset'
train_dir = os.path.join(data_dir, 'Training')
test_dir = os.path.join(data_dir, 'Testing')

# Transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Read and combine sets, split
train_source_dataset_raw = datasets.ImageFolder(train_dir, transform=None)
test_source_dataset_raw = datasets.ImageFolder(test_dir, transform=None)
full_dataset_raw = ConcatDataset([train_source_dataset_raw, test_source_dataset_raw])
class_names = train_source_dataset_raw.classes
num_classes = len(class_names)

# Always set manual seed for reproducibility!
torch.manual_seed(42)
train_size = int(0.8 * len(full_dataset_raw))
val_size = len(full_dataset_raw) - train_size
train_subset, val_subset = random_split(full_dataset_raw, [train_size, val_size])

class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    def __len__(self):
        return len(self.subset)

train_dataset = TransformedSubset(train_subset, transform=data_transforms['train'])
val_dataset = TransformedSubset(val_subset, transform=data_transforms['test'])

BATCH_SIZE = 32
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
    'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
print(f"Classes: {class_names}")
print(f"Train: {dataset_sizes['train']} images, Val/Test: {dataset_sizes['val']} images")

# ========================
# 3. Training & Validation
# ========================
LEARNING_RATE = 0.01
MOMENTUM = 0.9
NUM_EPOCHS = 50
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = MobileNetV2(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

def train_model(model, criterion, optimizer, num_epochs=22):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f'New best validation accuracy: {best_acc:.4f}')
        print()
    model.load_state_dict(best_model_wts)
    print(f'Training complete. Best Val Acc: {best_acc:.4f}')
    return model

print("Training MobileNetV2 BTGC model...")
trained_model = train_model(model, criterion, optimizer, num_epochs=NUM_EPOCHS)

model_save_path = '/home/aditya/Desktop/Everything/federated_learning/BTGC_MobileNetV2.pth'
torch.save(trained_model.state_dict(), model_save_path)
print(f"Model saved at {model_save_path}")

# ========================
# 4. Test, Accuracy, Confusion Matrix
# ========================
# The val_dataset is our test set (from previous split for fair evaluation)
test_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Reload best model for evaluation
model = MobileNetV2(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.eval()

all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)
accuracy = np.mean(all_preds == all_labels)
print(f'Test Accuracy: {accuracy*100:.2f}%')

cm = confusion_matrix(all_labels, all_preds)
print("Confusion matrix:")
print(cm)
print("Classification report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# ========================
# 5. Confusion Matrix Plot
# ========================
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(cm, class_names)


# ========================
# 6. Explainable AI with Grad-CAM
# ========================
# Run 'pip install grad-cam' in your terminal first!
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2 # OpenCV is used by the grad-cam library for visualization

print("\nGenerating Grad-CAM explanations...")

# 1. Reload the model (already done in section 4, but let's be explicit)
model = MobileNetV2(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.eval()

# 2. Specify the target layer for Grad-CAM
# This is typically the last convolutional layer in the network.
# In your MobileNetV2, a good choice is the final block of the 'features' sequential module.
target_layers = [model.features[-1]]

# 3. Get a few images from the validation set to explain
num_images_to_explain = 5
unnormalized_imgs = []
input_tensors = []

# We need the original, un-normalized images for visualization
# So let's create a temporary dataset without normalization
vis_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
vis_dataset = TransformedSubset(val_subset, transform=vis_transform)

for i in range(num_images_to_explain):
    # Get the un-normalized image for visualization
    unnormalized_img, label = vis_dataset[i]
    unnormalized_imgs.append(unnormalized_img)
    
    # Get the normalized image to feed into the model
    normalized_img, _ = val_dataset[i]
    input_tensors.append(normalized_img.unsqueeze(0))

# 4. Initialize Grad-CAM
cam = GradCAM(model=model, target_layers=target_layers)

# 5. Generate and display the heatmaps
for i in range(num_images_to_explain):
    input_tensor = input_tensors[i].to(device)
    
    # Get model prediction
    output = model(input_tensor)
    _, prediction_idx = torch.max(output, 1)
    predicted_class = class_names[prediction_idx]
    
    # You can also explain the 'target' class if you want
    # For now, we explain the model's actual prediction
    targets = [ClassifierOutputTarget(prediction_idx.item())]
    
    # Generate the CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :] # Get the first (and only) CAM in the batch
    
    # Overlay CAM on the original image
    # The original image needs to be in numpy format (H, W, C) and values between 0-1
    rgb_img = unnormalized_imgs[i].permute(1, 2, 0).numpy()
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    # Get the true label for the title
    true_label_idx = vis_dataset[i][1]
    true_class = class_names[true_label_idx]

    # Plot the results
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rgb_img)
    axes[0].set_title(f'Original Image\nTrue Class: {true_class}')
    axes[0].axis('off')
    
    axes[1].imshow(visualization)
    axes[1].set_title(f'Grad-CAM Overlay\nPredicted: {predicted_class}')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()