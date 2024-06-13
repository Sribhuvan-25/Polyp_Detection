from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(Image.fromarray(image))
        return image, label


def create_densenet_model(num_classes):
    base_model = models.densenet121(pretrained=True)
    
    # Replace the classifier
    num_ftrs = base_model.classifier.in_features
    base_model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)  # No softmax here; CrossEntropyLoss will handle it
    )
    
    return base_model


from torch.optim.lr_scheduler import StepLR
from torch.optim import SGD

train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Create dataset
# dataset = CustomDataset(images, labels, transform=transform)
labels_numpy = train_labels.numpy() if isinstance(train_labels, torch.Tensor) else train_labels

# Set up Stratified K-Fold
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize metrics
accuracies, precisions, recalls = [], [], []

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 2

model = create_densenet_model(num_classes).to(device)

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels_numpy)), labels_numpy)):
    print(f"Fold {fold + 1}/{n_splits}")
    model = create_densenet_model(num_classes).to(device)
    # Create data subsets for training and validation
    # Assuming train_images and train_labels are defined and loaded previously
    train_subset = CustomDataset([train_images[i] for i in train_idx], [train_labels[i] for i in train_idx], transform=train_transform)
    val_subset = CustomDataset([train_images[i] for i in val_idx], [train_labels[i] for i in val_idx], transform=val_transform)

    # Create data loaders
    BATCH_SIZE = 10
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)

    # Assuming create_densenet_model is defined elsewhere
    model = create_densenet_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Training loop for a specified number of epochs
    for epoch in range(50):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Step the scheduler
        scheduler.step()

        # Validation phase
        model.eval()
        val_predictions, val_targets = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        # Evaluation and metrics calculation after last epoch
        if epoch == 49:
            accuracy = accuracy_score(val_targets, val_predictions)
            precision = precision_score(val_targets, val_predictions, average='weighted')
            recall = recall_score(val_targets, val_predictions, average='weighted')

            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)

            print(f'Fold {fold+1} Results - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}')

# Calculate average scores across all folds
average_accuracy = np.mean(accuracies)
average_precision = np.mean(precisions)
average_recall = np.mean(recalls)

print(f'\nAverage scores for all folds:')
print(f'Accuracy: {average_accuracy}')
print(f'Precision: {average_precision}')
print(f'Recall: {average_recall}')


torch.save(model, 'denseNet_model_new_image_1')