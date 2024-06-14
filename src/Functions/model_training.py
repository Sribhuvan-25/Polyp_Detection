import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from PIL import Image
from tqdm import tqdm
from torch.optim import SGD

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

def train_denseNetModel(train_images, train_labels, num_classes, path, num_epochs=50, batch_size=10, n_splits=5):
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

    # Convert labels to numpy array if they are in tensor format
    labels_numpy = train_labels.numpy() if isinstance(train_labels, torch.Tensor) else train_labels

    # Set up Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Initialize metrics
    accuracies, precisions, recalls = [], [], []

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels_numpy)), labels_numpy)):
        print(f"Fold {fold + 1}/{n_splits}")

        # Create data subsets for training and validation
        train_subset = CustomDataset([train_images[i] for i in train_idx], [train_labels[i] for i in train_idx], transform=train_transform)
        val_subset = CustomDataset([train_images[i] for i in val_idx], [train_labels[i] for i in val_idx], transform=val_transform)

        # Create data loaders
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size)

        # Assuming create_densenet_model is defined elsewhere
        model = create_densenet_model(num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

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
            if epoch == num_epochs - 1:
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

    torch.save(model, 'denseNet')




def evaluate_model(model, test_images, test_labels, batch_size, device):
    model.eval()

    # Define transformations for testing
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Create testing dataset and data loader
    test_dataset = CustomDataset(test_images, test_labels, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluation
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Calculate and print accuracy, precision, and recall
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    return accuracy, precision, recall
