import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Define transformations for training and testing
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# Load pre-trained models
def load_models(image_model_path, patch_model_path):
    model_patch = torch.load(patch_model_path)
    model_patch.eval()
    model_full = torch.load(image_model_path)
    model_full.eval()
    return model_patch, model_full

# Helper functions
def get_heatmap(input_tensor, model, target_layers):
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    pred = model(input_tensor)
    _, predicted_class = pred.max(1)
    targets = [ClassifierOutputTarget(predicted_class.item())]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    return grayscale_cam, predicted_class.item(), torch.softmax(pred, dim=1)[0, predicted_class].item()

def adaptive_thresholding(heatmap, predicted_label):
    threshold_percentile = 75 if predicted_label == 1 else 95
    threshold_value = np.percentile(heatmap, threshold_percentile)
    return (heatmap >= threshold_value).astype('uint8')

def detect_features_in_channel(channel, mask=None):
    features = cv2.goodFeaturesToTrack(channel, mask=mask, maxCorners=100, qualityLevel=0.01, minDistance=150)
    return np.int0(features).reshape(-1, 2) if features is not None else np.array([])

# Extract features and labels from images
def extract_features(image_path, label, model_patch, model_full, transform):
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    input_tensor = transform(image).unsqueeze(0)
    
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
    
    heatmap_full, pred_full, conf_full = get_heatmap(input_tensor, model_full, [model_full.features.norm5])
    heatmap_resized_full = cv2.resize(heatmap_full, (image_np.shape[1], image_np.shape[0]))
    heatmap_normalized_full = heatmap_resized_full / np.max(heatmap_resized_full)
    
    binary_mask_full = adaptive_thresholding(heatmap_normalized_full, pred_full)
    
    salient_points_filtered = detect_features_in_channel(cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY), binary_mask_full)
    
    patch_confidences = []
    patch_probs = []
    for pt in salient_points_filtered:
        top_left_x = max(pt[0] - 137, 0)
        top_left_y = max(pt[1] - 150, 0)
        bottom_right_x = min(top_left_x + 275, image_np.shape[1])
        bottom_right_y = min(top_left_y + 300, image_np.shape[0])
        
        patch = image_np[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        patch_tensor = torch.from_numpy(np.transpose(cv2.resize(patch, (224, 224)), (2, 0, 1)).astype('float32') / 255.0).unsqueeze(0)
        if torch.cuda.is_available():
            patch_tensor = patch_tensor.cuda()
        with torch.no_grad():
            output = model_patch(patch_tensor)
            prob = torch.softmax(output, dim=1)
            pred = torch.argmax(prob, dim=1)
            conf = prob[0][pred].item()
        
        patch_confidences.append(conf)
        patch_probs.append(prob.cpu().numpy().flatten())

    if patch_confidences:
        avg_conf = np.mean(patch_confidences)
        std_conf = np.std(patch_confidences)
        min_conf = np.min(patch_confidences)
        max_conf = np.max(patch_confidences)
        median_conf = np.median(patch_confidences)
        patch_probs_mean = np.mean(patch_probs, axis=0)
    else:
        avg_conf = std_conf = min_conf = max_conf = median_conf = 0.0
        patch_probs_mean = np.array([0.5, 0.5])

    return [avg_conf, std_conf, min_conf, max_conf, median_conf, pred_full, conf_full] + list(patch_probs_mean), label

def train_meta_classifier(train_image_features, train_labels):
    # Normalize features
    scaler = StandardScaler()
    train_image_features = scaler.fit_transform(train_image_features)

    # Apply PCA
    pca = PCA(n_components=0.95)  # Retain 95% of variance
    train_image_features_pca = pca.fit_transform(train_image_features)

    # Define base models for voting
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)),
        ('lr', LogisticRegression())
    ]

    # Define voting classifier
    voting_clf = VotingClassifier(estimators=base_models, voting='soft')
    voting_clf.fit(train_image_features_pca, train_labels)

    return scaler, pca, voting_clf

def get_combined_prediction(features, scaler, pca, voting_clf):
    features = scaler.transform([features])  # Normalize features
    features_pca = pca.transform(features)  # Apply PCA
    combined_pred = voting_clf.predict(features_pca)
    return combined_pred[0]

def extract_and_visualize(image_path, label, model_patch, model_full, transform, output_dir, scaler, pca, voting_clf):
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    input_tensor = transform(image).unsqueeze(0)
    
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()

    heatmap_full, pred_full, conf_full = get_heatmap(input_tensor, model_full, [model_full.features.norm5])
    heatmap_resized_full = cv2.resize(heatmap_full, (image_np.shape[1], image_np.shape[0]))
    heatmap_normalized_full = heatmap_resized_full / np.max(heatmap_resized_full)

    binary_mask_full = adaptive_thresholding(heatmap_normalized_full, pred_full)
    
    overlay_img_full = cv2.addWeighted(image_np, 0.6, cv2.applyColorMap(np.uint8(255 * heatmap_normalized_full), cv2.COLORMAP_JET), 0.4, 0)
    
    salient_points_filtered = detect_features_in_channel(cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY), binary_mask_full)
    
    image_with_all_salient_points = image_np.copy()
    for pt in salient_points_filtered:
        cv2.circle(image_with_all_salient_points, (pt[0], pt[1]), 5, (0, 0, 255), 5)
    
    image_with_classified_patches = image_np.copy()
    patch_confidences = []
    patch_probs = []
    for pt in salient_points_filtered:
        top_left_x = max(pt[0] - 137, 0)
        top_left_y = max(pt[1] - 150, 0)
        bottom_right_x = min(top_left_x + 275, image_with_classified_patches.shape[1])
        bottom_right_y = min(top_left_y + 300, image_with_classified_patches.shape[0])
        
        patch = image_with_classified_patches[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        patch_tensor = torch.from_numpy(np.transpose(cv2.resize(patch, (224, 224)), (2, 0, 1)).astype('float32') / 255.0).unsqueeze(0)
        if torch.cuda.is_available():
            patch_tensor = patch_tensor.cuda()
        with torch.no_grad():
            output = model_patch(patch_tensor)
            prob = torch.softmax(output, dim=1)
            pred = torch.argmax(prob, dim=1)
            conf = prob[0][pred].item()
        
        patch_confidences.append(conf)
        patch_probs.append(prob.cpu().numpy().flatten())
        
        color = (0, 255, 0) if pred.item() == 1 else (0, 0, 255)
        cv2.rectangle(image_with_classified_patches, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, 2)
        text = f"{conf:.2f}"
        cv2.putText(image_with_classified_patches, text, (top_left_x, top_left_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    avg_conf = np.mean(patch_confidences) if patch_confidences else 0.0
    std_conf = np.std(patch_confidences) if patch_confidences else 0.0
    min_conf = np.min(patch_confidences) if patch_confidences else 0.0
    max_conf = np.max(patch_confidences) if patch_confidences else 0.0
    median_conf = np.median(patch_confidences) if patch_confidences else 0.0
    patch_probs_mean = np.mean(patch_probs, axis=0) if patch_probs else np.array([0.5, 0.5])
    
    features = [avg_conf, std_conf, min_conf, max_conf, median_conf, pred_full, conf_full] + list(patch_probs_mean)
    final_prediction = get_combined_prediction(features, scaler, pca, voting_clf)
    
    fig, axs = plt.subplots(3, 2, figsize=(16, 24))
    axs[0, 0].imshow(heatmap_normalized_full, cmap='jet')
    axs[0, 0].axis('off')
    axs[0, 0].set_title(f'Heatmap Full (DenseNet) - Pred: {pred_full}, GT: {label}')
    
    axs[0, 1].imshow(overlay_img_full)
    axs[0, 1].axis('off')
    axs[0, 1].set_title('Overlay Image Full')
    
    axs[1, 0].imshow(image_with_all_salient_points)
    axs[1, 0].axis('off')
    axs[1, 0].set_title('All Salient Points')
    
    axs[1, 1].imshow(image_with_all_salient_points)
    axs[1, 1].axis('off')
    axs[1, 1].set_title('Filtered Salient Points')
    
    axs[2, 0].imshow(image_with_classified_patches)
    axs[2, 0].axis('off')
    axs[2, 0].set_title('Patch Classifications')
    
    axs[2, 1].axis('off')
    
    output_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_composite.png"
    plt.savefig(os.path.join(output_dir, output_filename))
    plt.close()
    
    return final_prediction, label