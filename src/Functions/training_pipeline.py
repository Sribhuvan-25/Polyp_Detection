from guided_patch_extraction import *


def train_pipeline(train_root_dir):
    transform = get_transform()
    model_patch, model_full = load_models()

    # Extract features from training images
    train_image_features = []
    train_labels = []
    # train_root_dir = 'Training'
    for subfolder in ['normal_training_images', 'abnormal_training_images']:
        folder_path = os.path.join(train_root_dir, subfolder)
        label = 0 if subfolder == 'normal_training_images' else 1
        for image_file in os.listdir(folder_path):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, image_file)
                features, lbl = extract_features(image_path, label, model_patch, model_full, transform)
                train_image_features.append(features)
                train_labels.append(lbl)

    # Train meta-classifier
    train_image_features = np.array(train_image_features)
    train_labels = np.array(train_labels)
    scaler, pca, voting_clf = train_meta_classifier(train_image_features, train_labels)

    return scaler, pca, voting_clf