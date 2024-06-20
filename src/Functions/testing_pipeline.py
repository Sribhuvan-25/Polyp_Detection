from guided_patch_extraction import *

def test_pipeline(scaler, pca, voting_clf, root_dir, output_dir):
    transform = get_transform()
    model_patch, model_full = load_models()

    # Directory and processing setup
    # root_dir = 'Testing/Images'
    # output_dir = 'Processed_Images_pred'
    os.makedirs(output_dir, exist_ok=True)

    all_predictions = []
    all_labels = []

    for subfolder in ['test/normal', 'train/normal']:
        folder_path = os.path.join(root_dir, subfolder)
        label = 0 if subfolder == 'normal_testing_images' else 1
        for image_file in os.listdir(folder_path):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, image_file)
                final_prediction, true_label = extract_and_visualize(image_path, label, model_patch, model_full, transform, output_dir, scaler, pca, voting_clf)
                all_predictions.append(final_prediction)
                all_labels.append(true_label)

    # Calculate and print accuracy, precision, and recall
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)

    # print(f"Completed processing. Check the output directory: {output_dir}")
    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall: {recall:.4f}")
    
    return accuracy, precision, recall
