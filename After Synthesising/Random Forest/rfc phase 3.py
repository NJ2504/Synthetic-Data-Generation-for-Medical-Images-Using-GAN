# Importing Libraries
import os
import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#Directories
malignant_dir = '/kaggle/input/skin-lesion-2/new_labelled_image_data/1'
benign_dir = '/kaggle/input/skin-lesion-2/new_labelled_image_data/0'
synthetic_dir = '/kaggle/input/skin-lesion-2/new_labelled_image_data/FID_FINAL_IMGS'
test_dir = '/kaggle/input/skin-lesion-2/new_labelled_image_data/Test Set'
metadata_path = '/kaggle/input/train-metadata/train-metadata.csv'

# Loading Images
def load_images(folder, label, image_size=(140, 140)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, image_size)
            images.append(img)
            labels.append(label)
    return images, labels

benign_images, benign_labels = load_images(benign_dir, 0)
malignant_images, malignant_labels = load_images(malignant_dir, 1)

# Loading Phase-3 Synthetic Data 
synthetic_images, synthetic_labels = load_images(synthetic_dir, 1)
synthetic_images_phase_3 = synthetic_images[:1560]
synthetic_labels_phase_3 = [1] * 1560
images_phase_3 = np.array(benign_images + malignant_images + synthetic_images_phase_3)
labels_phase_3 = np.array(benign_labels + malignant_labels + synthetic_labels_phase_3)

# Normalise and Reshaping
images_phase_3 = images_phase_3 / 255.0
images_flat_phase_3 = images_phase_3.reshape(images_phase_3.shape[0], -1)

# Data Splitting 
X_train, X_val, y_train, y_val = train_test_split(images_flat_phase_3, labels_phase_3, test_size=0.3, random_state=25, stratify=labels_phase_3)

# Model Building
rfc3 = RandomForestClassifier(random_state=25)
rfc3.fit(X_train, y_train)
y_val_pred = rfc3.predict(X_val)

# Model Evaluation 
accuracy = accuracy_score(y_val, y_val_pred)
classification_report = classification_report(y_val, y_val_pred, target_names=['Benign', 'Malignant'])
print("RFC - Phase 3")
print("----------------------------")
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report)
conf_matrix_1 = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_1, annot=True, fmt="d", cmap="cividis", xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Phase 3 Confusion Matrix (RFC)')
plt.show()

# Laoding Test Data 
metadata = pd.read_csv(metadata_path)
test_filenames = os.listdir(test_dir)
test_ids = [filename.split('.')[0] for filename in test_filenames]
test_metadata = metadata[metadata['isic_id'].isin(test_ids)]

# Mapping the images ids to their respective labels
id_conversion = dict(zip(test_metadata['isic_id'], test_metadata['target']))
test_labels = []
test_images = []
for filename in test_filenames:
    img_path = os.path.join(test_dir, filename)
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (140, 140))
        test_images.append(img)
        image_id = filename.split('.')[0]
        label = id_conversion.get(image_id)
        test_labels.append(label)

test_labels = np.array(test_labels)

# Normalising and Reshaping 
test_images = np.array(test_images) / 255.0
test_images_flat = test_images.reshape(test_images.shape[0], -1)
y_test_pred = rfc3.predict(test_images_flat)
test_accuracy = accuracy_score(test_labels, y_test_pred)
test_classification_report = classification_report(test_labels, y_test_pred, target_names=['Benign', 'Malignant'])
print("RFC - Phase 3 (Test Set)")
print("----------------------------")
print(f"Test Set Accuracy: {test_accuracy:.4f}")
print("Test Set Classification Report:\n", test_classification_report)

# Model Evaluation on Test Set
test_conf_matrix = confusion_matrix(test_labels, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(test_conf_matrix, annot=True, fmt="d", cmap="magma", xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Phase 3 Test Set Confusion Matrix (RFC)')
plt.show()