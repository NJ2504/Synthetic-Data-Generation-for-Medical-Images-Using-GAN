# Importing Libraries
import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Importing Dataset
malignant_dir = r'C:\Users\navee\OneDrive\Desktop\UOG\AI Capstone Project\Implementation\isic-2024-challenge\new_labelled_image_data\1'  
benign_dir = r'C:\Users\navee\OneDrive\Desktop\UOG\AI Capstone Project\Implementation\isic-2024-challenge\new_labelled_image_data\0'        
synthetic_dir = r'C:\Users\navee\OneDrive\Desktop\UOG\AI Capstone Project\Implementation\isic-2024-challenge\new_labelled_image_data\FID_FINAL_IMGS'  
test_dir = r'C:\Users\navee\OneDrive\Desktop\UOG\AI Capstone Project\Implementation\isic-2024-challenge\new_labelled_image_data\Test Set'            
metadata_path = r'C:\Users\navee\OneDrive\Desktop\UOG\AI Capstone Project\Implementation\isic-2024-challenge\train-metadata.csv'

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
images = np.array(benign_images + malignant_images)
labels = np.array(benign_labels + malignant_labels)
images = images / 255.0
print(f"Total images: {images.shape[0]}")
print(f"Image shape: {images.shape[1:]}")

# Splitting Dataset
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.3, stratify=labels, random_state=25)

# Image Reshaping 
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)

# Model Building
svc = SVC(random_state=25)
svc.fit(X_train_flat, y_train)
y_val_pred = svc.predict(X_val_flat)

# Model Evaluation
accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {accuracy:.4f}")
report = classification_report(y_val, y_val_pred, target_names=['Benign', 'Malignant'])
print("Classification Report:\n", report)
conf_matrix = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="magma", xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Support Vector Classifier')
plt.show()

# Loading Test Images
metadata = pd.read_csv(metadata_path)
id_conversion = dict(zip(metadata['isic_id'], metadata['target']))
def load_test_images(test_dir, id_conversion, image_size=(140, 140)):
    images = []
    labels = []
    for filename in os.listdir(test_dir):
        if filename.endswith(".jpg"):
            isic_id = filename.split(".")[0]
            label = id_conversion.get(isic_id)
            if label is not None:
                img_path = os.path.join(test_dir, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, image_size)
                    images.append(img)
                    labels.append(label)
    return np.array(images), np.array(labels)
X_test, y_test = load_test_images(test_dir, id_conversion)
X_test = X_test / 255.0
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Evaluating Test Set
y_test_pred = svc.predict(X_test_flat)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy (SVC): {test_accuracy:.4f}")
test_report = classification_report(y_test, y_test_pred, target_names=['Benign', 'Malignant'])
print("Test Classification Report (SVM):\n", test_report)
test_conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(test_conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Test Confusion Matrix (Support Vector Classifier)')
plt.show()






