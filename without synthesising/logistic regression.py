# Importing Libraries
import os
import pandas as pd
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Importing Directories
benign_dir = r'C:\Users\navee\OneDrive\Desktop\UOG\AI Capstone Project\Implementation\isic-2024-challenge\new_labelled_image_data\0'
malignant_dir = r'C:\Users\navee\OneDrive\Desktop\UOG\AI Capstone Project\Implementation\isic-2024-challenge\new_labelled_image_data\1'
test_dir = r'C:\Users\navee\OneDrive\Desktop\UOG\AI Capstone Project\Implementation\isic-2024-challenge\new_labelled_image_data\Test Set'
metadata_dir = r'C:/users/navee/OneDrive/Desktop/UOG/AI Capstone Project/Implementation/isic-2024-challenge/train-metadata.csv'
metadata = pd.read_csv(metadata_dir, low_memory=False)

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

# Splitting Data
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.3, stratify=labels, random_state=25)

# Image Reshaping 
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train_flat, y_train)
y_pred = lr.predict(X_val_flat)
print("Accuracy:", accuracy_score(y_val, y_pred))
print("Classification report: ",{classification_report(y_val, y_pred)})
print("Confusion matrix: ",{confusion_matrix(y_val, y_pred)})

# Loading Test Images
id_to_label = dict(zip(metadata['isic_id'], metadata['target']))
test_images = []
test_labels = []
for filename in os.listdir(test_dir):
    img_path = os.path.join(test_dir, filename)
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (140, 140))
        test_images.append(img)
        isic_id = filename.split('.')[0]
        label = id_to_label.get(isic_id)
        test_labels.append(label)
test_images = np.array(test_images) / 255.0  
test_images_flat = test_images.reshape(test_images.shape[0], -1)
test_labels = np.array(test_labels)

# Evaluating the Test Set
y_pred = lr.predict(test_images_flat)
print("Accuracy:", accuracy_score(test_labels, y_pred))
print("Classification report: ",{classification_report(test_labels, y_pred)})
print("Confusion matrix: ",{confusion_matrix(test_labels, y_pred)})

