# Importing Libraries
import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
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

# Splitting Data
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.3, stratify=labels, random_state=25)

# Image reshaping
X_train = X_train.reshape(-1, 140, 140, 3)
X_val = X_val.reshape(-1, 140, 140, 3)

# Model Building
def cnn_model(input_shape=(140, 140, 3)):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(224, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = cnn_model(input_shape=(140, 140, 3))
model.summary()

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Model Evaluation
y_val_pred_cnn = model.predict(X_val)
y_val_pred_cnn = (y_val_pred_cnn > 0.5).astype(int).flatten()
val_accuracy_cnn = accuracy_score(y_val, y_val_pred_cnn)
print(f"Accuracy (CNN): {val_accuracy_cnn:.4f}")
report_cnn = classification_report(y_val, y_val_pred_cnn, target_names=['Benign', 'Malignant'])
print("Classification Report (CNN):\n", report_cnn)
conf_matrix_cnn = confusion_matrix(y_val, y_val_pred_cnn)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_cnn, annot=True, fmt="d", cmap="Blues", xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (CNN)')
plt.show()

# Loading Test Data
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

# Test Data evaluation 
y_test_pred_cnn = model.predict(X_test_flat.reshape(-1, 140, 140, 3))  
y_test_pred_cnn = (y_test_pred_cnn > 0.5).astype(int).flatten()
test_accuracy_cnn = accuracy_score(y_test, y_test_pred_cnn)
print(f"Test Accuracy (CNN): {test_accuracy_cnn:.4f}")
test_report_cnn = classification_report(y_test, y_test_pred_cnn, target_names=['Benign', 'Malignant'])
print("Test Classification Report (CNN):\n", test_report_cnn)
test_conf_matrix_cnn = confusion_matrix(y_test, y_test_pred_cnn)
plt.figure(figsize=(8, 6))
sns.heatmap(test_conf_matrix_cnn, annot=True, fmt="d", cmap="Blues", xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Test Confusion Matrix (CNN)')
plt.show()