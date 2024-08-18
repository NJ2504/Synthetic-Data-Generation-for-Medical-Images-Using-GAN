# Importing Libraries
import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3 # type: ignore
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

# Loading Phase-5 Synthetic Data 
synthetic_images, synthetic_labels = load_images(synthetic_dir, 1)
synthetic_images_phase_5 = synthetic_images[:2600]
synthetic_labels_phase_5 = [1] * 2600
images_phase_5 = np.array(benign_images + malignant_images + synthetic_images_phase_5)
labels_phase_5 = np.array(benign_labels + malignant_labels + synthetic_labels_phase_5)
images_phase_5 = np.array(benign_images + malignant_images + synthetic_images_phase_5)
labels_phase_5 = np.array(benign_labels + malignant_labels + synthetic_labels_phase_5)

# Normalisation
images_phase_5 = images_phase_5 / 255.0

# Splitting the Data
X_train, X_val, y_train, y_val = train_test_split(images_phase_5, labels_phase_5, test_size=0.3, random_state=25, stratify=labels_phase_5)

# Image reshaping
X_train = X_train.reshape(-1, 140, 140, 3)
X_val = X_val.reshape(-1, 140, 140, 3)

# Model Building
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(140, 140, 3))
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Model Evaluation
y_val_pred = model.predict(X_val)
y_val_pred = (y_val_pred > 0.5).astype(int).flatten() 
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Accuracy (InceptionV3): {val_accuracy:.4f}")
val_report = classification_report(y_val, y_val_pred, target_names=['Benign', 'Malignant'])
print("Classification Report (InceptionV3):\n", val_report)

val_conf_matrix = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(val_conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (InceptionV3)')
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
y_test_pred = model.predict(X_test)
y_test_pred = (y_test_pred > 0.5).astype(int).flatten()
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy (InceptionV3): {test_accuracy:.4f}")
test_report = classification_report(y_test, y_test_pred, target_names=['Benign', 'Malignant'])
print("Test Classification Report (InceptionV3):\n", test_report)
test_conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(test_conf_matrix, annot=True, fmt="d", cmap="magma", xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Test Confusion Matrix (InceptionV3)')
plt.show()