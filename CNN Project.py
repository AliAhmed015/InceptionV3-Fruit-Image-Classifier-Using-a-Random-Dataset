import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import itertools
import random
import warnings

warnings.filterwarnings("ignore")

# Check if GPU is available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

print("Available GPU devices:", gpus)

# Load and preprocess data
def load_and_preprocess_data(directory):
    images, labels = [], []
    class_names = sorted(os.listdir(directory))  # Get class names
    class_map = {name: idx for idx, name in enumerate(class_names)}  # Map class names to indices
    
    for fruit in class_names:
        fruit_path = os.path.join(directory, fruit)
        for filename in os.listdir(fruit_path):
            if filename.lower().endswith(('.jpeg', '.jpg', '.png')):
                img_path = os.path.join(fruit_path, filename)
                try:
                    img = image.load_img(img_path, target_size=(128, 128))
                    img = image.img_to_array(img)
                    img = preprocess_input(img)
                    images.append(img)
                    labels.append(class_map[fruit])
                except Exception as e:
                    print(f"Error processing image: {e}")
    
    return np.array(images), np.array(labels), class_names

# Load dataset
directory = "D:\\GVSU\\CIS 378\\Final Project\\Dataset"
images, labels, class_names = load_and_preprocess_data(directory)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
# Fit the generator (optional but recommended for featurewise_center etc.)
datagen.fit(images)

# Split data
X_train_val, X_test, y_train_val, y_test = train_test_split(images, labels, test_size=0.15, random_state=42, stratify=labels)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.294, random_state=42, stratify=y_train_val)

# Load pre-trained InceptionV3 model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(class_names), activation='softmax')(x)

# Compile model
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model with augmentation
batch_size = 64
epochs = 10

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(X_train) // batch_size,
    validation_data=(X_val, y_val),
    epochs=epochs
)

# Fine-tune
for layer in base_model.layers[:249]:
    layer.trainable = False
for layer in base_model.layers[249:]:
    layer.trainable = True
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history_fine = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(X_train) // batch_size,
    validation_data=(X_val, y_val),
    epochs=10
)

# Save model
model.save('fruits_classifier.h5')

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.2f}")

# Plot Training & Validation Loss
def plot_loss(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training & Validation Loss')
    plt.show()

plot_loss(history)

# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

# Predict and visualize confusion matrix
y_pred = np.argmax(model.predict(X_test), axis=1)
plot_confusion_matrix(y_test, y_pred, class_names)

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=class_names))

# ROC Curve
def plot_roc_curve(y_true, y_pred_proba, num_classes):
    plt.figure(figsize=(10, 7))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true == i, y_pred_proba[:, i])
        plt.plot(fpr, tpr, label=f'Class {class_names[i]} (AUC = {auc(fpr, tpr):.2f})')
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

# Plot ROC Curve
y_pred_proba = model.predict(X_test)
plot_roc_curve(y_test, y_pred_proba, len(class_names))

# Data Distribution Visualization
def plot_data_distribution(y_train, y_test, class_names):
    plt.figure(figsize=(10, 5))
    train_counts = np.bincount(y_train)
    test_counts = np.bincount(y_test)
    
    X = np.arange(len(class_names))
    plt.bar(X - 0.2, train_counts, 0.4, label='Train')
    plt.bar(X + 0.2, test_counts, 0.4, label='Test')
    
    plt.xticks(X, class_names, rotation=45, ha='right')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Data Distribution')
    plt.show()

plot_data_distribution(y_train, y_test, class_names)
