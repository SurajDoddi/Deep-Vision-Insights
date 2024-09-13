import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50, VGG16, VGG19, InceptionV3
from tensorflow.keras.preprocessing import image

# Function to load and preprocess images
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)  # Resize for model compatibility
    img = img.astype(np.float32)
    img /= 255.0  # Normalize between 0 and 1
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to extract features from a specific layer
def extract_features(model, image_path, layer_name='block3_conv3'):
    img = load_and_preprocess_image(image_path)
    intermediate_layer_model = model.get_layer(layer_name)
    features = intermediate_layer_model.predict(img)
    features = features.reshape((features.shape[1], features.shape[2], features.shape[3]))  # Reshape for visualization
    return features

# Function to visualize features as a heatmap
def visualize_features(features):
    plt.imshow(features, cmap='viridis')
    plt.colorbar()
    plt.show()

# Load Stanford Cars dataset (replace with your dataset path)
dataset_path = '/path/to/StanfordCars/dataset'

# Choose random images
random_images = random.sample(os.listdir(dataset_path), 5)

# Pre-trained models to use
models = [ResNet50(weights='imagenet'), VGG16(weights='imagenet'), VGG19(weights='imagenet'), InceptionV3(weights='imagenet')]

# Layer names for feature extraction (adjust based on model architecture)
layer_names = {
    ResNet50: ['conv1_relu', 'conv4_block6_out', 'conv5_block3_out'],
    VGG16: ['block2_conv2', 'block4_conv3', 'block5_conv4'],
    VGG19: ['block2_conv2', 'block4_conv3', 'block5_conv4'],
    InceptionV3: ['mixed_0', 'mixed_7', 'mixed_10']
}

# Extract and visualize features
for image_path in random_images:
    full_path = os.path.join(dataset_path, image_path)
    for model, layer_names_model in zip(models, layer_names.values()):
        for layer_name in layer_names_model:
            features = extract_features(model, full_path, layer_name)

            # Visualize features as a heatmap
            visualize_features(features)

            # Print informative message
            print(f"Image: {image_path}, Model: {model.name}, Layer: {layer_name}, Feature Shape: {features.shape}")