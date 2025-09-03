import os
import math
from PIL import Image
import pandas as pd
import numpy as np
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Path to folder containing images
folder_path1 = "Cars Dataset/train/Audi"
folder_path2 = "Cars Dataset/train/Hyundai Creta"
folder_path3 = "Cars Dataset/train/Toyota Innova"

label_dict = ["Audi", "Hyundai", "Toyota"]

# Target size (all images will be resized to match)
target_size = (150, 125)  # (width, height)

images = []
labels = []
for filename in os.listdir(folder_path1):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(folder_path1, filename)
        img = Image.open(img_path).convert('RGB')  # RGB keeps 3 channels
        img = img.resize(target_size)              # Ensure consistent size
        img_array = np.array(img, dtype=np.float32)                 # Shape: (H, W, 3)
        images.append(img_array)
        labels.append(0)

for filename in os.listdir(folder_path2):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(folder_path2, filename)
        img = Image.open(img_path).convert('RGB')  # RGB keeps 3 channels
        img = img.resize(target_size)              # Ensure consistent size
        img_array = np.array(img, dtype=np.float32)                 # Shape: (H, W, 3)
        images.append(img_array)
        labels.append(1)

for filename in os.listdir(folder_path3):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(folder_path3, filename)
        img = Image.open(img_path).convert('RGB')  # RGB keeps 3 channels
        img = img.resize(target_size)              # Ensure consistent size
        img_array = np.array(img, dtype=np.float32)                  # Shape: (H, W, 3)
        images.append(img_array)
        labels.append(2)

#print(labels)
# Convert to 4D NumPy array: (num_images, height, width, channels)
images_array = np.array(images, dtype=np.float32) 
n, h, w, c = images_array.shape
print(images_array.shape) 
#print(images_array)

images_reshaped = images_array.reshape(n, h * w * c) / 255.0
#print(images_array[0])
print(images_reshaped[0])
print(labels[0])

training_data, validation_data, training_labels, validation_labels = train_test_split(images_reshaped, labels, test_size=0.2, random_state=100)
print(len(training_data))
print(len(training_labels))

classifier = KNeighborsClassifier(n_neighbors = 1)
classifier.fit(training_data, training_labels)
scoreValidation = (classifier.score(validation_data, validation_labels))

def transform_image(image):
    if image is None:
        raise ValueError("No image provided to transform_image")
    test_array = []
    img = image.convert('RGB')  # RGB keeps 3 channels
    img = img.resize(target_size) 
    test_array = np.array(img, dtype=np.float32) 
    print(test_array)
    h, w, c = test_array.shape
    test_reshaped = test_array.reshape(1, h * w * c) / 255.0
    return test_reshaped

def classify_image(image):
    print(image)
    image_reshaped = transform_image(image)
    print(image_reshaped)
    prediction = classifier.predict(image_reshaped)
    print(prediction)
    prediction_str = label_dict[prediction[0]]
    return prediction_str