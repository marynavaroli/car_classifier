import os
import math
from PIL import Image
import pandas as pd
import numpy as np
from skimage import io
from sklearn.neighbors import KNeighborsClassifier
import joblib

label_dict = ["Audi", "Hyundai", "Toyota"]

# Target size (all images will be resized to match)
target_size = (150, 125)  # (width, height)


classifier = joblib.load("knn_model.pkl")


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