import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import random
import sys

# --- Configuration ---
MODEL_PATH = 'calorie_classifier_model.h5' 
CLASS_NAMES_FILE = 'class_names.txt'
IMG_SIZE = (128, 128) 

# --- 1. Load Class Names and Calorie Map ---

try:
    # Load class names from the file saved by train_model.py
    with open(CLASS_NAMES_FILE, 'r') as f:
        CLASS_NAMES = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(CLASS_NAMES)} class names.")
except FileNotFoundError:
    print(f"[ERROR] Class names file '{CLASS_NAMES_FILE}' not found. Did you run train_model.py?")
    sys.exit()


CALORIE_MAP = {
    'chocolate_cake': 380, 
    'fish_and_chips': 700, 
    'hamburger': 270, 
    'ice_cream': 250,
    'pad_thai': 400, 
    'pizza': 280, 
    'ramen': 450,
    'sushi': 350, 
    'tacos': 400, 
    'chicken_curry': 350
} 


# --- 2. Load the Trained Model ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Successfully loaded model from: {MODEL_PATH}")
except Exception as e:
    print(f"\n[ERROR] Could not load model. Ensure '{MODEL_PATH}' exists.")
    print(f"Details: {e}")
    sys.exit()


# --- 3. Prediction Function ---
def estimate_calories(img_path):
    """Predicts food class and estimates calories based on the CALORIE_MAP."""
    if not os.path.exists(img_path):
        print(f"[ERROR] Image file not found: {img_path}")
        return None, None

    # Load and preprocess the image 
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 # Normalize

    # Make prediction
    predictions = model.predict(img_array, verbose=0)

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions[0])

    # Map the index back to the class name
    predicted_class_name = CLASS_NAMES[predicted_class_index]

    # Look up the estimated calorie from the map
    estimated_calories = CALORIE_MAP.get(predicted_class_name, 'N/A')

    print("\n" + "=" * 40)
    print(f"INPUT IMAGE: {os.path.basename(img_path)}")
    print(f"Predicted Food: **{predicted_class_name.replace('_', ' ').title()}**")
    print(f"Confidence: {predictions[0][predicted_class_index]:.4f}")
    print(f"Estimated Calories: **{estimated_calories} kcal**")
    print("=" * 40)

    return predicted_class_name, estimated_calories

# --- 4. Example Usage ---

TEST_IMAGE_PATH = 'food_101_small/test/pizza/141507.jpg' 

# 2. You can also prompt the user for an image path:
# user_image_path = input("Enter the path to the food image you want to analyze: ")
# estimate_calories(user_image_path)

# RUN THE TEST
print("\n--- Running Prediction Test ---")
estimate_calories(TEST_IMAGE_PATH)
