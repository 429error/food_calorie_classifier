import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import random

# --- Configuration & Setup ---
DATASET_ROOT = 'food_101_small' 
TRAIN_DIR = os.path.join(DATASET_ROOT, 'train')
VAL_DIR = os.path.join(DATASET_ROOT, 'validation')
MODEL_PATH = 'calorie_classifier_model.h5' # File where the trained model will be saved

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42

try:
    CLASS_NAMES = sorted([d for d in os.listdir(TRAIN_DIR) 
                          if os.path.isdir(os.path.join(TRAIN_DIR, d))])
    NUM_CLASSES = len(CLASS_NAMES)
    if NUM_CLASSES == 0:
        raise FileNotFoundError(f"No class folders found in: {TRAIN_DIR}")
except FileNotFoundError:
    print(f"\n[ERROR] Dataset root path not found. Expected: {TRAIN_DIR}")
    exit()

print(f"✅ Found {NUM_CLASSES} classes: {CLASS_NAMES}")

# IMPORTANT: Save the class names list for the prediction script to use later!
with open('class_names.txt', 'w') as f:
    for item in CLASS_NAMES:
        f.write(f"{item}\n")
print("Saved class names to 'class_names.txt'")


# --- Data Loading and Preprocessing ---
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    seed=SEED
)

validation_generator = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    seed=SEED
)

print("\n--- Building Model ---")
# --- CNN Model Definition (Same as before) ---
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5), 
    Dense(NUM_CLASSES, activation='softmax') 
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- Training the Model ---
print("\n--- Starting Model Training ---")
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)
print("--- Training Finished ---")

# --- SAVING THE MODEL ---
try:
    model.save(MODEL_PATH)
    print(f"\n Successfully saved trained model to: {MODEL_PATH}")
except Exception as e:
    print(f"\n[ERROR] Could not save the model. Details: {e}")
