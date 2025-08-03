# train_model.py

import os
import numpy as np
import shutil
import tempfile
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Constants
DATASET_DIR = "dataset"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Create temp folders for split
temp_dir = tempfile.mkdtemp()
train_dir = os.path.join(temp_dir, "train")
val_dir = os.path.join(temp_dir, "val")
os.makedirs(train_dir), os.makedirs(val_dir)

# Split dataset into temp/train and temp/val
def split_data():
    for class_name in os.listdir(DATASET_DIR):
        class_path = os.path.join(DATASET_DIR, class_name)
        if not os.path.isdir(class_path): continue
        images = os.listdir(class_path)
        train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)
        for folder, image_list in [(train_dir, train_images), (val_dir, val_images)]:
            target_path = os.path.join(folder, class_name)
            os.makedirs(target_path, exist_ok=True)
            for img in image_list:
                src = os.path.join(class_path, img)
                dst = os.path.join(target_path, img)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)

print("🔁 Splitting data...")
split_data()
NUM_CLASSES = len(os.listdir(train_dir))

# Data generators
train_gen = ImageDataGenerator(rescale=1./255, rotation_range=30, zoom_range=0.2, shear_range=0.2, horizontal_flip=True)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(train_dir, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode="categorical")
val_data = val_gen.flow_from_directory(val_dir, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False)

# Model
base_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

print("🚀 Training model...")
history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

model.save("animal_classifier_mobilenetv2.h5")
print("✅ Model saved as 'animal_classifier_mobilenetv2.h5'")

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
