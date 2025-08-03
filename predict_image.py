# predict_image.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tkinter import Tk, filedialog

# Load trained model
model = tf.keras.models.load_model("animal_classifier_mobilenetv2.h5")

# Get class labels from your dataset folder
CLASS_NAMES = sorted(os.listdir("dataset"))

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = 100 * np.max(predictions)

    print(f"\n🖼️ Image: {img_path}")
    print(f"✅ Predicted Animal: {predicted_class} ({confidence:.2f}%)")

def open_file_dialog():
    Tk().withdraw()  # Hide root window
    file_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
    )
    return file_path

if __name__ == "__main__":
    print("🐾 Animal Classifier - Select an image file to classify")
    img_path = open_file_dialog()

    if img_path and os.path.isfile(img_path):
        predict_image(img_path)
    else:
        print("❌ No image selected or invalid file.")
