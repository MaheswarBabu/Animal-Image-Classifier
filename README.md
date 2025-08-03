
# Animal Image Classifier

# Objective
A deep learning-based image classification system that identifies animals in photos using a pre-trained MobileNetV2 model.

# Technologies Used
- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-learn
- Tkinter (for image file upload)

# Features
- Supports 15 animal classes (e.g., Cat, Dog, Lion, Zebra, etc.)
- Transfer Learning with MobileNetV2 for efficient training
- File picker GUI for uploading test images
- Displays predicted animal name with confidence score
- No modification to original dataset

# Dataset Format
The dataset folder contains 15 subfolders:
```
dataset/
├── Cat/
├── Dog/
├── Lion/
├── Zebra/
├── ...
```

# How to Use

# Setup Environment
```bash
python -m venv animal_env
animal_env\Scripts\activate  # On Windows
# source animal_env/bin/activate  # On Linux/macOS

pip install tensorflow matplotlib scikit-learn
```

# Train the Model
```bash
python train_model.py
```
This will:
- Split your dataset (in memory)
- Train a MobileNetV2 classifier
- Save the model as `animal_classifier_mobilenetv2.h5`

# Run Image Prediction
```bash
python predict_image.py
```
This will:
- Open a file upload dialog

<img width="936" height="580" alt="Screenshot 2025-08-03 164758" src="https://github.com/user-attachments/assets/21ad90d7-1224-4598-a201-48306ee3e28a" />
Choosen image:

![Alt Text](test1.jepg)

- Display the predicted animal and confidence

<img width="1458" height="114" alt="Screenshot 2025-08-03 164813" src="https://github.com/user-attachments/assets/6a7f2b2e-e786-498c-bc8d-ceb5de26a31b" />


# Future Enhancements
- Add Grad-CAM visualization for model explainability
- Integrate with webcam for live prediction
- Deploy as a web app using Streamlit or Gradio
- Export predictions to CSV or JSON
