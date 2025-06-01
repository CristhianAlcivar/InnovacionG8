# src/covid_classifier/classify_tflite.py
import os
import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path

IMG_SIZE = (150, 150)
MODEL_PATH = Path(__file__).parent / 'model' / 'mobilenet_model.tflite'
LABELS = ['COVID', 'NORMAL', 'VIRAL_PNEUMONIA']

interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(image_path):
    img = cv2.imread(str(image_path))
    img = cv2.resize(img, IMG_SIZE)
    img = np.expand_dims(img / 255.0, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    label_index = np.argmax(output)
    return LABELS[label_index], float(output[0][label_index])

if __name__ == "__main__":
    test_img = Path(__file__).parent / 'data' / 'processed' / 'COVID' / os.listdir(Path(__file__).parent / 'data' / 'processed' / 'COVID')[0]
    pred, prob = predict(test_img)
    print(f"✅ Predicción: {pred} ({prob:.2f})")
