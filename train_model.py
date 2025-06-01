# src/covid_classifier/train_model.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from pathlib import Path

IMG_SIZE = (150, 150)
BATCH_SIZE = 8
EPOCHS = 10
DATA_DIR = Path(__file__).parent / 'data' / 'processed'
MODEL_DIR = Path(__file__).parent / 'model'
MODEL_DIR.mkdir(exist_ok=True)

def create_model():
    base = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
    base.trainable = False
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    datagen = ImageDataGenerator(rescale=1/255., validation_split=0.2,
                                 horizontal_flip=True, rotation_range=10)

    train = datagen.flow_from_directory(DATA_DIR, target_size=IMG_SIZE,
                                        batch_size=BATCH_SIZE, class_mode='categorical',
                                        subset='training')
    val = datagen.flow_from_directory(DATA_DIR, target_size=IMG_SIZE,
                                      batch_size=BATCH_SIZE, class_mode='categorical',
                                      subset='validation')

    model = create_model()
    model.fit(train, validation_data=val, epochs=EPOCHS)
    model.save(MODEL_DIR / 'mobilenet_model.h5')

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(MODEL_DIR / 'mobilenet_model.tflite', 'wb') as f:
        f.write(tflite_model)
    print("✅ Exportado .h5 y .tflite")

if __name__ == "__main__":
    main()
