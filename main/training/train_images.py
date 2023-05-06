import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf

def preprocess_images(images, target_size=(256, 256)):
    preprocessed_images = []
    for img in images:
        resized_img = cv2.resize(img, target_size)
        normalized_img = resized_img / 255.0
        preprocessed_images.append(normalized_img)
    return np.array(preprocessed_images)

def load_data(forested_dir, deforested_dir, target_size=(256, 256)):
    forested_images = []
    deforested_images = []

    for filename in os.listdir(forested_dir):
        img = cv2.imread(os.path.join(forested_dir, filename))
        if img is not None:
            img = cv2.resize(img, target_size)
            forested_images.append(img)

    for filename in os.listdir(deforested_dir):
        img = cv2.imread(os.path.join(deforested_dir, filename))
        if img is not None:
            img = cv2.resize(img, target_size)
            deforested_images.append(img)

    X = forested_images + deforested_images
    y = [0] * len(forested_images) + [1] * len(deforested_images)  # 0: forested, 1: deforested

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

forested_dir = 'images/forested'
deforested_dir = 'images/deforested'

X_train, X_test, y_train, y_test = load_data(forested_dir, deforested_dir)

# Preprocess images
X_train_preprocessed = preprocess_images(X_train)
X_test_preprocessed = preprocess_images(X_test)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.summary()

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

history = model.fit(X_train_preprocessed, y_train, epochs=50, validation_data=(X_test_preprocessed, y_test))

model.save('models/forest_model.h5')

test_loss, test_acc = model.evaluate(X_test_preprocessed, y_test, verbose=2)
print('\nTest accuracy:', test_acc)
