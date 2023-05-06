import cv2
import numpy as np
import tensorflow as tf
import numpy as np
from PIL import ImageGrab

# Load the trained model
model = tf.keras.models.load_model('models/forest_model.h5')

def preprocess_images(images, target_size=(256, 256)):
    preprocessed_images = []
    for img in images:
        resized_img = cv2.resize(img, target_size)
        normalized_img = resized_img / 255.0
        preprocessed_images.append(normalized_img)
    return np.array(preprocessed_images)

#screen capture

while(True):
    printscreen_pil =  ImageGrab.grab()
    printscreen_numpy =   np.array(printscreen_pil.getdata(),dtype='uint8')\
    .reshape((printscreen_pil.size[1],printscreen_pil.size[0],3)) 
    printscreen_numpy = printscreen_numpy[82:1071, 265:1911]
    img = cv2.cvtColor(printscreen_numpy, cv2.COLOR_BGR2RGB)
    cv2.imshow('window',img)
    preprocessed_img = preprocess_images([img])

    # Make a prediction
    prediction = model.predict(preprocessed_img)

    # Get the predicted label
    print(prediction[0][0], prediction[0][1])
    if prediction[0][0] > prediction[0][1]:
        label = 'forested'
    else:
        label = 'deforested'

    print('Predicted label:', label)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

# Load and preprocess a new image
#img = cv2.imread('images/testing/1.jpg')

