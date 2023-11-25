from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import requests
from PIL import Image
from io import BytesIO

def preprocess_dataset(images, labels):
    return preprocess_input(images), labels

def preprocess_labels(image, label):
    return image, to_categorical(label, num_classes=17)

def load_and_preprocess_image(img_path):
    # Download the image
    response = requests.get(img_path)
    img = Image.open(BytesIO(response.content))

    return preprocess_image(img)


def preprocess_image(img_bytes):
    img_file = BytesIO(img_bytes)

    with Image.open(img_file) as img:
        # Resize the image to match the model's expected input size
        img = img.resize((256, 256))
        # Convert the image to a numpy array
        img_array = img_to_array(img)

        # Expand dimensions to match the model's expected input format
        img_array = np.expand_dims(img_array, axis=0)

        # Preprocess the image
        img_array = preprocess_input(img_array)

        return img_array
