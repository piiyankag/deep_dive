from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.utils import to_categorical

def preprocess_dataset(images, labels):
    return preprocess_input(images), labels

def preprocess_labels(image, label):
    return image, to_categorical(label, num_classes=17)
