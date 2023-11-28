import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import image_dataset_from_directory, to_categorical
from tqdm import tqdm
from io import BytesIO


def initialize_model(train_ds):
    shape = None
    for images, labels in train_ds.take(1):
        # Access the first image
        first_image = images[0]
        shape = first_image.shape

    model = VGG19(weights="imagenet", include_top=False, input_shape=shape)

    #for layer in model.layers[-4:]:
    #    layer.trainable = True

    return model

def set_nontrainable_layers(model):
    model.trainable = False

    return model

def make_augmentation_layers():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomRotation(0.5),
        layers.RandomRotation(0.7),
    ])

def add_last_layers(model, class_names):
    '''Take a pre-trained model, set its parameters as non-trainable, and add additional trainable layers on top'''
    data_augmentation = make_augmentation_layers()
    base_model = set_nontrainable_layers(model)
    flatten_layer = layers.Flatten()
    #batch_norm_layer = layers.BatchNormalization()
    dense_layer = layers.Dense(500, activation='relu', kernel_regularizer=regularizers.l2(0.001))
    dense_layer2 = layers.Dense(200, activation='relu')
    #dense_layer = layers.Dense(500, activation='relu')
    #dropout_layer = layers.Dropout(0.3)
    prediction_layer = layers.Dense(len(class_names), activation='softmax')


    model = models.Sequential([
        base_model,
        data_augmentation,
        flatten_layer,
        #batch_norm_layer,
        dense_layer,
        dense_layer2,
        #dropout_layer,
        prediction_layer
    ])

    return model


def build_model(train_ds, class_names):
    model = initialize_model(train_ds)
    model = add_last_layers(model, class_names)

    opt = optimizers.Adam(learning_rate=1e-4)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy', tf.keras.metrics.F1Score(), tf.keras.metrics.Recall()])
    return model
