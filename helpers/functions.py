import os
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO
import urllib

image_size = 224


def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width))
            for img_path in img_paths]

    img_array = np.array([img_to_array(img) for img in imgs])

    imgs = [load_img(img_path, target_size=(img_height, img_width))
            for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    output = tf.keras.applications.resnet50.preprocess_input(img_array)
    return(output)


def read_and_prep_image_from_url(url):
    image_url = tf.keras.utils.get_file('delete_name', origin=url)
    img = tf.keras.preprocessing.image.load_img(
        image_url, target_size=(224, 224))
    os.remove(image_url)
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    output = tf.keras.applications.resnet50.preprocess_input(img_array)
    return(output)
