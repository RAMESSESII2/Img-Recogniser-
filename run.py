import os
import argparse

ap = argparse.ArgumentParser()

args = ap.parse_args()

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

import cv2

import numpy as np


m = Sequential()

def load_data(image_dr):
    #train_imgs_paths = [ os.path.join(image_dr, 'train/',img_path) for  img_path in os.listdir(os.path.join(image_dr, 'train/'))]
    #test_imgs_paths = [ os.path.join(image_dr, 'test/',img_path) for  img_path in os.listdir(os.path.join(image_dr, 'test/'))]
    # print(train_imgs_paths)
    # print(test_imgs_paths)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            os.path.join(image_dr, 'train/'),
            target_size=(200, 200),
            batch_size=20,
            class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
            os.path.join(image_dr, 'test/'),
            target_size=(200, 200),
            batch_size=32,
            class_mode='binary')
        
    return train_generator, validation_generator

def build_model():
    m.add(Conv2D(64, (3,3), input_shape=(200, 200, 3), activation='sigmoid'))

    m.add(MaxPooling2D(pool_size=(5,5)))

    m.add(Flatten())

    m.add(Dense(32, activation='sigmoid'))
    m.add(Dropout(0.4))
    m.add(Dense(1, activation='sigmoid'))

    m.compile(
        optimizer='adam',
        metrics=['accuracy'],
        loss='binary_crossentropy'
    )

def train_model(train_data, test_data):
    m.fit_generator(
        train_data,
        steps_per_epoch=2,
        epochs=1,
        validation_data=test_data,
        validation_steps=20,
        use_multiprocessing=False,
        workers=4
    )

def predict(image_dr):
    img_paths = [ os.path.join(image_dr, 'predict/',img_path) for  img_path in os.listdir(os.path.join(image_dr, 'predict/'))]

    images = [cv2.imread(img) for img in img_paths]
    images = [ cv2.resize(img, (im_height, im_width, 3))for img in images]
    images = [ np.reshape(img, [1, im_height, im_width, 3]) for img in images]

    return [m.predict_classes(img) for img in images]
try:
    if __name__=='__main__':
        train_data, test_data = load_data('images')
        build_model()
        train_model(train_data, test_data)

        predictions=predict('images')
        print(predictions)

except KeyboardInterrupt:
    print("\nUser aborted! ")