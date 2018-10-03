import numpy as np
import cv2 as cv
import os
from keras.preprocessing.image import img_to_array
from com.image.images.network.Model import Model
import keras as ks

train_images = []
train_labels = []
data_path = "/images/train_images/"


def load_data():
    filename = os.listdir(os.getcwd() + data_path)
    for file in filename:
        image = cv.imread(os.getcwd() + data_path + file)
        if image is not None:
            image = cv.resize(image, (28, 28))
            image = img_to_array(image)
            train_images.append(image)
            label = file.split("_")[0]
            if "*" in label:
                train_labels.append(10)
            elif "div" in label:
                train_labels.append(11)
            elif "+" in label:
                train_labels.append(12)
            elif "-" in label:
                train_labels.append(13)
            elif "=" in label:
                train_labels.append(14)
            else:
                train_labels.append(int(label))


def build_and_train_model():
    model = Model.buildModel(train_images[0].shape)
    model.compile(optimizer=ks.optimizers.Adam(lr=0.001), loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=25)
    model.save("my_model")


if __name__ == "__main__":
    load_data()
    train_images = np.array(train_images, dtype="float") / 255.0
    train_labels = np.array(train_labels)
    print(train_images[0].shape)
    build_and_train_model()
