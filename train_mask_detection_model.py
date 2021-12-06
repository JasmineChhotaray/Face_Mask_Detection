import argparse
from enum import auto
import os
import imp
from re import X
from sys import path
import numpy as np
from pathlib import Path
from scipy.ndimage.measurements import label
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import callbacks
from tensorflow.keras.models import Model
from tensorflow.python.framework.tensor_util import FastAppendBFloat16ArrayToTensorProto
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.layers.core import Flatten
from sklearn.metrics import classification_report


class TrainMaskDetector:

    def __init__(self, arg):
        self.train_image_folder = Path(arg['train'])
        self.test_image_folder = Path(arg['test'])
        self.validation_image_folder = Path(arg['validation'])

        # setting up the default parameters
        self.epochs = 20
        self.batchSize = 32

    def run(self):
        """
        Executes the functions
        """
        self.train_mask_detection()
        # self.evaluate_test_data()
        # self.predict_test_detection()

    def store_image_paths(self, path):
        file_paths = []

        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                file_paths.append(os.path.join(root, name))
        return file_paths

    def train_mask_detection(self):

        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest"
        )

        data_train = train_datagen.flow_from_directory(
            directory=self.train_image_folder,
            class_mode='categorical',
            shuffle=True,
            batch_size=self.batchSize,
            target_size=(224, 224)
        )

        val_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255
        )

        data_val = val_datagen.flow_from_directory(
            directory=self.validation_image_folder,
            class_mode='categorical',
            shuffle=False,
            batch_size=self.batchSize,
            target_size=(224, 224)
        )

        # Loading the pre-trained Model
        mobilenet = keras.applications.mobilenet_v2.MobileNetV2(
            weights='imagenet',        # use imagenet weights
            # pooling='avg',              # to flatten after covent layers
            include_top=False,            # only want the base of the Model
            input_shape=(224, 224, 3),
            # alpha=0.35                  # the amount of filters from original network, here we use only 35%
            # if 1, uses the model as it is.. If 1.5 uses more
        )

        # freeze base model to use imagenet weights!
        for layer in mobilenet.layers:
            layer.trainable = False

        X = keras.layers.AveragePooling2D(pool_size=(7, 7))(mobilenet.output)
        X = keras.layers.Flatten(name="flatten")(X)
        X = keras.layers.Dense(128, activation="relu")(X)
        X = keras.layers.Dropout(0.5)(X)
        prediction = keras.layers.Dense(2, activation="softmax")(X)
        model = Model(inputs=mobilenet.input, outputs=prediction)

        opt = keras.optimizers.Adam(learning_rate=1e-4, decay=1e-4/self.epochs)

        # compile the model
        model.compile(
            optimizer=opt,
            loss=keras.losses.categorical_crossentropy,
            metrics=["accuracy"]
        )

        # To prevent it from running forever
        callback = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            min_delta=0.0001,
            verbose=1,
            mode='auto',
            baseline=None,
            restore_best_weights=True
        )

        # fitting the model
        results = model.fit(
            data_train,
            epochs=20,
            verbose=1,
            # batch_size=self.batchSize,
            validation_data=data_val,
            # callbacks=[callback],
            # validation_split=0.3
        )

        # save the model
        model.save("trained_mask_model.h5")

        # Create classification report
        prediction = model.predict_generator(
            generator=data_val,
            verbose=1)
        y_pred = np.argmax(prediction, axis=1)
        print("Classification Report:")
        print(classification_report(data_val.classes, y_pred,
              target_names=os.listdir(self.validation_image_folder)))
        # plot the accuracy loss
        self.plot_loss_accuracy(results)

        return results

    def plot_loss_accuracy(self, results):
        """
        plot the training and loss accuracy
        """
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(results.history["loss"], label="train_loss")
        plt.plot(results.history["val_loss"], label="val_loss")
        plt.plot(results.history["accuracy"], label="train_acc")
        plt.plot(results.history["val_accuracy"], label="val_acc")

        plt.title("Training loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.show()

    def evaluate_test_data(self):
        model = keras.models.load_model("trained_mask_model.h5")

        test_data = keras.preprocessing.image.ImageDataGenerator().flow_from_directory(
            directory=self.test_image_folder,
            target_size=(224, 224),
            shuffle=True,
            classes=os.listdir(self.test_image_folder),
            class_mode='categorical'
        )

        print(model.evaluate(test_data))

    def predict_test_detection(self):
        # pre-process images in test folder and use our model to predict
        model = keras.models.load_model("trained_mask_model.h5")
        test_image_paths = self.store_image_paths(self.test_image_folder)
        for idx in range(0, len(test_image_paths)):
            image = keras.preprocessing.image.load_img(
                path=test_image_paths[idx], target_size=(224, 224))
            pic_array = keras.preprocessing.image.img_to_array(image)
            input_arr = np.array([pic_array])
            predictions = model.predict(input_arr)
            print(test_image_paths[idx], np.around(predictions))


if __name__ == "__main__":

    # construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train", type=str,
                        required=True, help="folder for train images")
    parser.add_argument("-te", "--test", type=str,
                        required=True, help="folder for test images")
    parser.add_argument("-val", "--validation", type=str,
                        required=True, help="folder for validation images")
    parser.add_argument("-m", "--model", type=str, default="maskdetector.model",
                        help="path to trained mask detector model")

    args = parser.parse_args()

    obj = TrainMaskDetector(vars(args))
    obj.run()
