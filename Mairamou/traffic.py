import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_test, y_test))

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []

    # Iterate through each category folder
    for category in range(NUM_CATEGORIES):
        category_path = os.path.join(data_dir, str(category))

        if not os.path.exists(category_path):
            continue

        # Iterate through all image files in the category directory
        for filename in os.listdir(category_path):
            img_path = os.path.join(category_path, filename)
            
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                continue  # Skip if the image cannot be read
            
            # Resize image
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            
            # Convert image to array and append
            images.append(img)
            labels.append(category)

    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model.
    """
    model = tf.keras.Sequential([
        # Convolutional layer with 32 filters, 3x3 kernel, and ReLU activation
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Second convolutional layer
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten the results for the dense layer
        tf.keras.layers.Flatten(),

        # Fully connected layer
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),  # Dropout to reduce overfitting

        # Output layer with NUM_CATEGORIES neurons (softmax activation for classification)
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # Compile model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
