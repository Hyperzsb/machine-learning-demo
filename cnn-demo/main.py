import tensorflow as tf
import os
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def model_fun():
    model_ = models.Sequential()
    model_.add(Reshape((28, 28, 1), input_shape=(28, 28)))
    model_.add(Conv2D(32, (3, 3), activation='relu'))
    model_.add(MaxPooling2D((2, 2)))
    model_.add(Conv2D(64, (3, 3), activation='relu'))
    model_.add(MaxPooling2D((2, 2)))
    model_.add(Conv2D(64, (3, 3), activation='relu'))
    model_.add(Flatten())
    model_.add(Dense(64, activation='relu'))
    model_.add(Dense(10, activation='softmax'))
    return model_


if __name__ == '__main__':
    # gpu config
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)

    # load data
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # get CNN model
    model = model_fun()

    # show
    model.summary()

    # train
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images,
              train_labels,
              batch_size=128,
              epochs=5)

    # test
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(test_acc)
