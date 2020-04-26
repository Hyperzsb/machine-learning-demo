# TensorFlow and tensorflow.keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import random
# GPU
import os


# Create a new model
def create_model():
    # Create the model
    new_model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)
    ])
    # Compile the model
    new_model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

    # Show model summary
    new_model.summary()
    # Return model
    return new_model


if __name__ == '__main__':
    # Tensorflow version
    print('Tensorflow: ', tf.__version__)

    # Enable GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)

    # Load dataset
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Preprocess dataset
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Create model
    model = create_model()

    # Create checkpoint config
    checkpoint_path = 'basic-saved-weights/checkpoint.ckpt'
    saved_path = 'basic-saved-weights'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    # Load model's weights or train model
    if os.path.exists(saved_path) & os.path.exists(checkpoint_path + '.index'):
        model.load_weights(checkpoint_path)
    else:
        model.fit(train_images, train_labels, epochs=15, callbacks=[cp_callback])
        model.save_weights(checkpoint_path.format(epoch=0))

    # # Create model saving config
    # model_path = 'basic-saved-model.h5'
    # # Load model or train model
    # if os.path.exists(model_path):
    #     model = keras.models.load_model(model_path)
    #     model.summary()
    # else:
    #     model = create_model()
    #     model.fit(train_images, train_labels, epochs=15)
    #     model.save(model_path)

    # Test the model
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)

    # Prediction model
    probability_model = tf.keras.Sequential([model, keras.layers.Softmax()])

    # Prediction examples
    # for loop in range(100):
    #     i = random.randint(0, 10000)
    #     single_img = np.expand_dims(test_images[i], 0)
    #     prediction_res = probability_model.predict(single_img)
    #     print('Prediction index: ', np.argmax(prediction_res[0]),
    #           'Prediction class: ', class_names[int(np.argmax(prediction_res[0]))])
    #     print('Reality index: ', test_labels[i],
    #           'Reality class: ', class_names[test_labels[i]], '\n')
