from __future__ import absolute_import, division, print_function, unicode_literals
import functools
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


# Create dataset by CSV files
def get_dataset(file_path):
    label_column = 'survived'
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=12,
        label_name=label_column,
        na_value="?",
        num_epochs=1,
        ignore_errors=True)
    return dataset


def process_continuous_data(mean, data):
    data = tf.cast(data, tf.float32) * 1 / (2 * mean)
    return tf.reshape(data, [-1, 1])


# Create model
def create_model(preprocess_layer_):
    model_ = tf.keras.Sequential([
        preprocess_layer_,
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    model_.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    return model_


if __name__ == '__main__':
    # Download CSV files
    TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
    TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"
    train_file_path = tf.keras.utils.get_file("train.tor-csv", TRAIN_DATA_URL)
    test_file_path = tf.keras.utils.get_file("eval.tor-csv", TEST_DATA_URL)
    # Make data more readable
    np.set_printoptions(precision=3, suppress=True)
    raw_train_data = get_dataset(train_file_path)
    raw_test_data = get_dataset(test_file_path)

    # Show raw_train_data
    # examples, labels = next(iter(raw_train_data))
    # print("EXAMPLES: \n", examples, "\n")
    # print("LABELS: \n", labels)
    '''
    OrderedDict([
        ('sex', <tf.Tensor: shape=(12,), dtype=string, numpy=array([b'male', b'male', b'male', b'female', b'male', b'male', b'male', b'male', b'male', b'male', b'male', b'male'], dtype=object)>),
        ('age', <tf.Tensor: shape=(12,), dtype=float32, numpy=array([28. , 40.5, 48. , 28. , 28. , 35. , 18. , 28. , 47. , 28. , 22. , 28. ], dtype=float32)>), 
        ('n_siblings_spouses', <tf.Tensor: shape=(12,), dtype=int32, numpy=array([0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0])>), 
        ('parch', <tf.Tensor: shape=(12,), dtype=int32, numpy=array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])>), 
        ('fare', <tf.Tensor: shape=(12,), dtype=float32, numpy=array([ 26.   ,   7.75 ,  52.   ,  51.862,  30.5  ,   8.05 ,  20.212, 221.779,  15.   ,   7.896,   7.25 ,   8.05 ], dtype=float32)>), 
        ('class', <tf.Tensor: shape=(12,), dtype=string, numpy=array([b'First', b'Third', b'First', b'First', b'First', b'Third', b'Third', b'First', b'Second', b'Third', b'Third', b'Third'], dtype=object)>), 
        ('deck', <tf.Tensor: shape=(12,), dtype=string, numpy=array([b'A', b'unknown', b'C', b'D', b'C', b'unknown', b'unknown', b'C', b'unknown', b'unknown', b'unknown', b'unknown'], dtype=object)>), 
        ('embark_town', <tf.Tensor: shape=(12,), dtype=string, numpy=array([b'Southampton', b'Queenstown', b'Southampton', b'Southampton', b'Southampton', b'Southampton', b'Southampton', b'Southampton', b'Southampton', b'Southampton', b'Southampton', b'Southampton'], dtype=object)>), 
        ('alone', <tf.Tensor: shape=(12,), dtype=string, numpy=array([b'y', b'y', b'n', b'n', b'y', b'y', b'n', b'y', b'y', b'y', b'n', b'y'], dtype=object)>)    
    ]) 
    '''

    # 分类型数据列构建
    CATEGORIES = {
        'sex': ['male', 'female'],
        'class': ['First', 'Second', 'Third'],
        'deck': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
        'embark_town': ['Cherbourg', 'Southhampton', 'Queenstown'],
        'alone': ['y', 'n']
    }
    categorical_columns = []
    for feature, vocab in CATEGORIES.items():
        cat_col = tf.feature_column.categorical_column_with_vocabulary_list(key=feature, vocabulary_list=vocab)
        categorical_columns.append(tf.feature_column.indicator_column(cat_col))

    # 连续型数据列构建
    MEANS = {
        'age': 29.631308,
        'n_siblings_spouses': 0.545455,
        'parch': 0.379585,
        'fare': 34.385399
    }
    numerical_columns = []
    for feature in MEANS.keys():
        num_col = tf.feature_column.numeric_column(feature,
                                                   normalizer_fn=functools.partial(process_continuous_data,
                                                                                   MEANS[feature]))
        numerical_columns.append(num_col)

    # 数据预处理层
    preprocess_layer = tf.keras.layers.DenseFeatures(categorical_columns + numerical_columns)

    # Create model
    model = create_model(preprocess_layer)

    # Create checkpoint config
    checkpoint_path = 'csv-saved-weights/checkpoint.ckpt'
    saved_path = 'csv-saved-weights'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    # Load model's weights or train model
    if os.path.exists(saved_path) & os.path.exists(checkpoint_path + '.index'):
        model.load_weights(checkpoint_path)
    else:
        model.fit(raw_train_data, epochs=15, callbacks=[cp_callback])
        model.save_weights(checkpoint_path.format(epoch=0))

    # Test the model
    test_loss, test_accuracy = model.evaluate(raw_test_data)
    print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))
