from __future__ import absolute_import, division, print_function, unicode_literals
import functools
import os
import tensorflow as tf


# Create dataset by CSV files
def get_dataset(file_path):
    selected_columns = ['FlowIATMean', 'FlowIATStd', 'FlowIATMax', 'FlowIATMin',
                        'FwdIATMean', 'FwdIATStd', 'FwdIATMax', 'FwdIATMin',
                        'BwdIATMean', 'BwdIATStd', 'BwdIATMax', 'BwdIATMin', 'label']
    label_column = 'label'
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=15,
        select_columns=selected_columns,
        label_name=label_column,
        na_value="?",
        num_epochs=1,
        ignore_errors=True)
    return dataset


def process_continuous_data(mean, data=0):
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
    train_file_path = 'tor-csv/train.csv'
    test_file_path = 'tor-csv/eval.csv'
    raw_train_data = get_dataset(train_file_path)
    raw_test_data = get_dataset(test_file_path)

    # 连续型数据列构建
    MEANS = {
        'FlowIATMean': 317667.6291, 'FlowIATStd': 221411.7258, 'FlowIATMax': 902211.6844,
        'FlowIATMin': 194286.8958,
        'FwdIATMean': 352298.1217, 'FwdIATStd': 230758.2655, 'FwdIATMax': 873455.3231,
        'FwdIATMin': 212625.6526,
        'BwdIATMean': 120670.5607, 'BwdIATStd': 119661.6554, 'BwdIATMax': 477679.2937,
        'BwdIATMin': 52154.79133
    }
    numerical_columns = []
    for feature in MEANS.keys():
        num_col = tf.feature_column.numeric_column(feature, normalizer_fn=functools.partial(process_continuous_data,
                                                                                            MEANS[feature]))
        numerical_columns.append(num_col)

    # 数据预处理层
    preprocess_layer = tf.keras.layers.DenseFeatures(numerical_columns)

    # Create model
    model = create_model(preprocess_layer)

    # Create checkpoint config
    checkpoint_path = 'tor-csv-saved-weights/checkpoint.ckpt'
    saved_path = 'tor-csv-saved-weights'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    # Load model's weights or train model
    if os.path.exists(saved_path) & os.path.exists(checkpoint_path + '.index'):
        model.load_weights(checkpoint_path)
    else:
        model.fit(raw_train_data, epochs=15, callbacks=[cp_callback])
        model.save_weights(checkpoint_path.format(epoch=0))

    # # Create model saving config
    # model_path = 'tor-csv-saved-model.h5'
    # # Load model or train model
    # if os.path.exists(model_path):
    #     model = tf.keras.models.load_model(model_path)
    #     model.summary()
    # else:
    #     model = create_model(preprocess_layer)
    #     model.fit(raw_train_data, epochs=15)
    #     model.save(model_path)

    # Test the model
    print('\nEvaluate the model:')
    test_loss, test_accuracy = model.evaluate(raw_test_data)
