import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import joblib
import pandas as pd
import os

original_train_dataset = pd.read_csv('train.csv')
original_test_dataset = pd.read_csv('eval.csv')

# 去掉无用字段
original_train_dataset.drop(['SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort',
                             'Protocol', 'FlowDuration', 'FlowBytes/s', 'FlowPackets/s'], 1, inplace=True)
original_test_dataset.drop(['SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort',
                            'Protocol', 'FlowDuration', 'FlowBytes/s', 'FlowPackets/s'], 1, inplace=True)

# 由于是非监督学习，不使用label
train_dataset = np.array(original_train_dataset.drop(['label'], 1).astype(float))
test_dataset = np.array(original_test_dataset.drop(['label'], 1).astype(float))
# 将每一列特征标准化为标准正态分布，注意，标准化是针对每一列而言的
train_dataset = preprocessing.scale(train_dataset)
test_dataset = preprocessing.scale(test_dataset)

# K取2进行聚类训练或加载模型
model_saved_path = 'k-means-model.m'
if os.path.exists(model_saved_path):
    print('Loading existed model')
    kmeans = joblib.load(model_saved_path)
    print('Loading complete')
else:
    print('Training new model')
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(train_dataset)
    joblib.dump(kmeans, model_saved_path)
    print('Training complete')

# 训练集的正确率
print('Start evaluate model in train dataset')
train_reality_res = np.array(original_train_dataset['label'])
train_correct = 0
for i in range(len(train_dataset)):
    train_prediction_dataset = np.array(train_dataset[i].astype(float))
    train_prediction_dataset = train_prediction_dataset.reshape(-1, len(train_prediction_dataset))
    train_prediction_res = kmeans.predict(train_prediction_dataset)
    if train_prediction_res[0] == train_reality_res[i]:
        train_correct += 1
print('Accuracy in train dataset: ', train_correct * 1.0 / len(train_dataset))

# 测试集的正确率
print('Start evaluate model in test dataset')
test_reality_res = np.array(original_test_dataset['label'])
test_correct = 0
for i in range(len(test_dataset)):
    test_prediction_dataset = np.array(test_dataset[i].astype(float))
    test_prediction_dataset = test_prediction_dataset.reshape(-1, len(test_prediction_dataset))
    test_prediction_res = kmeans.predict(test_prediction_dataset)
    if test_prediction_res[0] == test_reality_res[i]:
        test_correct += 1
print('Accuracy in test dataset: ', test_correct * 1.0 / len(test_dataset))
