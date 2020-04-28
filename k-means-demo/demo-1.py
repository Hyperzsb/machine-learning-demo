from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import os


def show_cluster_res(model, data):
    labels = model.labels_
    labels = pd.DataFrame(labels, columns=['labels'])
    data = pd.DataFrame(data)
    data.insert(20, 'label', labels)

    a = TSNE().fit_transform(data)
    graph_data = pd.DataFrame(a, index=data.index)

    d1 = graph_data[data['label'] == 0]
    d2 = graph_data[data['label'] == 1]
    plt.plot(d1[0], d1[1], 'r.', d2[0], d2[1], 'go')
    plt.show()


if __name__ == '__main__':
    # Load original dataset
    original_dataset = pd.read_csv('dataset/original-tor-dataset.csv')
    # Remove useless features
    original_dataset.drop(['SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort',
                           'Protocol', 'FlowDuration', 'FlowBytes/s', 'FlowPackets/s'], 1, inplace=True)
    # Convert label column datatype from string to number
    original_dataset['label'] = original_dataset['label'].apply(
        lambda x: x.replace('nonTOR', '0').replace('TOR', '1')).astype(int)
    # Because of unsupervised learning, label is not used, remove it
    # And make every column datatype to float
    train_dataset_feature = original_dataset.drop(['label'], 1).astype(float)
    # Normalize every column to normal distribution
    # This function return value type is <class 'numpy.ndarray'>
    train_dataset_feature = preprocessing.scale(train_dataset_feature)
    # Extract the label column
    train_dataset_label = original_dataset['label']
    # Divide the dataset to train dataset and test dataset
    train_feature, test_feature, train_label, test_label = train_test_split(train_dataset_feature, train_dataset_label,
                                                                            test_size=0.3, random_state=0)

    # Create and train a K-Means cluster with k = 2 or load existed cluster
    model_saved_path = 'saved-model/k-means-cluster-demo-1.m'
    if os.path.exists(model_saved_path):
        print('Loading existed model')
        kmeans = joblib.load(model_saved_path)
        print('Loading complete')
    else:
        print('Training new model')
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(train_feature)
        joblib.dump(kmeans, model_saved_path)
        print('Training complete')

    # Predict and test accuracy
    test_prediction = kmeans.predict(test_feature)
    print('mutual_info_score: ', mutual_info_score(test_label, test_prediction))
    print('adjusted_mutual_info_score: ', adjusted_mutual_info_score(test_label, test_prediction))
    print('normalized_mutual_info_score: ', normalized_mutual_info_score(test_label, test_prediction))
