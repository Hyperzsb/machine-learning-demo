from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import os


# 肘方法看k值
def find_best_k(data):
    inertia_list_ = []
    for i in range(1, 15):
        print('Train cluster when k = {}'.format(i))
        kmeans_ = KMeans(n_clusters=i)
        kmeans_.fit(data)
        inertia_list_.append(kmeans_.inertia_)
    # Show the trend graph
    plt.plot(range(1, 15), inertia_list_, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()


# 聚类可视化
def show_cluster_res(model, data, pre_label):
    centers = model.cluster_centers_  # 类别中心
    colors = ['r', 'c', 'b']
    columns_num = len(data.count())
    for i in range(columns_num - 2):
        plt.figure()
        for j in range(2):
            index_set = np.where(pre_label == j)
            cluster = data.iloc[index_set].iloc[0:5000, :]
            plt.scatter(cluster.iloc[:, i], cluster.iloc[:, i+1], c=colors[j], marker='.')
            plt.plot(centers[j][0], centers[j][1], 'o', markerfacecolor=colors[j], markeredgecolor='k',
                     markersize=8)  # 画类别中心
        plt.show()


if __name__ == '__main__':
    # Load original dataset
    original_dataset = pd.read_csv('dataset/original-tor-dataset.csv')
    # Remove useless features
    original_dataset.drop(['SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort',
                           'Protocol', 'FlowDuration', 'FlowBytes/s', 'FlowPackets/s'], 1, inplace=True)
    # Because of unsupervised learning, label is not used, remove it
    train_feature = original_dataset.iloc[:, :-1]
    # Extract the label column
    train_label = original_dataset.iloc[:, -1]
    # Normalize every column to normal distribution and cast into DataFrame with columns' name
    train_feature = preprocessing.scale(train_feature)
    train_feature = pd.DataFrame(train_feature, columns=original_dataset.iloc[:, :-1].columns)

    # Create and train a K-Means cluster with k = 2 or load existed cluster
    model_saved_path = 'saved-model/k-means-cluster-demo-2.m'
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
    train_prediction = kmeans.predict(train_feature)
    # Show metrics
    # 样本距离最近的聚类中心的距离总和
    inertia = kmeans.inertia_
    print('Inertia: {}'.format(inertia))
    # 调整后的兰德指数
    adjusted_rand_s = metrics.adjusted_rand_score(train_label, train_prediction)
    print('ARI:     {}'.format(adjusted_rand_s))
    # 互信息
    mutual_info_s = metrics.mutual_info_score(train_label, train_prediction)
    print('MI:      {}'.format(mutual_info_s))
    # 调整后的互信息
    adjusted_mutual_info_s = metrics.adjusted_mutual_info_score(train_label, train_prediction)
    print('AMI:     {}'.format(adjusted_mutual_info_s))
    # 同质化得分
    homogeneity_s = metrics.homogeneity_score(train_label, train_prediction)
    print('Homo:    {}'.format(homogeneity_s))
    # 完整性得分
    completeness_s = metrics.completeness_score(train_label, train_prediction)
    print('Comp:    {}'.format(completeness_s))
    # V-measure得分
    v_measure_s = metrics.v_measure_score(train_label, train_prediction)
    print('V-M:     {}'.format(v_measure_s))
    # 轮廓系数
    # silhouette_s = metrics.silhouette_score(train_feature, train_prediction, metric='euclidean')
    # print('Sil:     {}'.format(silhouette_s))

    # Show the cluster graph
    show_cluster_res(model=kmeans, data=train_feature, pre_label=train_prediction)
