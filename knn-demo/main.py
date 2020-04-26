from sklearn import datasets  # 引入数据集,sklearn包含众多数据集
from sklearn.model_selection import train_test_split  # 将数据分为测试集和训练集
from sklearn.neighbors import KNeighborsClassifier  # 利用邻近点方式训练数据
import pandas
import numpy as np

np.set_printoptions(threshold=np.inf)

# 引入数据
original_data = pandas.read_csv('SelectedFeatures-10s-TOR-NonTOR.csv')
dataset_data = original_data[[' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min',
                              'Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max', ' Fwd IAT Min',
                              'Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min',
                              'Active Mean', ' Active Std', ' Active Max', ' Active Min',
                              'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min']].values.tolist()
dataset_target = original_data[['label']].values.ravel()

# 利用train_test_split进行将训练集和测试集进行分开，test_size占30%
X_train, X_test, Y_train, Y_test = train_test_split(dataset_data, dataset_target, test_size=0.2)

# 引入训练方法
knn = KNeighborsClassifier()
# 进行填充测试数据进行训练
knn.fit(X_train, Y_train)

# # 预测特征值
# predict_file = open('predict.txt', 'w+')
# print(knn.predict(X_test), end=' ', file=predict_file)
# predict_file.close()
# # 真实特征值
# reality_file = open('reality.txt', 'w+')
# print(Y_test, end=' ', file=reality_file)
# reality_file.close()

predict_res = knn.predict(X_test)
reality_res = Y_test
count = 0
length = len(reality_res)
for i in range(length):
    if predict_res[i] != reality_res[i]:
        count += 1
print(count / length)
