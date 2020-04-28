from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
import os

# Load original dataset
original_dataset = pd.read_csv('dataset/original-tor-dataset.csv')
# Select feature columns from original dataset
train_dataset_feature = original_dataset[['FlowIATMean', 'FlowIATStd', 'FlowIATMax', 'FlowIATMin',
                                          'FwdIATMean', 'FwdIATStd', 'FwdIATMax', 'FwdIATMin',
                                          'BwdIATMean', 'BwdIATStd', 'BwdIATMax', 'BwdIATMin',
                                          'ActiveMean', 'ActiveStd', 'ActiveMax', 'ActiveMin',
                                          'IdleMean', 'IdleStd', 'IdleMax', 'IdleMin']].values.tolist()
# Extract the label column
train_dataset_label = original_dataset[['label']].values.ravel()
# Divide the dataset to train dataset and test dataset
train_feature, test_feature, train_label, test_label = train_test_split(train_dataset_feature, train_dataset_label,
                                                                        test_size=0.3, random_state=0)

# Create and train a K Nearest Neighbors classifier or load existed classifier
model_saved_path = 'saved-model/k-neighbors-classifier.m'
if os.path.exists(model_saved_path):
    print('Loading existed model')
    knn = joblib.load(model_saved_path)
    print('Loading complete')
else:
    print('Training new model')
    knn = KNeighborsClassifier()
    knn.fit(train_feature, train_label)
    joblib.dump(knn, model_saved_path)
    print('Training complete')

# Predict and test accuracy
prediction_res = knn.predict(test_feature)
train_dataset_accuracy = accuracy_score(prediction_res, test_label)
print('Accuracy: ', train_dataset_accuracy)
