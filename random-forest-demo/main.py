from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import joblib
import os

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
train_dataset_feature = np.array(original_dataset.drop(['label'], 1)).astype(float)
# Normalize every column to normal distribution
train_dataset_feature = preprocessing.scale(train_dataset_feature)
# Extract the label column
train_dataset_label = np.array(original_dataset['label'])
# Divide the dataset to train dataset and test dataset
train_feature, test_feature, train_target, test_target = train_test_split(train_dataset_feature, train_dataset_label,
                                                                          test_size=0.3, random_state=0)

# Create and train a random forest classifier or load existed classifier
model_saved_path = 'saved-model/random-forest-classifier.m'
if os.path.exists(model_saved_path):
    print('Loading existed model')
    clf = joblib.load(model_saved_path)
    print('Loading complete')
else:
    print('Training new model')
    clf = RandomForestClassifier()
    clf.fit(train_feature, train_target)
    joblib.dump(clf, model_saved_path)
    print('Training complete')

# Predict and test accuracy
predict_results = clf.predict(test_feature)
print('Accuracy: ', accuracy_score(predict_results, test_target))
