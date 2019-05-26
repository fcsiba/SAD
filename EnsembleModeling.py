import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn. ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
from feature_selector import FeatureSelector
from sklearn.svm import SVC
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  
from sklearn.feature_selection import VarianceThreshold
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ipykernel import kernelapp as app

# for clearing warnings
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv('dir\path\myTrain.csv', nrows= 3020 , low_memory=False)
test = pd.read_csv('dir\path\myTest.csv',nrows = 510, low_memory=False)

#Drop these features from train and test:
train = train.drop(['Flow_ID', 'Src_IP' , 'Dst_IP', 'Timestamp'], axis = 1) 
test = test.drop(['Flow_ID', 'Src_IP' , 'Dst_IP', 'Timestamp'], axis = 1) 

train = train.dropna()
test = test.dropna()
print(train.shape)
print(test.shape)

#X has features in it, and Y has labels
train_x = train.loc[:, train.columns != 'Label']
train_y = train.iloc[:,-1]

print(train_x.shape)

test_x = test.loc[:, test.columns != 'Label']
#test_x = test.iloc[:,1:]
test_y = test.iloc[:,-1]

#Check for Null values
train_x.isnull().values.any()

#DATA PREPOCESSING
#Remove them constants features from train n test
constant_filter = VarianceThreshold(threshold=0.3)  
constant_filter.fit(train_x) 
len(train_x.columns[constant_filter.get_support()]) 
constant_columns = [column for column in train_x.columns  
                    if column not in train_x.columns[constant_filter.get_support()]]

print(len(constant_columns))  

#Print them constant features
for column in constant_columns:  
    print(column)
            
print(train_x.shape)

#TO REMOVE CORRELATED FEATURES
correlated_features = set()  
correlation_matrix = train_x.corr()  
correlation_matrix = test_x.corr()  

for i in range(len(correlation_matrix .columns)):  
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)

len(correlated_features)  
print(correlated_features)  

train_x.drop(labels=correlated_features, axis=1, inplace=True)  
test_x.drop(labels=correlated_features, axis=1, inplace=True)  

#train_x.shape
#print(train_x.head(2)) 
train_x.dtypes

#descision tree
dt = DecisionTreeClassifier()
dt.fit(train_x,train_y)
dt.score(test_x,test_y)
dt.score(train_x,train_y)

#Random Forest - Ensemble of Descision Trees
rf = RandomForestClassifier(n_estimators=20)
rf.fit(train_x,train_y)
rf.score(test_x,test_y)

#Bagging --> Overfitting less compared to ada boosting
bg = BaggingClassifier(DecisionTreeClassifier(), max_samples= 0.8, max_features = 0.5, n_estimators = 20)
bg.fit(train_x,train_y)
bg.score(test_x,test_y)
bg.score(train_x,train_y)

#Boosting - Ada Boost
adb = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators = 5, learning_rate = 1)
adb.fit(train_x,train_y)
adb.score(test_x,test_y)

#Clearly over-fitting
adb.score(train_x,train_y)

# Voting Classifier - Multiple Model Ensemble 

lr = LogisticRegression()
dt = DecisionTreeClassifier()
svm = SVC(kernel = 'poly', degree = 4 )

evc = VotingClassifier( estimators= [('lr',lr),('dt',dt),('svm',svm)], voting = 'hard')
evc.fit(train_x.iloc[1:3000],train_y.iloc[1:3000])
evc.score(test_x, test_y)
evc.score(train_x, train_y)

#print(test_y,)
from yellowbrick.classifier import ClassificationReport

# Instantiate the classification model and visualizer
visualizer = ClassificationReport(evc, classes=['xss','ssh','portscan','Benign'])

visualizer.fit(train_x, train_y)  # Fit the training data to the visualizer
visualizer.score(test_x, test_y)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data
