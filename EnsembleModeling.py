#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[99]:


train = pd.read_csv('C:\\Users\sunil\OneDrive\Documents\FYP\Train\myTrain.csv', nrows= 3020 , low_memory=False)
test = pd.read_csv('C:\\Users\sunil\OneDrive\Documents\FYP\Test\myTest.csv',nrows = 510, low_memory=False)


# In[100]:


#Drop these features from train and test:
train = train.drop(['Flow_ID', 'Src_IP' , 'Dst_IP', 'Timestamp'], axis = 1) 
test = test.drop(['Flow_ID', 'Src_IP' , 'Dst_IP', 'Timestamp'], axis = 1) 


#Check em datatypes
#train.dtypes


# In[101]:


#train.head(50)
#train_y.head()
#DROP THEM NULL ROWS
#train.replace(["NaN", 'NaT'], np.nan, inplace = True)

train = train.dropna()
test = test.dropna()


# In[91]:


print(train.shape)
print(test.shape)


# In[102]:


#X has features in it, and Y has labels
#train_y = train.loc[:, train.columns == 'Label']
#train_x = train.iloc[:,-1]
#train_y = train.iloc[:,1]
#train_x = train.iloc[:,1:]
train_x = train.loc[:, train.columns != 'Label']
train_y = train.iloc[:,-1]

print(train_x.shape)
#print(train_y.shape)
#train_y.head()
#train_x.head


# In[103]:


test_x = test.loc[:, test.columns != 'Label']
#test_y = test.loc[:, test.columns == 'Label']
#test_x = test.iloc[:,1:]
test_y = test.iloc[:,-1]


#test_x = test.iloc[:,1:]
#test_y = test.iloc[:,0]


# In[104]:


#Check for Null values
train_x.isnull().values.any()
#train_x.isnull().sum().sum()

#train_x.isna().any()
#train_x.loc[:, train_x.isna().any()]

#test_x.loc[:, test_x.isna().any()]


# In[9]:


#DATA PREPOCESSING
#Remove them constants features from train n test
constant_filter = VarianceThreshold(threshold=0.3)  
constant_filter.fit(train_x) 
len(train_x.columns[constant_filter.get_support()]) 


# In[10]:


constant_columns = [column for column in train_x.columns  
                    if column not in train_x.columns[constant_filter.get_support()]]

print(len(constant_columns))  


# In[11]:


#Print them constant features
for column in constant_columns:  
    print(column)


# In[94]:


#Remove them constants features from train n test
#train_x = constant_filter.transform(train_x)  
#test_x = constant_filter.transform(test_x)

#train_x.shape, test_x.shape  
#train_x.head
#train_x = train_x.drop([constant_columns]) 
#test_x = test_x.drop([constant_columns]) 

#for column in train_x.columns: 
#    if [constant_columns] in train_x[col] : 
#            del train_x[col] 
            
print(train_x.shape)


# In[105]:


#TO REMOVE CORRELATED FEATURES
correlated_features = set()  
correlation_matrix = train_x.corr()  
correlation_matrix = test_x.corr()  


# In[106]:


for i in range(len(correlation_matrix .columns)):  
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)


# In[107]:


len(correlated_features)  
print(correlated_features)  


# In[108]:


train_x.drop(labels=correlated_features, axis=1, inplace=True)  
test_x.drop(labels=correlated_features, axis=1, inplace=True)  

#train_x.shape
#print(train_x.head(2)) 
train_x.dtypes


# In[109]:


#descision tree
dt = DecisionTreeClassifier()
dt.fit(train_x,train_y)


# In[51]:


dt.score(test_x,test_y)


# In[52]:


dt.score(train_x,train_y)


# In[53]:


#Random Forest - Ensemble of Descision Trees

rf = RandomForestClassifier(n_estimators=20)
rf.fit(train_x,train_y)


# In[54]:


rf.score(test_x,test_y)


# In[55]:


#Bagging --> Overfitting less compared to ada boosting

bg = BaggingClassifier(DecisionTreeClassifier(), max_samples= 0.8, max_features = 0.5, n_estimators = 20)
bg.fit(train_x,train_y)


# In[56]:


bg.score(test_x,test_y)


# In[57]:


bg.score(train_x,train_y)


# In[58]:


#Boosting - Ada Boost

adb = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators = 5, learning_rate = 1)
adb.fit(train_x,train_y)


# In[59]:


adb.score(test_x,test_y)


# In[60]:


#Clearly over-fitting
adb.score(train_x,train_y)


# In[61]:


# Voting Classifier - Multiple Model Ensemble 

lr = LogisticRegression()
dt = DecisionTreeClassifier()
svm = SVC(kernel = 'poly', degree = 4 )


# In[62]:


evc = VotingClassifier( estimators= [('lr',lr),('dt',dt),('svm',svm)], voting = 'hard')


# In[63]:


evc.fit(train_x.iloc[1:3000],train_y.iloc[1:3000])


# In[64]:


evc.score(test_x, test_y)


# In[65]:


evc.score(train_x, train_y)


# In[36]:


#print(test_y,)
from yellowbrick.classifier import ClassificationReport

# Instantiate the classification model and visualizer
visualizer = ClassificationReport(evc, classes=['xss','ssh','portscan','Benign'])

visualizer.fit(train_x, train_y)  # Fit the training data to the visualizer
visualizer.score(test_x, test_y)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data

