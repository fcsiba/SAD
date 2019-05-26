import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn. ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
from feature_selector import FeatureSelector
from sklearn.svm import SVC
from sklearn.externals import joblib 
#import tensorflow as tf
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  
from sklearn.feature_selection import VarianceThreshold
from ipykernel import kernelapp as app
# for clearing warnings
import warnings
warnings.filterwarnings("ignore")

#Load train data
train = pd.read_csv('/home/emaan/Documents/FYP2/myTrain.csv',nrows= 3000 , low_memory=False)

#Filter attributes
train = train.drop(['Flow_ID', 'Src_IP' , 'Dst_IP', 'Timestamp'], axis = 1) 
train = train.dropna()

#Split into attributes and class label
train_x = train.loc[:, train.columns != 'Label']
train_y = train.iloc[:,-1]

#Remove them constants features from train n test
constant_filter = VarianceThreshold(threshold=0.3)  
constant_filter.fit(train_x) 
len(train_x.columns[constant_filter.get_support()]) 

#constant_columns = [column for column in train_x.columns  
#                    if column not in train_x.columns[constant_filter.get_support()]]

#print(len(constant_columns))  

#Print them constant features
#for column in constant_columns:  
#    print(column)

#Remove correlated features
correlated_features = set()  
correlation_matrix = train_x.corr()  

for i in range(len(correlation_matrix .columns)):  
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)


len(correlated_features)  
print(correlated_features)  


train_x.drop(labels=correlated_features, axis=1, inplace=True)  
print('Correlated Features Dropped')  

lr = LogisticRegression()
print('LR DONE')
dt = DecisionTreeClassifier()
print('dt DONE')
svm = SVC(kernel = 'poly', degree = 4 )
print('SVM DONE')

evc = VotingClassifier( estimators= [('lr',lr),('dt',dt),('svm',svm)], voting = 'hard')
print('ENsemble Formed!')

evc.fit(train_x,train_y)
print('Model Trained!')

# Save the model as a pickle in a file 
joblib_file = "joblib_model.pkl"  
joblib.dump(evc, joblib_file)

print('modelfile saved!')

