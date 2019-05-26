
import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn. ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
from feature_selector import FeatureSelector
from sklearn.svm import SVC
from sklearn.externals import joblib 
from sklearn.model_selection import train_test_split  
from sklearn.feature_selection import VarianceThreshold
from ipykernel import kernelapp as app
# for clearing warnings
import warnings
warnings.filterwarnings("ignore")


# In[99]:
from subprocess import Popen, PIPE
import time
import os

p = os.popen('sudo tcpdump -c 50 -i wlp8s0 -w /home/emaan/Documents/FYP2/caps/cap.pcap -C 10')
time.sleep(10)
p.close()


p2 = os.popen('/home/emaan/cicflowmeter-4/CICFlowMeter-4.0/bin/./cfm /home/emaan/Documents/FYP2/caps /home/emaan/Documents/FYP2/caps')
time.sleep(10)
p2.close()

#Load the test data
test = pd.read_csv('/home/emaan/Documents/FYP2/caps/cap.pcap_Flow.csv', low_memory=False)


#Change attribute names from original file


#Drop these features from train and test:
test = test.drop(['Flow_ID', 'Src_IP' , 'Dst_IP', 'Timestamp'], axis = 1) 

#Drop null rows
test = test.dropna()
#Visualize the data
print(test.shape)

#Split dataframe into features and label
test_x = test.loc[:, test.columns != 'Label']
test_y = test.iloc[:,-1]

#Check for Null values
train_x.isnull().values.any()

#DATA PREPOCESSING
#Remove them constants features from train n test
constant_filter = VarianceThreshold(threshold=0.3)  
constant_filter.fit(test_x) 
len(test_x.columns[constant_filter.get_support()]) 

constant_columns = [column for column in test_x.columns  
                    if column not in test_x.columns[constant_filter.get_support()]]

print(len(constant_columns))  


#Print the constant features
for column in constant_columns:  
    print(column)

#Remove them constants features from train n test

test2_x = constant_filter.transform(test_x)
test2_x = test2_x.drop([constant_columns]) 

for column in train_x.columns: 
    if [constant_columns] in test_x[col] : 
            del test_x[col] 
            
print(test2_x.shape)

#TO REMOVE CORRELATED FEATURES
correlated_features = set()  
correlation_matrix = test_x.corr()  

for i in range(len(correlation_matrix .columns)):  
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)


len(correlated_features)  
print(correlated_features)  

test_x.drop(labels=correlated_features, axis=1, inplace=True)  

#train_x.shape
#print(train_x.head(2)) 
test_x.dtypes

  
# Load the ensemble model from the file 
ensemble_from_joblib = joblib.load('modelFile.pkl')  
  
# Use the loaded model to make predictions 
ensemble_from_joblib.predict(test_x) 
