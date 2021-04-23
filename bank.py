
import pandas as pd


#loading the transet
tran = pd.read_csv("C:/Users/Bibhuti/OneDrive/Desktop/360digiTMG assignment/LR/bank_data.csv")

tran_details =pd.DataFrame({"column name":tran.columns,
                            "tran type(in Python)": tran.dtypes})
#3.	tran Pre-trancessing
#3.1 tran Cleaning, Feature Engineering, etc
         
#details of tran 
tran.info()
tran.describe()          

#tran types        
tran.dtypes

#checking for na value
tran.isna().sum()
tran.isnull().sum()

#checking unique value for each columns
tran.nunique()

EDA ={"column ": tran.columns,
      "mean": tran.mean(),
      "median":tran.median(),
      "mode":tran.mode(),
      "standard deviation": tran.std(),
      "variance":tran.var(),
      "skewness":tran.skew(),
      "kurtosis":tran.kurt()}

EDA

# covariance for tran set 
covariance = tran.cov()
covariance

# Correlation matrix 
co = tran.corr()
co

import seaborn as sns
####### graphitran repersentation

#boxplot for every continuous type data
tran.columns
tran.nunique()

tran.boxplot(column=['age', 'balance', 'duration', 'campaign','pdays','default'])   #no outlier
 
sns.pairplot(tran.iloc[:, [0,2,5,6,7,8]])

# Boxplot of independent variable distribution for each category of default 
sns.boxplot(x = "default", y = "housing", data = tran)
sns.boxplot(x = "default", y = "loan", data = tran)
sns.boxplot(x = "default", y = "poutfailure", data = tran)
sns.boxplot(x = "default", y = "divorced", data = tran)
sns.boxplot(x = "default", y = "married", data = tran)

# Scatter plot for each categorical default of car
sns.stripplot(x = "default", y = "housing", jitter = True, data = tran)
sns.stripplot(x = "default", y = "loan", jitter = True, data = tran)
sns.stripplot(x = "default", y = "poutfailure", jitter = True, data = tran)
sns.stripplot(x = "default", y = "divorced", jitter = True, data = tran)
sns.stripplot(x = "default", y = "married", jitter = True, data = tran)

# Scatter plot between each possible pair of independent variable and also histogram for each independent variable 

sns.pairplot(tran, hue = "default") # With showing the category of each car default in the scatter plot

# Rearrange the order of the variables

tran = tran.iloc[:, [31, 0,1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]]

trans = tran.rename({"joadmin.":"joadmin","joblue.collar":"joblue_collar","joself.employed":"joself_employed"},axis =1)

"""
5.	Model Building
5.1	Build the model on the scaled data (try multiple options)
5.2	Perform Logistic Regression model.
5.3	Train and Test the data and compare accuracies by Confusion Matrix, plot ROC AUC curve. 
5.4	Briefly explain the model output in the documentation. 
		
 """
from sklearn.model_selection import train_test_split # train and test 

import statsmodels.formula.api as sm
from sklearn import metrics
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt

### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(trans, test_size = 0.2,random_state =7) # 20% test data

# Model building 
# import statsmodels.formula.api as sm
logit_model = sm.logit('y ~ age + default + balance + housing+loan + duration+  campaign + pdays+ previous+  poutfailure+ poutother+ poutsuccess+ poutunknown+ con_cellular+ con_telephone+ con_unknown+  divorced+ married+ single+ joadmin+ joblue_collar+ joentrepreneur+ johousemaid+ jomanagement+ joretired+ joself_employed+ joservices+ jostudent+ jotechnician+ jounemployed+ jounknown', data = train_data).fit()

#summary
logit_model.summary2() # for AIC
logit_model.summary()

#prediction
train_pred = logit_model.predict(train_data.iloc[ :, 1: ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(36168)

# taking threshold value as 0.5 and above the prob value will be treated as correct value 
train_data.loc[train_pred > 0.5, "train_pred"] = 1

# classification report
classification = classification_report(train_data["train_pred"], train_data["y"])
classification

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['y'])
confusion_matrx

accuracy_train = (31153 + 1352)/(36168)
print(accuracy_train)

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(train_data["y"], train_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc = metrics.auc(fpr, tpr)  #AUC = 0.90

# Prediction on Test data set
test_pred = logit_model.predict(test_data)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(9043)

# taking threshold value as 0.5 and above the prob value will be treated as correct value 
test_data.loc[test_pred > 0.5, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['y'])

confusion_matrix
accuracy_test = (7839 + 356)/(9043) 
accuracy_test
