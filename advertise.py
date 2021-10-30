
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ad_data =pd.read_csv("C:/Users/Bibhuti/OneDrive/Desktop/360digiTMG assignment/LR/advertising.csv")

ad_data.head()
ad_data.describe()

#Data visualization
sns.set_style('whitegrid')
ad_data['Age'].hist(bins=30)
plt.xlabel('Age')

#Checking for null values
ad_data.isnull().sum()

# Extract datetime variables using timestamp column
ad_data['Timestamp'] = pd.to_datetime(ad_data['Timestamp']) 

# Converting timestamp column into datatime object in order to extract new features
ad_data['Month'] = ad_data['Timestamp'].dt.month 

# Creates a new column called Month
ad_data['Day'] = ad_data['Timestamp'].dt.day  

# Creates a new column called Day
ad_data['Hour'] = ad_data['Timestamp'].dt.hour   

# Creates a new column called Hour
ad_data["Weekday"] = ad_data['Timestamp'].dt.dayofweek 

# Dropping timestamp column to avoid redundancy
ad_data = ad_data.drop(['Timestamp'], axis=1) # deleting timestamp

ad_data.head()

#visualize target variable
sns.countplot(x = 'Clicked_on_Ad', data = ad_data)

from sklearn.model_selection import train_test_split

X = ad_data[['Daily_Time_ Spent _on_Site', 'Age', 'Area_Income','Daily Internet Usage', 'Male']]
y = ad_data['Clicked_on_Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression(solver='lbfgs')
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))

# Importing a pure confusion matrix from sklearn.metrics family
from sklearn.metrics import confusion_matrix

# Printing the confusion_matrix
print(confusion_matrix(y_test, predictions))

# Importing a pure confusion matrix from sklearn.metrics family
from sklearn.metrics import confusion_matrix

# Printing the confusion_matrix
print(confusion_matrix(y_test, predictions))


accuracy_test = (131 + 155)/(402) 
accuracy_test

