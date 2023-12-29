#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the dependencies
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[4]:


# Data collection and Processing 
titanic_data = pd.read_csv("C:\DATA SCIENCE\PYTHON\Train.csv")
titanic_data.head()


# In[5]:


#Number of rows and columns
titanic_data.shape


# In[7]:


#Information about data
titanic_data.info()


# In[8]:


#Number of missing values in each column
titanic_data.isnull().sum()


# In[11]:


#Drop the cabin column from dataframe
titanic_data = titanic_data.drop(columns='Cabin',axis=1)


# In[12]:


#Replacing the missing values in each column with mean value
titanic_data['Age'].fillna(titanic_data['Age'].mean(),inplace=True)


# In[13]:


#Finding the mode value of "Embarked" column
print(titanic_data["Embarked"].mode())


# In[14]:


print(titanic_data["Embarked"].mode()[0])


# In[17]:


#Replacing the missing values in "Embarked" column with mode values 
titanic_data["Embarked"].fillna(titanic_data["Embarked"].mode()[0],inplace=True)


# In[18]:


#Check the number of missing values in each column
titanic_data.isnull().sum()


# In[19]:


#Statistical measures about the data
titanic_data.describe()


# In[21]:


#Number of people survived and not survived
titanic_data['Survived'].value_counts()


# In[23]:


sns.set()


# In[28]:


#Count plot for "Survived" column
sns.countplot(x='Survived',data=titanic_data)


# In[30]:


#Male and  female count 
titanic_data['Sex'].value_counts()


# In[29]:


#Count plot for "Sex" column
sns.countplot(x='Sex',data=titanic_data)


# In[35]:


#Number of survivor Genderwise
sns.countplot(x='Sex',hue='Survived',data=titanic_data)


# In[36]:


#Count plot for "Pclass" column
sns.countplot(x='Pclass',data=titanic_data)


# In[37]:


#Number of survivor Classwise
sns.countplot(x='Pclass',hue='Survived',data=titanic_data)


# In[40]:


#Encoding the categorical columns
titanic_data['Sex'].value_counts()


# In[41]:


titanic_data['Embarked'].value_counts()


# In[66]:


#converting categorical column
titanic_data.replace({'Sex':{'male':0,'female':1},'Embarked':{'S':0,'C':1,'Q':3}},inplace=True)
titanic_data.head()


# In[67]:


#Separating target and feature
X = titanic_data.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
Y = titanic_data['Survived']


# In[69]:


print(X)


# In[68]:


print(Y)


# In[70]:


#Splitting the data into Training data & Test data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2,random_state=2)


# In[71]:


print(X.shape,X_train.shape,X_test.shape)


# In[72]:


#Model Training using logistic regression
model = LogisticRegression()


# In[74]:


#training the Logistic Regression model with training data
model.fit(X_train,Y_train)


# In[76]:


#Model Evaluation


# In[84]:


#Accuracy score of training data
X_train_prediction = model.predict(X_train)
print(X_train_prediction)


# In[81]:


training_data_accuracy= accuracy_score(Y_train,X_train_prediction)
print('Accruracy score of training data:', training_data_accuracy)


# In[83]:


#Accuracy score on test data
X_test_prediction = model.predict(X_test)
print(X_test_prediction)


# In[85]:


Test_data_accuracy= accuracy_score(Y_test,X_test_prediction)
print('Accruracy score of Test data:', Test_data_accuracy)


# In[86]:


print('Accruracy score of training data:', training_data_accuracy)
print('Accruracy score of Test data:', Test_data_accuracy)

