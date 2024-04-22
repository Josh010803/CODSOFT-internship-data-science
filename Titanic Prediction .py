#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as se


# In[2]:


sp=pd.read_csv(r"C:\Users\CS30007TX\Desktop\CODSOFT\Titanic-Dataset.csv")
sp.head(10)


# In[3]:


sp.shape()


# In[4]:


sp.shape


# In[5]:


sp.describe()


# In[6]:


sp['Survived'].valuecounts()


# In[7]:


sp['Survived'].value_counts()


# In[8]:


se.countplot(x=sp['Survived'],hue=sp['Pclass'])


# In[9]:


df['Sex']


# In[10]:


sp['Sex']


# In[11]:


se.countplot(x=sp['Sex'],hue=sp['Survived'])


# In[12]:


df.groupby('Sex')[['Survived']].mean()


# In[13]:


sp.groupby('Sex')[['Survived']].mean()


# In[14]:


df['Sex'].unique()


# In[15]:


sp['Sex'].unique()


# In[17]:


from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()

sp['Sex']=labelencoder.fit_transform(sp['Sex'])

sp.head()


# In[18]:


sp.isna().sum()


# In[19]:


sp=sp.drop([Age],axis=1)


# In[20]:


sp=sp.drop(['Age'],axis=1)


# In[21]:


sp_final=sp
sp_final.head(10)


# ### Model Training

# In[22]:


x=sp[['Pclass','Sex']]
y=sp['Survived']


# In[23]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test size=0.4,random state=0)


# In[24]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random state=0)


# In[25]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=0)


# In[26]:


from sklearn.linear_model import LogisticRegression 
log=LogisticRegression(random_state=0)
log.fit(x_train,y_train)


# MODEL PREDICTION
# 

# In[27]:


pred=print(log.predict(x_test))


# In[28]:


print(y_test)


# In[ ]:




