#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as se


# In[2]:


sp= pd.read_csv(r"C:\Users\CS30007TX\Desktop\CODSOFT\advertising.csv")
sp.head()


# In[3]:


sp.shape


# In[4]:


sp.describe()


# ##### What we have observe is
# ####       Average spend is highest on tv
# ####       average expense is lowest on RAdio
# ####       Max sales is 27 and min is 1.6

# In[5]:


se.pairplot(sp,x_vars=["TV","Radio","Newspaper"],y_vars="Sales",kind="scatter")
plt.show()
        


# In[6]:


sp['TV'].plot.hist(bins=100) 


# In[7]:


sp['TV'].plot.hist(bins=10)


# In[8]:


sp['Radio'].plot.hist(bins=10)


# In[9]:


sp['Newspaper'].plot.hist(bins=10)


# In[10]:


se.heatmap(sp.corr(),annot=True)
plt.show()


# In[28]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(sp[['TV']],sp[['Sales']],test_size = 0.3,random_state=0)


# In[29]:


print(x_train)


# In[30]:


print(x-test)


# In[31]:


print(x_test)


# In[32]:


print(y_train)


# In[33]:


print(y_test)


# In[34]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)


# In[35]:


res=model.predict(x_test)
print(res)


# In[36]:


model.coef_


# In[37]:


model.intercept_


# In[38]:


0.05473199*69.2+7.14382225


# In[39]:


plt.plot(res)


# In[45]:


plt.scatter(x_test,y_test)
plt.plot(x_test,7.14382225+x_test*0.05473199,'r')
plt.show()


# In[ ]:




