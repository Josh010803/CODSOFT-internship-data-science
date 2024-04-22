#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as se


# In[2]:


sp=pd.read_csv(r"C:\Users\CS30007TX\Desktop\CODSOFT\IRIS.csv")
sp.head()


# In[3]:


sp['species'],categories=pd.factorize(sp['species'])
sp.head()


# In[4]:


sp.describe


# In[5]:


sp.isna().sum()


# In[6]:


from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(sp.petal_length,sp.petal_width,sp.species)
ax.set_xlabel('Petallengthcm')
ax.set_ylabel('Petalwidthcm')
ax.set_zlabel('Species')
plt.title('3D scatter plot example')
plt.show()


# In[8]:


se.scatterplot(data=sp,x='sepal_length',y='sepal_width',hue='species')


# In[9]:


se.scatterplot(data=sp,x='petal_width',y='petal_length',hue='species')


# ## Applying Elbow technique

# In[10]:


k_rng=range(1,10)
sse=[]

for k in k_rng:
    km= KMeans(n_clusters=k)
    km.fit(df[['petal_length','petal_width']])
    sse.append(km.inertia_)


# In[11]:


k_rng=range(1,10)
sse=[]

for k in k_rng:
    km= KMeans(n_clusters=k)
    km.fit(sp[['petal_length','petal_width']])
    sse.append(km.inertia_)


# In[12]:


sse


# In[13]:


plt.xlabel('k_rng')
plt.ylabel('Sum of squared errors')
plt.plot(k_rng,sse)


# ### Kmean Algorithm:

# In[14]:


km=KMeans(n_clusters=3,random_state=0)
y_predicted = km.fit_predict(sp[['petal_length','petal_width']])
y_predicted


# In[15]:


sp['cluster']=y_predicted
sp.head(150)


# In[16]:


rom sklearn.metrics import confusion_matrix
cm=confusion_matrix(sp.species,sp.cluster)
cm


# In[17]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(df.species,df.cluster)
cm


# In[18]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(sp.species,sp.cluster)
cm


# In[23]:


true_labels=sp.species
predicted_labels=sp.cluster

cm=confusion_matrix(true_labels,predicted_labels)
class_labels=['Setosa','Versicolor','Virginia']
plt.imshow(cm,interpolation='nearest',cmap=plt.cm.Reds)
plt.title('Confusion matrix')
plt.colorbar()
tick_marks=np.arange(len(class_labels))
plt.xticks(tick_marks,class_labels)
plt.yticks(tick_marks,class_labels)


for i in range(len(class_labels)):
    for j in range(len(class_labels)):
        plt.text(j,i,str(cm[i][j]),ha='center',va='center',color='white')

plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()


# In[ ]:




