#!/usr/bin/env python
# coding: utf-8

# In[36]:


import sys
print(sys.version)
import scipy
print(scipy.__version__)
import numpy
print(numpy.__version__)
import pandas
print(pandas.__version__)
import matplotlib
print(matplotlib.__version__)
import sklearn
print(sklearn.__version__ )


# In[82]:


import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import seaborn as sns


# In[38]:


df=pd.read_csv("C:\\Users\\asus\\Desktop\\iris.csv")


# In[39]:


names=['sepal_length','sepal_width','petal_length','petal_width','species']


# In[40]:


print(df.shape)


# In[41]:


print(df.head())


# In[42]:


print(df.head(90))


# In[43]:


print(df.describe())


# In[44]:


print(df.info())


# In[45]:


df.isnull().sum().sum()


# In[46]:


print(df.dropna())


# In[47]:


df.drop(["Unnamed: 5"],axis=1)


# In[53]:


print(df.groupby('species').size())


# In[54]:


df.describe()


# In[85]:


df1=df.drop(["Unnamed: 5"],axis=1)


# In[56]:


print(df1.head())


# In[57]:


print(df1.groupby('species').size())


# In[58]:


df1.describe()


# In[68]:


df1.plot(kind="box",subplots=True,layout=(2,2),sharex=False)
plt.show()
#The boxes determine the minimum, median and maximum in the graph and the dots denote the density of the data


# In[69]:


df1.hist()
plt.show()
#depicts how often each distinct value in the set of data occurs


# In[70]:


scatter_matrix(df1)
plt.show()
#visualise the bivariate relationship between combination of variables


# In[74]:


array=df1.values
#splitting of data
X=array[:,0:4]
Y=array[:,4]
t_size=0.20
seed=6
#model selection is used to set blueprint for the splitting of the data and used to measure the new data
#When we use an integer for random_state, the function will produce the same results across different executions. 
#The results are only changed if we change the integer value.
X_train,X_test,Y_train,Y_test= model_selection.train_test_split(X,Y,test_size=t_size,random_state=seed)


# In[75]:


seed=6
scoring="accuracy"
#accuracy is the ratio of correctly predicted no. of instances to total no. of instances in the data set  into 100


# In[80]:


#Building the model
#LR,LDA,KNN,CART,NB,SVM
models=[]
models.append(('LR',LogisticRegression()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC()))
results=[]
names=[]
for name,model in models:
    kfold=model_selection.KFold(n_splits=10,random_state=seed,shuffle=True )
    cv_results=model_selection.cross_val_score(model,X_train,Y_train,cv=kfold,scoring=scoring)
    results.append(name)
    msg="%s: %f (%f)"%(name,cv_results.mean(),cv_results.std())
    print(msg)


# In[92]:


df1.corr()
plt.figure(figsize=(15,8))
sns.heatmap(df1.corr(),annot=True)
plt.show()


# In[ ]:




