#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy import genfromtxt
from sklearn import linear_model


# In[2]:


#读入数据
data = genfromtxt(r"longley.csv",delimiter=',')
print(data)


# In[3]:


# 切分数据
x_data = data[1:,2:]
y_data = data[1:,1]
print(x_data)
print(y_data)


# In[4]:


#创建模型
model=linear_model.LassoCV()
model.fit(x_data,y_data)

#LASSO系数
print(model.alpha_)
#相关系数
print(model.coef_)


# In[7]:


print(model.predict(x_data[-2,np.newaxis]))


# In[ ]:




