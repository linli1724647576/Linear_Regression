#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


#载入数据
data=np.genfromtxt("data.csv",delimiter=",")
x_data=data[:,0]
y_data=data[:,1]
plt.scatter(x_data,y_data)
plt.show()
print(x_data.shape)


# In[3]:


x_data=data[:,0,np.newaxis]  #加上一个维度
print(x_data.shape)


# In[4]:


x_data=data[:,0,np.newaxis]
y_data=data[:,1,np.newaxis]
#创建并拟合模型
model=LinearRegression()
model.fit(x_data,y_data)


# In[8]:


#画图
plt.plot(x_data, y_data, 'b.')
plt.plot(x_data, model.predict(x_data) , 'r')
plt.show()


# In[ ]:




