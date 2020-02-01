#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt


# In[14]:


##载入数据
data=np.genfromtxt("data.csv",delimiter=",")
x_data=data[:,0,np.newaxis]
y_data=data[:,1,np.newaxis]
plt.scatter(x_data,y_data)
plt.show()


# In[15]:


print(np.mat(x_data).shape)
print(np.mat(y_data).shape)
#给样本添加偏置项
X_data=np.concatenate((np.ones((100,1)),x_data),axis=1)   #合并
print(X_data.shape)


# In[16]:


print(X_data[:3])


# In[25]:


#标准方程法
def weights(xArr,yArr):
    #将np.array变成matrix
    xMat=np.mat(xArr)
    yMat=np.mat(yArr)
    xTx=xMat.T * xMat  #矩阵乘法
    #计算行列式
    if np.linalg.det(xTx) == 0.0:
        print("This matrix cannot do inverse")
        return
    ws=xTx.I * xMat.T *yMat  #矩阵的逆.I
    return ws


# In[27]:


ws=weights(X_data,y_data)
print(ws)


# In[30]:


#画图
x_test=np.array([[20],[80]])
print(x_test)
y_test=ws[0]+ x_test * ws[1]
plt.plot(x_data,y_data,'b.')  #b.代表蓝色的点
plt.plot(x_test,y_test,'r')
plt.show()


# In[ ]:




