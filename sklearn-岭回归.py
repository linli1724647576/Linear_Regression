#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from numpy import genfromtxt
from sklearn import linear_model
import matplotlib.pyplot as plt


# In[3]:


# 读入数据 
data = genfromtxt(r"longley.csv",delimiter=',')
print(data)


# In[4]:


# 切分数据
x_data = data[1:,2:]
y_data = data[1:,1]
print(x_data)
print(y_data)


# In[5]:


# 创建模型
# 默认生成50个值,为岭回归系数
alphas_to_test = np.linspace(0.001, 1 ,100)
# 创建模型，保存误差值  Ridge岭回归 CV表示交叉验证
model = linear_model.RidgeCV(alphas=alphas_to_test, store_cv_values=True)
model.fit(x_data, y_data)

# 岭系数
print(model.alpha_)
# loss值  (误差)
print(model.cv_values_.shape)


# In[9]:


# 画图
# 岭系数跟loss值的关系,对16个样本求平均值
plt.plot(alphas_to_test, model.cv_values_.mean(axis=0))
# 选取的岭系数值的位置
plt.plot(model.alpha_, min(model.cv_values_.mean(axis=0)),'ro')
plt.show()


# In[7]:


print(model.predict(x_data[2,np.newaxis]))

