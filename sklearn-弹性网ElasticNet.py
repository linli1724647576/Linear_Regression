#!/usr/bin/env python
# coding: utf-8

# ![](https://raw.githubusercontent.com/Qinbf/tf-model-zoo/master/README_IMG/01.jpg)
# AI MOOC： **www.ai-xlab.com**  
# 如果你也是AI爱好者，可以添加我的微信一起交流：**sdxxqbf**

# In[ ]:


import numpy as np
from numpy import genfromtxt
from sklearn import linear_model


# In[2]:


# 读入数据 
data = genfromtxt(r"longley.csv",delimiter=',')
print(data)


# In[3]:


# 切分数据
x_data = data[1:,2:]
y_data = data[1:,1]
print(x_data)
print(y_data)


# In[4]:


# 创建模型
model = linear_model.ElasticNetCV()
model.fit(x_data, y_data)

# 弹性网系数
print(model.alpha_)
# 相关系数
print(model.coef_)


# In[5]:


model.predict(x_data[-2,np.newaxis])


# In[ ]:




