#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[22]:


data=np.genfromtxt("linear.csv",delimiter=",")
x_data=data[1:,0]
y_data=data[1:,1]
X_data=x_data[:,np.newaxis]
Y_data=y_data[:,np.newaxis]


# In[23]:


model=LinearRegression()
model.fit(X_data,Y_data)


# In[28]:


plt.scatter(x_data,y_data)
plt.plot(X_data,model.predict(X_data),'r')
plt.xlabel("time")
plt.ylabel("quality")
plt.show()


# In[ ]:




