#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_or = pd.read_csv('GSPC.csv')


# In[2]:


data_or.head()


# In[3]:


# Odayago = data_or['Open'][:-1]
# today = data_or['Open'][1:]

# plt.plot(Odayago, today,'.')
# plt.xlabel('One day ago Open value')
# plt.ylabel('Today Open value')
# plt.show()


# In[4]:


Odayago = data_or['Open'][1:-1]
Tdayago = data_or['Open'][:-2]
today = data_or['Open'][2:]

ax = plt.axes(projection='3d')

ax.scatter(Odayago, Tdayago, today)
ax.set_xlabel('One day ago value')
ax.set_ylabel('Two days ago value')
ax.set_zlabel('Today value')
plt.show()


# In[5]:


X = np.ones((len(today),1))
X = np.c_[X, Odayago, Tdayago]
y = today


# ### The formula is:
# 
# $$y = X\theta$$
# $$X = \left[ \begin{matrix}X_{11} & X_{12} & X_{13}\\
# X_{21} & X_{22} & X_{23}\\
# .. & .. & ..\\
# X_{n1} & X_{n2} & X_{n3} \end{matrix}\right]
# \ \ \ \ \  \theta = \left[ \begin{matrix}\theta_0\\ \theta_1\\ \theta_2\end{matrix}\right]
# \ \ \ \ \  y = \left[ \begin{matrix}y_0\\ y_1\\ .. \\ y_n\end{matrix}\right]$$
# $$ \theta = (X^TX)^{-1} X^T y$$
# 

# In[6]:


XTXI = np.linalg.inv(np.dot(X.transpose(),X))
theta = np.dot(XTXI,np.dot(X.transpose(),y))
print(theta)


# In[7]:


y_pred = np.dot(X,theta)


fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('One day ago/Two day ago Vs Todays value')

ax1.plot(X[:,1],y,'.r',label = 'True y')
ax1.plot(X[:,1],y_pred,'.b', label = 'Predict y')

ax1.legend()
ax2.plot(X[:,2],y,'.y', label = 'True y')
ax2.plot(X[:,2],y_pred,'.b', label = 'Predict y')
ax2.legend()

plt.show()


# In[ ]:




