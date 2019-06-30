
# coding: utf-8

# In[3]:


#Linear Regression with One Variable
#importing necessary library files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


#loading the data file  
data=pd.read_csv("C:\\Octave\\ex1.txt",header=None)


# In[5]:


#plotting the given data
x=data.iloc[:,0] 
y=data.iloc[:,1]
m=len(y)
plt.scatter(x, y)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()


# In[6]:


#setting the parameters
x=x[:,np.newaxis]
y=y[:,np.newaxis]
theta=np.zeros([2,1])
iterations=1500
alpha=0.01
ones = np.ones((m,1))
x=np.hstack((ones,x))


# In[8]:


#defining cost function
def computeCost(x,y,theta):
    t=np.dot(x,theta)-y
    return np.sum(np.power(t,2))/(2*m)
j=computeCost(x,y,theta)
print(j)


# In[7]:


#defining gradient descent function and finding the new theta
def gd(x,y,theta,alpha,iterations):
    for _ in range(iterations):
        temp = np.dot(x, theta) - y
        temp = np.dot(x.T, temp)
        theta = theta - (alpha/m) * temp
    return theta
theta = gd(x, y, theta, alpha, iterations)
print(theta)


# In[9]:


#checking the new value of cost function
j=computeCost(x,y,theta)
print(j)


# In[12]:


#plotting the the resultant line on the given data
plt.scatter(x[:,1], y)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.plot(x[:,1], np.dot(x, theta))
plt.show()

