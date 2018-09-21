# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 22:52:10 2018

@author: Xu
"""
import numpy as np
from scipy.sparse import diags
# Local linear kernel regression model. You can adjust the b array easily from the input or directly inside the function. 
# And the choice of b array (especially the earlier points) and x will directly affect the result of back calculation of T5(0).
# t is time. p is pressure. The default setting b=0.0001 is suited for t in seconds unit.

def local_linear_kernel_regression(t, p, x="auto", b=0.0001):
    
    if x=="auto":
        x=np.linspace(t[0], t[-1], 100)
        
    elif isinstance(x, (list, tuple, np.ndarray)):
        pass

    # You can adjust the equation here. Choose whatever curve of b along x, if you don't choose to do it in the input.
    # The function we use now is a shifted, flipped, and scaled logistic function. 
    if not isinstance(b, (list, tuple, np.ndarray)):
        b=5*b/(1+np.exp((x-t[0])/(0.8*b)))+b
    else:
        b=b
    
    # Kernels here
    try:
        k=np.array([np.exp(-(x-t[i])**2/(2*b**2)) for i in range(len(t))])
    except:
        print("error in calculating Gaussian kernels. Please check the input data to local_linear_kernel_regression()")
    
    # Below are calculating a formula you can find in Elements of Statistical Learning
    
    Onemat=np.array([1 for i in range(len(t))])

    B=np.transpose(np.stack([Onemat,t], axis=0))

    y=np.zeros(len(x))


    for i in range(len(x)):

        b=np.array([1,x[i]])
        W=diags(k[:,i],0)
        temp=np.transpose(B).dot(W*B)
        temp=np.linalg.inv(temp)
        #temp=np.linalg.inv(np.linalg.multi_dot([np.transpose(B),W,B]))
        y[i]=np.transpose(b).dot(temp).dot(np.transpose(B)).dot(W*p)
    
    return (x,y)
