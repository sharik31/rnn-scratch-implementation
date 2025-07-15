#!/usr/bin/env python
# coding: utf-8

# ## BUILDING RNN FROM SCRATCH :

# In[52]:


import numpy as np


# In[53]:


def softmax(x):
    e_x = np.exp(x - np.max(x))  
    return e_x / np.sum(e_x)


# ### Forward Propagation

# #### For one time step:

# In[54]:


def single_step_rnn(xt, a_prev,Wax,Waa,Wya,ba,by):
    a_next= np.tanh(np.dot(Wax,xt)+np.dot(Waa,a_prev)+ba)
    yt=softmax(np.dot(Wya,a_next)+by)
   
    cache = (a_next, a_prev, xt, Wax,Waa,Wya,ba,by)

    return a_next,yt,cache


# #### Forward propgation through time

# In[55]:


def rnn_forward(X,a0,Wax,Waa,Wya,ba,by):
    n_x,m,T=X.shape
    n_y, n_a = Wya.shape
    a=np.zeros((n_a,m,T))
    y_pred=np.zeros((n_y,m,T))

    a_next=a0
    caches=[]
    for t in range(T):
        xt=X[:,:,t]
        a_next,yt,cache=single_step_rnn(xt,a_next,Wax,Waa,Wya,ba,by)
        a[:,:,t]=a_next
        y_pred[:,:,t]=yt
        caches.append(cache)
        

    caches = (caches, X)
    
    return a, y_pred, caches

    


# ### Backpropagation 

# #### For one time step

# In[56]:


def single_step_backward(da_next,cache):
    a_next,a_prev,xt,Wax,Waa,Wya,ba,by=cache

    dtanh= da_next*(1-(a_next)**2)
    dWax = np.dot(dtanh, xt.T)        
    dWaa = np.dot(dtanh, a_prev.T)
    dba  = np.sum(dtanh, axis=1, keepdims=True) 
    dxt  = np.dot(Wax.T, dtanh)    
    da_prev = np.dot(Waa.T, dtanh) 

    gradients = {
        "dxt": dxt,
        "da_prev": da_prev,
        "dWax": dWax,
        "dWaa": dWaa,
        "dba": dba
    }
    
    return gradients


# #### Backpropgation through Time

# In[57]:


def rnn_backward(da, caches): 
    (list_caches, x) = caches  
    (a1, a0, x1,Wax,Waa,Wya,ba,by) = list_caches[0]  

    n_a, m, T = da.shape       
    n_x, m = x1.shape  

    dx    = np.zeros((n_x, m, T))
    dWax  = np.zeros((n_a, n_x))
    dWaa  = np.zeros((n_a, n_a))
    dba   = np.zeros((n_a, 1))
    da0   = np.zeros((n_a, m))
    da_prevt = np.zeros((n_a, m))
    
    for t in reversed(range(T)):
    
        da_current = da[:, :, t] + da_prevt
        
        gradients = single_step_backward(da_current, list_caches[t])
        
        dxt     = gradients["dxt"]
        da_prevt = gradients["da_prev"]
        dWaxt   = gradients["dWax"]
        dWaat   = gradients["dWaa"]
        dbat    = gradients["dba"]
        dx[:, :, t] = dxt
         
        dWax += dWaxt
        dWaa += dWaat
        dba  += dbat

    da0 = da_prevt

    gradients = {
        "dx": dx,
        "da0": da0,
        "dWax": dWax,
        "dWaa": dWaa,
        "dba": dba
    } 
    return gradients


# ### Ctegorical Cross Entropy Loss

# In[58]:


def compute_loss(y_pred, Y_true):

    m = Y_true.shape[1]
    loss = -np.sum(Y_true * np.log(y_pred + 1e-9)) / m
    return loss


# ### Backprop. for da

# In[59]:


def output_backward(y_pred, Y_true, caches):
    (list_caches, X) = caches
    T = Y_true.shape[2]
    n_a = list_caches[0][0].shape[0]
    m = Y_true.shape[1]
    n_y = y_pred.shape[0]
    
    dZy = y_pred - Y_true   # (n_y, m, T)
    da = np.zeros((n_a, m, T))
    dWya = np.zeros((n_y, n_a))
    dby = np.zeros((n_y, 1))
    
    for t in range(T):
        a_next, a_prev, xt, Wax, Waa, Wya, ba, by = list_caches[t]
        
        da[:, :, t] = np.dot(Wya.T, dZy[:, :, t])
        
        dWya += np.dot(dZy[:, :, t], a_next.T)
        dby  += np.sum(dZy[:, :, t], axis=1, keepdims=True)
    
    return da, dWya, dby


# ### Intializing RNN Dimensions, Inputs, and Targets

# In[60]:


np.random.seed(1)
n_x = 2; n_a = 4; n_y = 2; T = 3; m = 1

X = np.random.randn(n_x, m, T)
Y_true = np.zeros((n_y, m, T))
Y_true[0, 0, :] = 1 


# ### Initializing Weights and biases

# In[61]:


Wax = np.random.randn(n_a, n_x) * 0.1
Waa = np.random.randn(n_a, n_a) * 0.1
Wya = np.random.randn(n_y, n_a) * 0.1
ba = np.zeros((n_a, 1))
by = np.zeros((n_y, 1))
a0 = np.zeros((n_a, m))


# In[62]:


a, y_pred, caches = rnn_forward(X, a0, Wax, Waa, Wya, ba, by)


# In[63]:


loss = compute_loss(y_pred, Y_true)
print("Initial loss:", loss)


# In[64]:


da = output_backward(y_pred, Y_true, caches)


# In[65]:


print(grads['dx'])


# ### Applying gradient descent on weights and biases

# In[66]:


def update_parameters(Wax, Waa, Wya, ba, by, grads, dWya, dby, lr=0.01):
    Wax -= lr * grads["dWax"]
    Waa -= lr * grads["dWaa"]
    ba  -= lr * grads["dba"]
    Wya -= lr * dWya
    by  -= lr * dby
    return Wax, Waa, Wya, ba, by


# In[67]:


np.random.seed(1)
n_x = 2; n_a = 4; n_y = 2; T = 3; m = 1

X = np.random.randn(n_x, m, T)
Y_true = np.zeros((n_y, m, T))
Y_true[0, 0, :] = 1 


# ### Training 

# In[68]:


epochs = 10
lr = 0.1

for epoch in range(epochs):
    a, y_pred, caches = rnn_forward(X, a0, Wax, Waa, Wya, ba, by)
    
    loss = compute_loss(y_pred, Y_true)
    da, dWya, dby = output_backward(y_pred, Y_true, caches)
    grads = rnn_backward(da, caches)
    
    Wax, Waa, Wya, ba, by = update_parameters(Wax, Waa, Wya, ba, by, grads, dWya, dby, lr)
    
    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")

print("\nFinal predictions after training:")
print("Softmax probs:\n", y_pred[:,0,:])


# In[ ]:





# In[ ]:





# 
