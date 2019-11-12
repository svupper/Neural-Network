# TODO:
#Implement multiple layer model
#implement multiple perceptrons on layers




import numpy as np



import math as m
import matplotlib.pyplot as plt

labda=0.1
X=np.random.randn(50,2) # x1 and x2
Y=2*np.random.randint(0,2,size=(50))-1

W=np.random.randn(1,2)*0.01    
bias=0.1

loss=lambda y,yd:m.pow(y-yd,2)
sig=lambda v:(m.exp(v)-m.exp(-v))/(m.exp(v)+m.exp(-v))
sigp=lambda v:sig(v)*(1-sig(v))

fig=plt.figure()
ax=plt.axes()
x= np.linspace(0,10,1000)
ax.plot(x,(-W[0,0]*x-bias)/W[0,1])

for i in range(len(X)):  
    if Y[i]==1:
        ax.scatter(X[i,0],X[i,1],c='red')
    else:
        ax.scatter(X[i,0],X[i,1],c='blue')

plt.show()
e=0
for i in range(len(X)):
    #forward pass
    v=np.dot(W,X[i,:])+bias
    y=sig(v)
    
    #loss function
    l=loss(y,Y[i])
    plt.scatter(e,l,c='green')
    e+=1
    #backward pass
    dl=sigp(v)*(1/2)*(y-Y[i])
    
    dW= np.asarray([j*labda*dl for j in X[i]])
    #W=np.transpose(W)
    W = np.subtract(W,dW)
    bias-=labda*dl

plt.show()

fig2=plt.figure()
ax=plt.axes()
x= np.linspace(0,10,1000)
ax.plot(x,(-W[0,0]*x-bias)/W[0,1])

for i in range(len(X)):  
    if Y[i]==1:
        ax.scatter(X[i,0],X[i,1],c='red')
    else:
        ax.scatter(X[i,0],X[i,1],c='blue')
        
plt.show()