import numpy as np
import math as m

labda=0.5
X=np.random.randn(10,2) # x1 and x2
Y=2*np.random.randint(0,2,size=(10))-1

W=np.random.randn(2,1)*0.01    
bias=10


loss=lambda y,yd:(y-yd)^2
sig=lambda v:(m.exp(v)-m.exp(-v))/(m.exp(v)-m.exp(-v))
sigp=lambda v:sig(v)(1-sig(v))



for i in range(len(X)):
    #forward pass
    v=np.dot(X(i),W)+bias
    y=sig(v)
    
    #loss function
    l=loss(y,Y(1))
    
    #backward pass
    dl=sigp(v)*(1/2)*(y-Y(1))
    W-=labda*dl*X(i)
    bias-=labda*dl

