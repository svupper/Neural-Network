import numpy as np
import math as m

X=np.random.randn(9,1)
X=np.concatenate[X,1]
Y=np.random.random_sample
W=np.random.randn(10,1)*0.01    # +bias

v=np.dot(W,X)


loss=lambda y,yd:(1/2)*(y-yd)
sig=lambda v:(m.exp(v)-m.exp(-v))/(m.exp(v)-m.exp(-v))
sigp=lambda v:sig(v)(1-sig(v))

#dx=dx*dv #dL gradient incoming, dv gradient of the gate, dx gradient outputting
y=sig(v)

act={}

##
#
#for i in range(size(W(1,1,:))):
#	y=v(W*X)
#	
#	
#
#for i in X:
#	dL=1
#	dx=dL*dv
#    */
##
