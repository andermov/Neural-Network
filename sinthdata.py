import random
import math
import numpy

file= open("data.txt","w")
for i in range(20):
	x=(i+0.5)/5.0
	y=numpy.random.normal(0,0.1,1)[0]+math.log( x )
	file.write(str(x)+"	"+str(y)+"\n")
	
