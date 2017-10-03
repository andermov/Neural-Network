import numpy as np
import math

"""Neural network with one hidden layer.
1 input, 1 output, 5 hidden nodes"""


"""Initialisation of the weights"""
bias1=0.5
bias2=0.5
a=np.zeros((6))


"""Data into arrays"""
data=np.loadtxt("data.txt")

xdata=np.array(data[:,0])
ydata=np.array(data[:,1])

"""Training of the network"""
"""Maximum likelihood approach with the given data (from file)."""

def error(x,t):
	i=0
	for w in w1s:
		a[i]=np.tanh(x*w+bias1)
		i+=1

	i=0
	y=0
	for w in w2s:
		y+=w*a[i]+bias2
		i+=1
	err=y-t
	return(err)

def error1(x,t):
	i=0
	for w in w1s:
		a[i]=np.tanh(x*w+bias1)
		i+=1

	i=0
	y=0
	for w in w2s:
		y+=w*a[i]+bias2
		i+=1
	err=(1-a**2)*w2s*(y-t)
	return(err)

def error2(x,t):
	i=0
	for w in w1s:
		a[i]=np.tanh(x*w+bias1)
		i+=1

	i=0
	y=0
	for w in w2s:
		y+=w*a[i]+bias2
		i+=1
	err=a*(y-t)
	return(err)

w1sarray=np.zeros((10,6))
w2sarray=np.zeros((10,6))
errarray=np.zeros((10))
biasarr1=np.zeros(10)
biasarr2=np.zeros(10)
prec=0.0005
for j in range(10):
	dev=10
	w1s=5*np.random.rand((6))-2.5
	w2s=5*np.random.rand((6))-2.5
	bias2=np.random.rand(1)-0.5
	bias1=np.random.rand(1)-0.5
	while (dev > prec):
		w1s0=w1s
		w2s0=w2s
		bias20=bias2
		bias10=bias1
		for i in range(len(xdata)):
			w1s=w1s-0.01*error1(xdata[i],ydata[i])
			w2s=w2s-0.01*error2(xdata[i],ydata[i])
			bias2=bias2-0.01*error(xdata[i],ydata[i])
			bias1=bias1-0.01*bias1*error(xdata[i],ydata[i])
		dev=math.sqrt(np.linalg.norm(w1s0-w1s)**2+np.linalg.norm(w2s0-w2s)**2+np.linalg.norm(bias2-bias20)**2+np.linalg.norm(bias1-	bias10)**2)
	totalerror=0
	for i in range(len(data)):
		totalerror+=(error(xdata[i],ydata[i]))**2
	errarray[j]=totalerror
	w1sarray[j,:]=w1s
	w2sarray[j,:]=w2s
	biasarr1[j]=bias1
	biasarr2[j]=bias2
	print (j)
print(errarray)
j=0
for i in range(10):
	if (errarray[i]<errarray[j]):
		j=i

w1s=w1sarray[j,:]
w2s=w2sarray[j,:]
bias1=biasarr1[j]
bias2=biasarr2[j]

"""The Neural Network output"""
file = open("output.txt", "w")
for j in range(800):
	x=float(j)/200.0
	
	i=0
	for w in w1s:
		a[i]=np.tanh(x*w+bias1)
		i+=1
	
	i=0
	y=0
	for w in w2s:
		y+=w*a[i]+bias2
		i+=1
	file.write(str(x)+"	"+str(y)+"\n")

file.close() 
