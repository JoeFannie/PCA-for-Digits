#!/usr/bin/env python

import sys
import numpy as np
import string
from numpy import linalg as la
import matplotlib.pyplot as plt
import pylab

def image2col(image, num_rows, num_cols):
    col= np.zeros((num_rows*num_cols,1))
    for i in np.arange(0,num_cols):
        for j in np.arange(0,num_rows):
            col[i*num_cols+j,0] = string.atoi(image[j:j+1][0][i])
    return col

def cols2matrix(num_dict, label):
    num_sample = len(num_dict[label])
    dim_feature = num_dict[label][0].size
    matrix = np.zeros((dim_feature, num_sample))
    for i in np.arange(0,num_sample):
        matrix[:,i] = num_dict[label][i].reshape(1024,)
        matrix[:,i] -= sum(matrix[:,i])/dim_feature
    return matrix

f = open("optdigits-orig.cv/optdigits-orig.cv","r")
num_rows = 32
num_cols = 32
lines = f.readlines()
total_num = string.atoi(lines[8:9][0][7:])
numbers = lines[21:]
f.close()

num_dict = {}
for i in np.arange(0,10):
    num_dict[i] = []
    
for i in np.arange(0,total_num):
    image = numbers[i*(num_rows+1):i*(num_rows+1)+32]
    col = image2col(image,num_rows,num_cols)
    label = string.atoi(numbers[i*(num_rows+1)+32:i*(num_rows+1)+33][0])
    num_dict[label].append(col)
    
x=[]
y=[]
for label in np.arange(0,10):
    matrix = cols2matrix(num_dict,label)
    X = np.dot(matrix,matrix.T)
    S,U = la.eig(X)
    PCA_matrix = np.dot(U[0:2,:],matrix)
    x.append(PCA_matrix[0,:])
    y.append(PCA_matrix[1,:])
    print "Number %d is finished..." % label

fig = plt.figure()
colors = "rbgykmc"
styles = "o."
for i in np.arange(0,10):
    color = colors[i%7]
    style = styles[i%2]
    plt.plot(x[i],y[i],color+style,label="Number %d" % i)
pylab.xlim([-0.02,0.06])
plt.legend(loc=4)
fig.suptitle("PCA for Digits")
plt.show()