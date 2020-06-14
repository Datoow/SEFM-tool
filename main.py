# coding: utf-8

from sklearn import datasets
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.sparse as sp
import time
import sefm
import pre
import expl

print("Please input a name or file of a dataset:")
name = input() 

X_train, X_test, y_train, y_test, minF, maxF, minf, maxf, Dataset = pre.pre_data(name) #data preparation

F = len(X_train[0][0]) #Number of features

print("Please input k:")
k = int(input())
print("Please input b:")
b = int(input())
print("Please input step_size:")
step = float(input())

pr = []
mean_acc = 0.0
pr.append('rank=%d b=%d step_size=%.2f'%(k,b,step))
start = time.time()
for num in range(0,5): 
    print(name, "#%d" %num)
    model = sefm.SEFM(k = k, b = b, F=F, step = step) #SEFM
    acc = model.se_fm(X_train[num], y_train[num], X_test[num], y_test[num], minf, maxf) #calculate accuracy
    #explanation
    print("==========================")
    expl.global_expl(model, minF, maxF)
    print("==========================")
    for i in range(0,5):
      print("Sample #%d" %i)
      expl.local_expl(model, Dataset[0][i], Dataset[1][i], minF, maxF)
      print("==========================")
    mean_acc += acc
    pr.append('%.2f%%' %(acc * 100))
end = time.time()
mean_acc /= 5
pr.append('mean=%.2f%%' %(mean_acc * 100))
pr.append('%fs' %((end-start)/5))
print(pr)
print("==========================")

print("The Accuracy is:")
print('%.2f%%' %(mean_acc * 100))
print("And Running Time is:")
print('%fs' %((end-start)/5))

