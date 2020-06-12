from sklearn import datasets
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
import numpy as np

def pre_data(name):
  
  dataset = [[],[]]
  if(name == "moons"):
  	dataset = datasets.make_moons(n_samples=5000)
  elif(name == "circles"):
  	dataset = datasets.make_circles(n_samples=5000)
  else:
  	X,y = load_svmlight_file(name)
  	dataset[0] = X.toarray()
  	dataset[1] = y 
  	D1 = dataset[1][0]
  	for i in range(0, len(dataset[1])):
  	    if dataset[1][i] == D1:
  		    dataset[1][i] = 0
  	    else:
  		    dataset[1][i] = 1
  
  minF = [0.0]*len(dataset[0][0])
  maxF = [0.0]*len(dataset[0][0])
  minf = [0.0]*len(dataset[0][0])
  maxf = [0.0]*len(dataset[0][0])
  for f in range(0, len(dataset[0][0])):
      maxF[f] = dataset[0][0][f]
      minF[f] = dataset[0][0][f]
      for i in range(1, len(dataset[0])):
          maxF[f] = max(maxF[f],dataset[0][i][f])
          minF[f] = min(minF[f],dataset[0][i][f])
      if(maxF[f] > minF[f]):
          for i in range(0, len(dataset[0])):
              dataset[0][i][f] = (dataset[0][i][f] - minF[f])/(maxF[f] - minF[f])
      else:
          for i in range(0, len(dataset[0])):
              dataset[0][i][f] = 0
      maxf[f] = dataset[0][0][f]
      minf[f] = dataset[0][0][f]
      for i in range(1, len(dataset[0])):
          maxf[f] = max(maxf[f],dataset[0][i][f])
          minf[f] = min(minf[f],dataset[0][i][f])  
  
  
  X_train = [[],[],[],[],[],[],[],[],[],[]]
  y_train = [[],[],[],[],[],[],[],[],[],[]]
  X_test = [[],[],[],[],[],[],[],[],[],[]]
  y_test = [[],[],[],[],[],[],[],[],[],[]]
  for num in range(0, 5):
      X_train[num], X_test[num], y_train[num], y_test[num] = train_test_split(dataset[0],dataset[1],test_size=0.3,random_state=num)
  for num in range(5, 10):
      X_train[num], X_test[num], y_train[num], y_test[num] = train_test_split(X_train[num-5],y_train[num-5],test_size=0.3,random_state=num)
  
  for num in range(0,10):
      for l in range(0, len(X_train[num])):
              if y_train[num][l] == 0:
  		            y_train[num][l] = -1
      for l in range(0, len(X_test[num])):
              if y_test[num][l] == 0:
                  y_test[num][l] = -1
  return X_train, X_test, y_train, y_test, minF, maxF, minf, maxf
