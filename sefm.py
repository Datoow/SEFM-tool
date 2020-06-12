import numpy as np
import scipy.sparse as sp
from fastFM import sgd
import time

def subcode(X_train, X_test, b, F, minf, maxf):
        X_train_o = np.zeros((len(X_train), F*b))
        X_test_o = np.zeros((len(X_test), F*b))
        a = []
        p = []
        B = np.zeros(F, dtype=int)
        for f in range(0, len(X_train[0])):
            fl = []
            a.append([])
            p.append([])
            for l in range(0, len(X_train)):
                fl.append(X_train[l][f])
            for l in range(0, len(X_test)):
                fl.append(X_test[l][f])
            a[f],p[f] = np.unique(fl, return_inverse = True)
        for f in range(0, F):
          if len(a[f]) == 1:
            B[f] = int(0)
          elif len(a[f]) >= b:
            B[f] = int(b)
          else:
            B[f] = int(len(a[f]))
        for l in range(0, len(X_train)):
          f_en = 0
          for f in range(0, F):
            if B[f] == 0:
              f_en = f_en
            elif B[f] == b:
              if (X_train[l][f] - minf[f]) == (maxf[f] - minf[f]):
                X_train_o[l][f_en + b - 1] = 1
              else:
                X_train_o[l][f_en + int(float(X_train[l][f] - minf[f]) / (maxf[f] - minf[f]) * b)] = 1
              f_en += b
            else:
              X_train_o[l][f_en + p[f][l]] = 1
              f_en += B[f]
        if f_en < F*b:
          X_train_b = np.zeros((len(X_train), f_en))
          for i in range(len(X_train)):
            for j in range(f_en):
              X_train_b[i][j] = X_train_o[i][j]
          X_train_o = X_train_b
        for l in range(0, len(X_test)):
          f_en = 0
          for f in range(0, F):
            if B[f] == 0:
              f_en = f_en
            elif B[f] == b:
              if (X_test[l][f] - minf[f]) == (maxf[f] - minf[f]):
                X_test_o[l][f_en + b - 1] = 1
              else:
                X_test_o[l][f_en + int(float(X_test[l][f] - minf[f]) / (maxf[f] - minf[f]) * b)] = 1
              f_en += b
            else:
              X_test_o[l][f_en + p[f][len(X_train) + l]] = 1
              f_en += B[f]
        if f_en < F*b:
          X_test_b = np.zeros((len(X_test), f_en))
          for i in range(len(X_test)):
            for j in range(f_en):
              X_test_b[i][j] = X_test_o[i][j]
          X_test_o = X_test_b
        return X_train_o, X_test_o, B, f_en

    
class SEFM():
     def __init__(self, k, b, step, F, n_iter = 1000, init_stdev = 0.01):
        self.n_iter = n_iter
        self.init_stdev = init_stdev
        self.k = k
        self.b = b
        self.step = step
        self.F = F
     
     def se_fm(self, X_train, y_train, X_test, y_test, minf, maxf):
        X_train_o, X_test_o, self.B, self.length = subcode(X_train, X_test, self.b, self.F, minf, maxf)
        fm = sgd.FMClassification(n_iter=self.n_iter, init_stdev=self.init_stdev, rank=self.k, step_size=self.step)
        X = sp.csc_matrix(np.array(X_train_o), dtype=np.float64)
        fmm = fm.fit(X, y_train)
        X2 = sp.csc_matrix(np.array(X_test_o), dtype=np.float64)
        self.w_ = fmm.w_
        self.V_ = fmm.V_
        self.w_tilde = np.matmul(self.V_.T, self.V_)
        y_pre = fm.predict(X2)
        s = 0.0
        for l in range(0, len(y_pre)):
            if y_test[l] == round(y_pre[l]):
                s += 1.0
        return s/len(y_pre)


