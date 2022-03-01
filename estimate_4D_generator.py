import numpy as np
from GeneratorFunctions import *

T = np.load("data/transitions/transitions_123456.npy")
#da = diagonalAdjustment(T,tau=1e-14)
T_norm = np.zeros_like(T,dtype=float)

for i,j in zip(range(T.shape[0]),range(T.shape[2])):
    T_sub = T[i,:,j,:]
    row_sum = T_sub.sum(axis=0)
    norm = T_sub / row_sum[:,np.newaxis]
    norm[(np.isnan(norm)) | (np.isinf(norm))] = 0
    T_norm[i,:,j,:] = norm
print(T_norm)
print(T_norm.sum(axis=0))
