#!/usr/bin/python3

import math
import string
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import optimize
from scipy.linalg import expm, logm
import os.path
from os import walk
import pylab
from collections import defaultdict
import scipy.integrate as integrate
from pandas import *
import pandas as pd
#from rpy2.robjects.packages import importr
#utils = importr('utils')
#utils.install_packages('gutenbergr', repos='https://cloud.r-project.org')
#utils.install_packages('ctmcd')
#import rpy2.robjects as ro
#from rpy2.robjects import pandas2ri
#pandas2ri.activate()
#ctmcd = importr('ctmcd')
import os
import time
from scipy import linalg
from random import *


from helperFunctions import *
from constants import *

#Defined for i->j
def diagonalAdjustment(matrix, tau=1, k=0, epsilon=0.001, maxIterations=20):
    #input is ETM or EPM, returns generator
    #take log
    logMatrix = isRealLog(normalizeMatrix(matrix, k=k), epsilon=eps, maxIterations=maxIters)/tau
#    logMatrix=logm(matrix)
    #set off diagonals to zero
    for i in range(logMatrix.shape[0]):
        for j in range(logMatrix.shape[0]):
            if(i!=j and logMatrix[i,j]<0):
                logMatrix[i,j] = 0

    #make diagonals the negative sum of rest of row
    for i in range(logMatrix.shape[0]):
        logMatrix[i,i]=0 #first set diagonals to zero
        logMatrix[i,i] = -1 * logMatrix[i].sum() #by row
    return logMatrix

#Defined for i->j
def weightedAdjustment(matrix, tau=1, k=0, epsilon=0.001, maxIterations=20): #input is ETM or EPM
    #returns Generator
    #take log
    logMatrix = isRealLog(normalizeMatrix(matrix, k=k), epsilon=eps, maxIterations=maxIters)/tau
    #set off diagonals to zero as in DA
    for i in range(logMatrix.shape[0]):
        for j in range(logMatrix.shape[0]):
            if(i!=j and logMatrix[i,j]<0):
                logMatrix[i,j] = 0
    absMatrix = abs(np.copy(logMatrix))
    for i in range(logMatrix.shape[0]):
        for j in range(logMatrix.shape[0]):
            matrix[i,j] = logMatrix[i,j] - absMatrix[i,j] * logMatrix[:,i].sum() / absMatrix[:,i].sum()
    return matrix



def EM(matrix, tau=1, k=0):
    df = npMatrixToPDdf(matrix)
    DAmat = diagonalAdjustment(matrix, tau=tau, k=k)
    EMmat = ctmcd.gm(tm=df, te=tau, method="EM", gmguess=DAmat)[0]
    return EMmat

def MLE(matrix, t=1, iterations=250000,pseudobeta=1, noiseR=0.1, noiseP=0, smooth=0.0001):
    N0=matrix 
    N=normalizeMatrix(matrix+1)
    n=N.shape[0]
    P=guessP(N)
    R=guessR(N,P)


    #R = np.random.rand(R.shape[0], R.shape[0])
#    for i in range(R.shape[0]):
#      R[i,i] = 0
#      R[i,i] = -1 * R[:,i].sum() #should be column sum
#    print("randR")
#    for i in detailedBalance(R):
#      print(i)

    print("#Iterations: %s"%iterations)
    print("#Pseudobeta: %s"%pseudobeta)
    print("#noiseR: %s"%noiseR)
    print("#noiseP: %s"%noiseP)
    print("#smooth: %s"%smooth)
   
    logl = calcLL(N0,R,t)
    
    seed()

    rejected=0
    rejectedLastThousand=0
    adjusted=np.zeros(n)
    for step in range(1,iterations+1):
        i=randint(0,n-1)
        if (t%2==0 or noiseP==0):
            j=n
            while j>=n or j<0:
                j=i+1-2*randint(0,1)
            dr=-R[i,j]
            #while R[i,j]+dr<=0:# or R[j,i]+P[i]/P[j]*dr<=0: #off diagonals need to still be greater than 0
            while R[i,j]+dr<=0 or R[j,i]-dr<=0: #off diagonals need to still be greater than 0
                # or R[j,i]+P[j]/P[i]*dr<=0
                dr=(random()-0.5)*noiseR
            R[i,j]+=dr
            R[i,i]-=dr
            #R[j,i]-=dr
            #R[j,j]+=dr
           # R[j,i]+=dr*P[i]/P[j]
           # R[j,j]-=dr*P[i]/P[j]
        else:
            dp=(random()-0.5)*noiseP
            for j in range(n):
                if i!=j:
                    P[j]-=(dp*P[i])/n
            P[i]*=(1+dp)
            if (i<n-1):
                R[i+1,i+1]-=R[i+1,i]*dp
                R[i+1,i]*=1+dp
            if (i>0):
                R[i-1,i-1]-=R[i-1,i]*dp
                R[i-1,i]*=1+dp

        #r=sp.linalg.expm(R)
        loglt=0
        #for ii in range(n):
        #    for jj in range(n):
        #        if N[ii,jj]*r[ii,jj]>0:
        #            loglt+=log(r[ii,jj])*N[ii,jj]
        #if smooth>0:
        #    for ii in range(n-1):
        #        D[ii]=R[ii,ii+1]*sqrt(P[ii+1]/P[ii])
        #    for ii in range(n-2):
        #        loglt-=(D[ii]-D[ii+1])**2/(2*smooth**2)+(log(P[ii]/P[ii+1]))**2/(2*smooth**2)

        loglt = calcLL(N0, R, t)
        dlog = (loglt) - (logl) #these numbers are always negative, thus if loglt>logl this will be positive
        r = random()
        if math.isnan(loglt) or math.isinf(loglt) or  (r>np.exp(pseudobeta*(dlog))): #rejection criterion
            if (t%2==0 or noiseP==0):
                R[i,j]-=dr
                R[i,i]+=dr
                #R[j,i]+=dr
                #R[j,j]-=dr
                ##R[j,i]-=dr*P[i]/P[j]
                #R[j,j]+=dr*P[i]/P[j]
            else:
                P[i]/=(1+dp)
                for j in range(n):
                    if i!=j:
                        P[j]+=(dp*P[i])/n
                if (i<n-1):
                    R[i+1,i]/=1+dp
                    R[i+1,i+1]+=R[i+1,i]*dp
                if (i>0):
                    R[i-1,i]/=1+dp
                    R[i-1,i-1]+=R[i-1,i]*dp
            rejected +=1.
            rejectedLastThousand +=1.
        else:
            logl=loglt
            adjusted[i]+=1
        if step%1000==0:
###########
            #noiseR = noiseR * min(1,(1 - rejectedLastThousand/1000)+0.5)
	    #noiseR = 1 - rejectedLastThousand/1000
            #noiseP = noiseP * min(1,(1 - rejectedLastThousand/1000)+0.5)
            #if (rejectedLastThousand/1000*100 > 95):
            #  print("Iteration: %d, Logl: %.2f, TotalReject: %.2f%%, RecentReject: %.2f%%, noiseR = %.2f" %(step, logl, rejected/float(step)*100, rejectedLastThousand/1000*100, noiseR))
            #  return R
            print("Iteration: %d, Logl: %.2f, TotalReject: %.2f%%, RecentReject: %.2f%%, noiseR = %.2f" %(step, logl, rejected/float(step)*100, rejectedLastThousand/1000*100, noiseR))
############
            #print("Iteration: %d, Logl: %.2f, TotalReject: %.2f%%, RecentReject: %.2f%%" %(step, logl, rejected/float(step)*100, rejectedLastThousand/1000*100))
            rejectedLastThousand=0
            if step%5000==0:
                for i in detailedBalance(R):
                    print(i)
    return R

#Helper function by which to optimize the frobenius distance between the two matrices.
def optimizeFunc(x, i, j, Q, P):#x is exp(Q(q_{i,j}))
    Q[i,i] += Q[i,j] - x
    Q[i,j] = x
    return frobenius(iterative_expm(Q), P)

                
#Input is a ETM or EPM
def CWO(matrix, tau=1, k=0, epsilon=0.001, maxIterations=20):
    calculations=0
    #It is noted that any method can be used here.  Not just DA.
    Q = diagonalAdjustment(matrix, tau=tau, k=k, epsilon=eps, maxIterations=maxIters)
    matrix = normalizeMatrix(matrix, k=k)
    for i in range(Q.shape[0]):
        for j in range(Q.shape[0]):
            if(i!=j):
                if(Q[i,j]>1e-10):
                    calculations+=1
                    #Run an optimization on each row over the first function defined in this cell
                    x = optimize.fmin(optimizeFunc, Q[i,j], args=(i,j,Q,matrix), maxiter=200, full_output=False, disp=False)[0]#argmin(i, j, Q, c)
                    Q[i,j] = x
    return Q

def QOG(matrix, tau=1, k=0, epsilon=eps, maxIterations=maxIters):
    logMatrix = isRealLog(normalizeMatrix(matrix,k=k), epsilon=eps, maxIterations=maxIters)/tau
    
    #step 2 of algorithm
    sortedMatrix, unsortKey = sortMatrix(logMatrix)
    
    #step 3 of algorithm
    m = np.zeros(matrix.shape[0])
    for i in range(matrix.shape[0]):
        m[i] = findMValue(sortedMatrix[i])
    
    #step 4 of algorithm
    copyMatrix=np.copy(sortedMatrix)
    for i in range(matrix.shape[0]):#for each row
        for j in range(2,int(m[i])+1):#include m[i]
            sortedMatrix[i,j]=0
        for j in range(int(m[i])+1,matrix.shape[0]):#for each value not zero'd
            for k in range(int(m[i])+1,matrix.shape[0]): #summation
                sortedMatrix[i,j] -= copyMatrix[i,k] / (matrix.shape[0] - m[i] + 1) 
            sortedMatrix[i,j] -= copyMatrix[i,0] / (matrix.shape[0] - m[i] + 1) 
        for k in range(int(m[i]+1),matrix.shape[0]):
            sortedMatrix[i,0] -= copyMatrix[i,k] / (matrix.shape[0] - m[i] + 1)
        sortedMatrix[i,0] -= copyMatrix[i,0] / (matrix.shape[0] - m[i] + 1)

    #step 5 - shuffle rows back into order.
    quasi = unsortMatrix(sortedMatrix, unsortKey)
    return quasi

def findMValue(array): #step 3 of algorithm
    n = len(array)-1 #last index
    val=0
    for i in range(1,n+1): #i loops from 1 to n
        val = (n+1-i)*array[i+1]-array[0]
        for j in range(n-i):#from 0 to n-1-i
            val -= array[n-j]
        if(val>=0): #truth condition of algorithm
            return i

    return -1 #otherwise return that row cannot be optimized.

def sortMatrix(matrix): #returns sortMatrix and unsortKey
    sortMatrix = np.copy(matrix)
    for i in range(matrix.shape[0]):
        sortMatrix[i].sort()
    sortMatrix = sortMatrix

    unsortKey = np.zeros((matrix.shape[0], matrix.shape[0]))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            f=0
            while(unsortKey[i,j]==0):
                if(sortMatrix[i,f]==matrix[i,j]):
                    unsortKey[i,j] = f + 1
                f+=1
    return sortMatrix, unsortKey

def unsortMatrix(matrix, key): #take in sorted matrix and key to unsort
    unsortedMatrix = np.zeros((matrix.shape[0],matrix.shape[0]))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            unsortedMatrix[i,j] = matrix[i,int(key[i,j])-1]
    return unsortedMatrix
