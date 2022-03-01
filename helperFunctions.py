#!/usr/bin/python
import sys
import math
import string
import glob
import numpy as np
import scipy as sp
from scipy import optimize
from scipy.linalg import expm, logm
import os.path
from os import walk
from collections import defaultdict
import scipy.integrate as integrate
from pandas import *
import pandas as pd
import os
import time
from scipy import linalg
from random import *
from constants import *





def writeLineAndDB(db, line, output):
#simple function to write out the line of the potential and the detailed balance.
#Arguments:#
#db - 1Darray which contains values of detailed balance
#line - coefficient values which correspond to increasing values of x^n
#output - filename for output file
#Returns:#
#none
#TODO: make writing the line into a for loop for variable order polynomials.
    file = open(output, "a")
    file.write("#Detailed balance followed by equation of the line\n")
    for i in db:
        file.write(str(i)+"\n")
    z=line
    file.write("\n")
    file.write("y=%.6fx^8 + %.6fx^7 + %.6fx^6 + %.6fx^5 + %.6fx^4  + %.6fx^3 + %.6fx^2 + %.6fx + %.6f"%(z[0],z[1],z[2],z[3],z[4],z[5],z[6], z[7], z[8]))
    file.close()



def detailedBalance(mat, ij=True):#input is ETM or EPM
#Calculate detailedBalance for Transition Matrix or Generator Matrix.  
#This is a J->I convention, i.e. P_i,j is the transitions from J to I
#arguments:#
#mat - input matrix
#ij - specifies the matrix represents i->j transitions, false is j->i convention.
#Returns:#
#FE - 1Darray of free energies as determined through detailed balance.
#TODO: Add input arguments for kBT
    FE = np.zeros(mat.shape[0])
    for i in range(mat.shape[0]-1):
        if(mat[i+1,i]==0 or mat[i,i+1]==0):
            FE[i] = 0
        else:
            if ij:
                FE[i] = FE[i-1] - kBT * np.log(mat[i,i+1]/mat[i+1,i])
            else:
                FE[i] = FE[i-1] - kbT * np.log(mat[i+1,i]/mat[i,i+1])
    FE = FE + FE.min()
    return FE



#frobenius is a measure of distance between two matrices
#this function takes two square matrices (q1, q2) and finds their frobenius distance
#find the distance between the two matrices
def frobenius(q1, q2):
#Arguments:#
#q1/2 - input matrices to calculate distance

    n = q1.shape[0]
    dist = 0.0
    for i in range(n):
        for j in range(n):
            dist += (abs(q1[i,j]) - abs(q2[i,j]))**2
    return np.sqrt(dist)




#Iteratively take the log of a matrix.  This is a function which always produces real numbers
def isRealLog(matrix, epsilon=eps, maxIterations=maxIters):
#Arguments:#
#matrix - take the log of this matrix
#epsilon - convergenace criterion
#maxIterations - maximum iterations until we just say it has converged arbitrarily
    matrix = normalizeMatrix(matrix)
    iterations = 1
    #print("taking log")
    ret = np.zeros((matrix.shape))
    previousIteration = np.copy(matrix)
    i=1
    ret = ret + float(i)/float(iterations) * (matrix - np.identity(len(matrix[0])))**iterations
    distance = frobenius(previousIteration, ret)
    #print(distance)
    while(distance>epsilon):
        iterations+=1
        i=i*-1
        ret = ret + float(i)/float(iterations) * (matrix - np.identity(len(matrix[0])))**iterations
        distance = frobenius(ret, previousIteration)
        previousIteration = np.copy(ret)
        #print(distance)
        if iterations == maxIterations+1:
            distance = 0
    #print("Log Iterations: %s" %iterations)
    return ret



#iteratively take the exponential of a matrix as per the expansion above.
def iterative_expm(matrix, epsilon=eps, maxIterations=maxIters):
#see isRealLog
    #do not normalize first
    previousIteration = np.copy(matrix)
    iters=1
    ret = np.identity(len(matrix[0]))
    distance = epsilon + 1
    while(distance > epsilon):
        ret += (matrix**iters)/float(math.factorial(iters))
        distance = frobenius(ret, previousIteration)
        previousIteration = np.copy(ret)
        if iters == maxIterations:
            distance = 0
        iters += 1
    return ret




def calcLL(Nij, Rij, t=1):#input is empirical transition matrix
#Calculate Log Likelihood
#Nij - Empirical Transition Matrix
#Rij - Rate Matrix to compare
#t - tau/dt (i.e. window size as reported in paper)
#Returns:#
#Log Likelihood
    n = Nij.shape[0]
    logl = 0
    tiny = np.zeros((n,n))
    tiny.fill(1.e-16)
    #r = np.maximum(iterative_expm(t*Rij), tiny)
    r = np.maximum(expm(t*Rij), tiny)
    for i in range(n):
        for j in range(n):
            #if r[i,j]*Nij[i,j]>0:
                logl+=Nij[i,j]*np.log(r[i,j])

    return logl



def guessP(N):
    n = N.shape[0]
    # initial guess for P
    P=np.zeros((n))
    TotalP=1.0
    N = normalizeMatrix(N+0.00000000000001)
    for i in range(n):
        if i==0:
            P[i]=1.0
        else:
            P[i]=N[i-1,i]*P[i-1]/N[i,i-1]
            #print(i, P[i], "=", N[i,i-1],"/",N[i-1,i],"*",P[i-1])
            TotalP+=P[i]

    for i in range(n):
        P[i]/=TotalP
    P = P / P.sum()
    for i in P:
        print(i)
    return P




def guessR(N, P):
    # initial guess for R
    n = N.shape[0]
    R=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i==j+1 or j==i+1:
                R[i,j]=P[j]*(N[i,j]+N[j,i])/(N[i,i]*P[j]+N[j,j]*P[i])
            else:
                R[i,j]=0
    #set diagonals to negative of sum of rest of ####row - Column.
    for l in range(n):
        R[l,l]-=R[l].sum()
    print("R")
    for i in detailedBalance(R):
        print(i)
    return R




#Simply read a square matrix, return said matrix as a 2d numpy array
def readMatrix(fileName):
    i=0
    file = open(fileName)
    for line in file:
        if "," not in line:
            array = line.split()
        else:
            array = line.split(", ")
        if(i==0):
            matrix = np.zeros((len(array),len(array)))
        matrix[i] = array
        i+=1
    if(matrix.shape[0]!=matrix.shape[1]):
        print("Error: Your Matrix is not square, Exiting!")
        quit()
    file.close()
    return matrix

#Simply take a NxN numpy array and write it to a file.
def writeMatrix(matrix, fileName):
    file = open(fileName, "w")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            file.write("%f\t"%matrix[i,j])
        file.write("\n")
    file.close()




#Function to Normalize a square Matrix.
#k=0 is row normalization, k=1 is column normalization.
#Row Normalization corresponds to i->j
#Column Normalization corresponds to j->i
def normalizeMatrix(matrix, k=0):
    normalizedMatrix = np.copy(matrix).astype(float)
    if (k==0): #by row
        for i in range(normalizedMatrix.shape[0]):
            normalizedMatrix[i] = normalizedMatrix[i]/normalizedMatrix[i].sum()
        return normalizedMatrix
    else: #by column
        for i in range(normalizedMatrix.shape[0]):
            normalizedMatrix[:,i] = normalizedMatrix[:,i]/normalizedMatrix[:,i].sum()
        return normalizedMatrix

def compressMatrix_by2(m):
    bins = m.shape[0]
    compressedMatrix = np.zeros((int(bins/2),int(bins/2)))
    for i in range(0,bins,2):
        for j in range(0,bins,2):
            compressedMatrix[int(i/2),int(j/2)] = m[i,j]  + m[i+1,j] + m[i,j+1] + m[i+1,j+1]
    return compressedMatrix


#This is a helper function to convert a np Matrix to a pandas data frame
def npMatrixToPDdf(matrix):
    indexList = []
    columnsList = []
    for i in range(matrix.shape[0]):
        indexList.append(i)
    for i in range(matrix.shape[1]):
        columnsList.append(i)
    return pd.DataFrame(data=matrix, index=indexList, columns=columnsList)

#Calculate D as in Hummer,2005.  Both possible methods are included, the 2nd is commented.
def calcD(generatormatrix,potentialFunction=0, bins=47, width=4):#,binWidth=0.08695652174): #4/46
    D = np.ones(generatormatrix.shape[0]-1)
    binWidth=width/bins
    generatormatrix+=1/bins
    for i in range(len(D)):
        D[i] = binWidth**2 * np.sqrt(generatormatrix[i+1,i]) * np.sqrt(generatormatrix[i,i+1])
        #D[i] = binWidth**2 * generatormatrix[i+1,i] * np.exp(-1 * beta *(potentialFunction((i+1)*binWidth-2)-potentialFunction((i+1)*binWidth-2)))
    return D

#discrete version
def MFPT_from_Generator(filename, tau, first=12, second=34, delta=1, bins=47, width=4, createTables=False):
    a = readMatrix(filename)#read Generator
    c = detailedBalance((a))
    #z = np.polyfit(np.arange(len(c))/len(c)*4-2, c-c.min(), 8)
    potentialFunction = c
    #potentialFunction = np.poly1d(z)
    #wellErr = abs(potentialFunction(-1)-potentialFunction(1)) #34/35, 11/12
    D = calcD(a/tau, potentialFunction, bins=bins, width=width)#+1e-6#calculate position dependent diffusion constants
    dAvg = np.average(D[12:34])
    #pylab.plot(np.arange(len(c)),potentialFunction(np.arange(len(c))),"r--")
    #print(D)
    def outerSum(lower=first, upper=second, delta=delta, D=D):
        value = 0
        i = lower
        j=1
        while(i<=upper):
            value += p(i) * innerSum(0, i) /D[j]* (width/bins)
            j+=1
            i+=delta
        return value
    def p(x, beta=beta):
        return np.exp(beta * potentialFunction[x])

    def n(x, beta=beta):
        return np.exp(-1*beta*potentialFunction[x])

    def innerSum(lower=first, upper=second, delta=delta):
        value = 0
        i = lower
        while(i<=upper):
            value += n(i) * delta * (width/bins)
            i+=delta
        return value
    if createTables==True:
        err = calcdGError(c)
        wellErr= abs((c[34]+c[35])/2  - (c[11] + c[12]) / 2)
        return outerSum(), dAvg, err, wellErr
    return outerSum(), dAvg

#calculate the \Delta G as per the height of the barrier vs the average of the two wells.
#Expected values for the project are the defaults.
def calcdG(detailedBal, maxRange=[14,29], min1Range=[4,21], min2Range=[21,40], totalRange=[5,41]): #input is a detailed Balance, returns dG
    detailedBal += 0-(detailedBal[totalRange[0]:totalRange[1]].min())
    MAX = max(detailedBal[maxRange[0]:maxRange[1]])
    MIN1 = min(detailedBal[min1Range[0]:min1Range[1]])
    MIN2 = min(detailedBal[min2Range[0]:min2Range[1]])
    return MAX - (MIN1+MIN2)/2

def calcDMinima(detailedBal, min1Range=[4,21], min2Range=[21,40]): #input is a detailed Balance, returns dG of two minima
    MIN1 = min(detailedBal[min1Range[0]:min1Range[1]])
    MIN2 = min(detailedBal[min2Range[0]:min2Range[1]])
    return abs(MIN1-MIN2)

#input is a detailed Balance and the expected dG, returns dG
def calcdGError(detailedBal, expected=2.5):
    dG = calcdG(detailedBal)
    return abs(expected - dG) / expected * 100
