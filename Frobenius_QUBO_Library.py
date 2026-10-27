import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import time
import statistics as st
import scipy
import dimod
from dwave.samplers import SimulatedAnnealingSampler

#===================================LIBRARY====================================
#A routine computing the QUBO matrix for a given system A*x=b for a user-defined
#radix-2 representation with N1 integer qubits and N2 fractional qubits
def QUBO_Matrix(A,b,N1,N2):
  N=len(b)
  q=np.zeros([N,2*N*(N1+N2)])
  for i in range(N):
    for j in range(2*(N1+N2)):
      if j<N1:
        q[i][2*i*(N1+N2)+j]=2**(N1-1-j)
      elif N1<=j<N1+N2:
        q[i][2*i*(N1+N2)+j]=2**(N1-j-1)
      elif N1+N2<=j<2*N1+N2:
        q[i][2*i*(N1+N2)+j]=-2**(2*N1+N2-1-j)
      else:
        q[i][2*i*(N1+N2)+j]=-2**(2*N1+N2-j-1)
  Ad=np.matmul(A,q)
  M=len(q[0])
  Ad=np.matmul(A,q)
  QQ=np.zeros([M,M])
  for k in range(M):
    for n in range(M):
        if k>n:
            continue
        if k==n:
           QQ[k][n]=sum(Ad[j][n]*(Ad[j][n]-2*b[j]) for j in range(N))
        else:
           QQ[k][n]=2*sum(Ad[i][k]*Ad[i][n] for i in range(N))
  return q,QQ

#R=[r_1,...,r_N] is a vector containing the positions of the scatterers.
#g=[g_1,...,g_N] is a vector containing the "strengths" associated to each scatterer.
#k is the wavenumber of the exterior
#The Foldy_Lax_QUBO returns the equivalent Foldy-Lax matrix used for QUBO formulations.
def Foldy_Lax_QUBO(R,g1,g2,k):
  A=np.zeros([2*len(R),2*len(R)])
  ReG=np.zeros([len(R),len(R)])
  ImG=np.zeros([len(R),len(R)])
  for i in range(len(R)):
    for j in range(len(R)):
      if i==j:
        ReG[i][j]=1
        ImG[i][j]=0
        #G[i][j]=1
        A[i][j]=1
        A[i+len(R)][j+len(R)]=-1
      else:
        ReG[i][j]=g1[j]*scipy.special.spherical_jn(0,k*np.linalg.norm(R[i]-R[j]),derivative=False)
        ImG[i][j]=g2[j]*scipy.special.spherical_yn(0,k*np.linalg.norm(R[i]-R[j]),derivative=False)
        A[i][j]=ReG[i][j]
        A[i][j+len(R)]=ImG[i][j]
        A[i+len(R)][j]=ImG[i][j]
        A[i+len(R)][j+len(R)]=-ReG[i][j]
  return A

def Foldy_Lax_QUBO_Real(R,g1,k):
  ReG=np.zeros([len(R),len(R)])
  for i in range(len(R)):
    for j in range(len(R)):
      if i==j:
        ReG[i][j]=1
      else:
        ReG[i][j]=g1[j]*scipy.special.spherical_jn(0,k*np.linalg.norm(R[i]-R[j]),derivative=False)
  return ReG

def Foldy_Lax_QUBO_Imaginary(R,g2,k):
  ImG=np.zeros([len(R),len(R)])
  for i in range(len(R)):
    for j in range(len(R)):
      if i==j:
        ImG[i][j]=0
      else:
        ImG[i][j]=g2[j]*scipy.special.spherical_yn(0,k*np.linalg.norm(R[i]-R[j]),derivative=False)
  return ImG

def QUBO_solution(A,b,N1,N2):
  QQ=QUBO_Matrix(A,b,N1,N2)
  q=QQ[0]
  sampleset=dimod.as_samples(SimulatedAnnealingSampler().sample_qubo(QQ[1]))
  u=np.zeros(len(b))
  for i in range(len(b)):
    u[i]=np.dot(sampleset[0],q[i])
  return u
#Computes the vector evaluating the incident field at locations r=[r_1,...,r_N].
#k is the wavenumber of the medium.
#type can be either 'plane' and in that case a is the vector dictating the 
#direction of the plane wave while for type 'spherical' a is the location 
#of the point source emanating the spherical wave.
def Incident(r,a,k,type):
  b=np.zeros(2*len(r))
  if type=='plane':
    for i in range(len(r)):
      b[i]=math.cos(k*np.dot(r[i],a))
      b[i+len(r)]=math.sin(k*np.dot(r[i],a))
  if type=='spherical':
    for i in range(len(r)):
      b[i]=scipy.special.spherical_jn(0,k*np.linalg.norm(a-r[i]),derivative=False)
      b[i+len(r)]=scipy.special.spherical_nn(0,k*np.linalg.norm(a-r[i]),derivative=False)
  return b
#Creates locations. For type 'random' creates a random set of scatterers within
#a cube of size 2*distance.
#For type 'sphere' creates a random set of scatterers within a sphere of radius distance.
def Scatterers(number,distance,type):
  r=np.zeros([int(number),3])
  if type=='random':
    for i in range(int(number)):
      r[i][0]=np.random.uniform(-distance,distance)
      r[i][1]=np.random.uniform(-distance,distance)
      r[i][2]=np.random.uniform(-distance,distance)
  if type=='sphere':
    for i in range(int(number)):
      r[i][0]=distance*np.cos(2*np.pi*np.random.uniform(0,1))*np.sin(np.pi*np.random.uniform(0,1))
      r[i][1]=distance*np.sin(2*np.pi*np.random.uniform(0,1))*np.sin(np.pi*np.random.uniform(0,1))
      r[i][2]=distance*np.cos(np.pi*np.random.uniform(0,1))
  return r
#Creates a vector with the scattering coefficients for the problem.
#If type=='random', then alpha is the maximum value of the coefficients and
#the routine returns a real-valued vector.
#If type=='spheres', then alpha is the electric radius k*a, with k the medium's
#wavenumber and a the sphere's radius.
#The routine returns a numberx2 matrix with the first column having the
#real part and the second column having the imaginary part of the coefficients.
def Scattering_Coefficients(number,type,alpha):
  if type=='random':
    g=np.zeros(int(number))
    for i in range(int(number)):
      g[i]=np.random.uniform(0,alpha)
  if type=='spheres':
    g=np.zeros([int(number),2])
    for i in range(int(number)):
      g[i]=np.array([0.5*(np.cos(2*alpha)-1),-0.5*(np.sin(2*alpha)+1)])
  return g

#Returns the QUBO matrices A_R, A_I and M/A_R
def Reduced_QUBO (R,g1,g2,k):
  ReG=Foldy_Lax_QUBO_Real(R,g1,k)
  ImG=Foldy_Lax_QUBO_Imaginary(R,g2,k)
  Msyn=ReG+np.matmul(np.matmul(ImG,np.linalg.inv(ReG)),ImG)#The Schur complement, M/A_R
  return Msyn,ReG,ImG
#Returns the known vectors associated to the QUBO formulation \tilde{u}_R and \tilde{u}_I
def Reduced_QUBO_vector(X,Y,b):
  Reb=np.zeros([int(len(b)/2)])
  Imb=np.zeros([int(len(b)/2)])
  for i in range(int(len(b)/2)):
    Reb[i]=b[i]
    Imb[i]=b[i+int(len(b)/2)]
  Rb=Reb+np.dot(np.matmul(Y,np.linalg.inv(X)),Imb)
  Ib=np.dot(np.matmul(Y,np.linalg.inv(X)),Reb)-Imb
  return Rb,Ib
#Returns the QUBO solution by means of the reduced system.
#number is the number of scatterers, type1 refers to the cluster's topology
#distance refers to the qube withing the cluster lies.
#a refers to the source of incidence: point-source location (spherical wave incidence)
#or direction of propagation (plane wave incidence)
#k is the wavenumber of the medium that contains the source of incidence
#type2 refers to the type of incidence (plane or spherical)
#R refers to the radius of the spheres composing the cluster or the interval which the random stregnths are assigned
#type3 refers to the type of the assigned strengths: random, energy-related, volume-related, etc
#N1 refers to the number of integer qubits, N2 refers to the number of fractional qubits
#Returns: the quantum-obtained solution, the exact solution, the error between the quantum and the classical solution
#the locations of the scatterers and the vector containing the evaluations of the incident field on the scatterers' locations.
def Quantum_Foldy_Lax_solution(number,distance,type1,a,k,type2,R,type3,N1,N2):
  r=Scatterers(number,distance,type1)
  b=Incident(r,a,k,type2)
  g1=np.transpose(Scattering_Coefficients(len(b),type3,k*R))[0]
  g2=np.transpose(Scattering_Coefficients(len(b),type3,k*R))[1]
  A=Foldy_Lax_QUBO(r,g1,g2,k)
  Msyn=Reduced_QUBO(r,g1,g2,k)[0]
  ReG=Reduced_QUBO(r,g1,g2,k)[1]
  ImG=Reduced_QUBO(r,g1,g2,k)[2]
  Rb=Reduced_QUBO_vector(ReG,ImG,b)[0]
  Ib=Reduced_QUBO_vector(ReG,ImG,b)[1]
  Rex=QUBO_solution(Msyn,Rb,N1,N2)
  Imx=QUBO_solution(Msyn,Ib,N1,N2)
  x=np.linalg.solve(A,b)
  return np.array([Rex[i]+1j*Imx[i] for i in range(len(Rex))]), np.array([x[i]+1j*x[i+int(len(x)/2)] for i in range(int(len(x)/2))]), np.array([abs(Rex[i]-x[i])+abs(Imx[i]-x[i+int(len(x)/2)]) for i in range(int(len(x)/2))]), np.array([abs(Rex[i]-x[i]) for i in range(int(len(x)/2))]),np.array([abs(Imx[i]-x[i+int(len(x)/2)]) for i in range(int(len(x)/2))]),r,b

