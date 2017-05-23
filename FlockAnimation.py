# -*- coding: utf-8 -*-
"""
Created on Tue May 16 21:46:44 2017

@author: Ari
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from scipy.spatial.distance import pdist,squareform
import matplotlib
from AnimateFunc import ScatterAnimation as SA

############
# Parameters
############

eps = 0.1
d=0.7 #25
r=1.2*d
d_a = (np.sqrt(1+eps*d**2)-1)/eps
r_a = (np.sqrt(1+eps*r**2)-1)/eps
a=5
b=5
c=.5*np.abs(a-b)/np.sqrt(a*b)
h=.2 #0<h<1
dt=0.01
c_q=10
c_p=5

# Get parameters from user
num_boids = input("Enter number of boids: ")
num_iters = input("Enter number of iterations: ")
dim = input("Enter number of dimensions [2/3]: ")
num_boids,num_iters,dim = int(num_boids),int(num_iters),int(dim)
save = input("Do you want to save this animation [y/n]: ")
if save=='y':
    fname = input("Type file name [no extension]: ")
else:
    fname = None

##################
# Useful functions
##################

def sig_norm(z): # Sigma norm
    return (np.sqrt(1+eps*np.sum(z**2,axis=2).reshape((num_boids,num_boids,1)))-1)/eps

def sig_grad(z,norm=None,eps=eps): # Gradient of sigma norm
    if type(norm) == "NoneType":
        return z/(1+eps*sig_norm(z))
    else:
        return z/(1+eps*norm)
    
def rho_h(z):
    return  np.logical_and(z>=0,z<h)+np.logical_and(z<=1,z>=h)*(0.5*(1+np.cos(np.pi*(z-h)/(1-h))))

def phi(z):
    return 0.5*((a+b)*sig_grad(z+c,1)+(a-b))

def phi_a(z):
    return rho_h(z/r_a)*phi(z-d_a)
    
def differences(q):
    return q[:,None,:] - q

def uUpdate(q,p):
    diff=differences(q)
    norms = sig_norm(diff)
    diffp=differences(p)
    return np.sum(phi_a(norms)*sig_grad(diff,norms),axis=0)+np.sum(rho_h(norms/r_a)*diffp,axis=0)

def differentiate(v):
    dv = v.copy()
    dv[1:]-=v[:-1]
    return dv/dt

#####################
# Generate trajectory
#####################
if dim == 2:
    # Pick a path for gamme agent
    gamma_path = input("Select path for gamma agent ['circle','eight']: ")
    
    if gamma_path == "circle":
        x=np.cos(np.linspace(0,2*np.pi,num_iters))
        y=np.sin(np.linspace(0,2*np.pi,num_iters))

    elif gamma_path == "eight":
        x=np.cos(np.linspace(0,2*np.pi,num_iters))
        y=np.sin(np.linspace(0,4*np.pi,num_iters))
    
    else:
        assert(False)

    q_g=np.stack((x,y),axis=1) 
    p_g=np.stack((differentiate(x),differentiate(y)),axis=1)

if dim ==3:
    # Pick a path for gamma agent
    gamma_path = input("Select path for gamma agent ['circle','wild']: ")
    
    if gamma_path == "circle":    
        # Gamma agent (moves in a circle)
        x=np.cos(np.linspace(0,2*np.pi,num_iters))
        y=np.sin(np.linspace(0,2*np.pi,num_iters))
        z=np.zeros(num_iters)
    
    elif gamma_path == "wild":
        x=np.cos(np.linspace(0,2*np.pi,num_iters))
        y=np.cos(np.linspace(0,4*np.pi,num_iters))
        z=np.sin(np.linspace(0,8*np.pi,num_iters))
    
    else:
        assert(False)

    q_g=np.stack((x,y,y),axis=1)
    p_g=np.stack((differentiate(x),differentiate(y),differentiate(z)),axis=1)


################
# Run simulation
################

# Random init boids 
q=np.random.normal(0.0,1.0,size=(num_boids,dim))
p=0.01*np.random.rand(num_boids,dim)

# Run simulation
X = np.zeros((num_iters,num_boids,dim))
for i in range(num_iters):
    z=uUpdate(q,p)
    q+=p*dt
    X[i,:,:] = q
    p+=(z-c_q*(q-q_g[i])-c_p*(p-p_g[i]))*dt

# Add the gamma agent
X = np.concatenate((X,q_g[:,None,:]),axis=1) 


#########
# Animate
#########

flock = SA(X)
flock.animate(fname=fname)


