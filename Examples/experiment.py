# Authors: Ari Herman & Taiyo Terada

# Displays counter example to lemma 3 from Olfati paper

import sys
sys.path.insert(0,"../ClassDefinitions")
import numpy as np
from FlockParameters import SimulationParams, AnimationParams
from FlockSimulation import OlfatiFlockingSimulation
from FlockAnimation import ScatterAnimation
from matplotlib import pyplot as plt

# Set simulation parameters
a = 5
b= 5
c = 0
r_a = 0.346786940882
d_a = 0.242070103255
eps = 0.1
h = 0.2
dt = 0.01
num_iters = 1000 #2500
num_boids = 600 #550
dim = 3
c_p = 5
c_q = 10
fname = "experiment_array"

# Set animation parameters 
ani_params = AnimationParams()
ani_params.set_show(True)
ani_params.set_save(False)
ani_params.set_quiver(False)

# Functions
def sig_norm(d): # Sigma norm
    return (np.sqrt(1+eps*d**2)-1)/eps

def l2_norm(z):
    return np.sqrt(np.sum(z**2,axis=2,keepdims=True))

def sig_grad(d): # Gradient of sigma norm
    return d/(np.sqrt(1+eps*d**2))
    
def rho_h(z):
    return  np.logical_and(z>=0,z<h)+np.logical_and(z<=1,z>=h)*(0.5*(1+np.cos(np.pi*(z-h)/(1-h))))

def phi_a(z):
    return 0.5*rho_h(z/r_a)*((a+b)*sig_grad(z-d_a+c)+(a-b))

def differences(q): # Returns array of pairwise differences 
    return q[:,None,:]-q

def f(dist):
    norm = sig_norm(dist)
    dq = phi_a(norm)/(1+eps*norm)
    dp = rho_h(norm/r_a)
    return dq,dp

def uUpdate(q,p):
    diff=differences(q)
    diffp=differences(p)
    dist = l2_norm(diff)
    dq,dp = f(dist)
    return np.sum(diff*dq,axis=0) + np.sum(dp*diffp,axis=0)
    
# Gamma agent
x=np.cos(np.linspace(0,np.pi*num_iters*dt,num_iters))
y = np.zeros(num_iters)
z = np.zeros(num_iters)
dx=-np.pi*np.sin(np.linspace(0,np.pi*num_iters*dt,num_iters))
dy = np.zeros(num_iters)
dz = np.zeros(num_iters)
q_g=np.stack((x,y,y),axis=1)
p_g = np.stack((dx,dy,dz),axis=1)

# Init
q = np.random.normal(0.0,1.0,size=(num_boids,dim))
p = np.random.normal(0.0,1.0,size=(num_boids,dim))

# Main
X = np.zeros((num_iters,num_boids,dim))
V = np.zeros((num_iters,num_boids,dim))
for i in range(num_iters):
    z = uUpdate(q,p)
    q+=p*dt
    p+=(z-c_q*(q-q_g[i])-c_p*(p-p_g[i]))*dt
    X[i,:,:] = q
    V[i,:,:] = p

# Add the gamma agent
X = np.concatenate((X,q_g[:,None,:]),axis=1) 
V = np.concatenate((V,p_g[:,None,:]),axis=1)

# Save array
np.save(fname,[X,V])


# Animation
flock = ScatterAnimation(ran = 2.0)
flock.params = ani_params
flock.setQ(X)
flock.setP(V)
flock.initAnimation()
flock.animate()

