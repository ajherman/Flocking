# Authors: Ari Herman & Taiyo Terada

# Displays counter example to lemma 3 from Olfati paper

import sys
sys.path.insert(0,"../ClassDefinitions")
import numpy as np
from FlockParameters import SimulationParams, AnimationParams
from FlockSimulation import OlfatiFlockingSimulation
from FlockAnimation import ScatterAnimation
from matplotlib import pyplot as plt

###########################
# Set simulation parameters
###########################

a = 5
b= 5
c = 0
r_a = 0.346786940882
d_a = 0.242070103255
eps = 0.1
h = 0.2
dt = 0.01
num_iters = 2500
num_boids = 550
dim = 3
c_p = 5
c_q = 10
fname = "experiment_array"

q_init = np.random.normal(0.0,1.0,size=(num_boids,dim))
p_init = np.random.normal(0.0,1.0,size=(num_boids,dim))


####################################
# Get animation parameters from user
####################################

ani_params = AnimationParams()
ani_params.set_show(True)
ani_params.get_save()
ani_params.get_quiver()


def sig_norm(z): # Sigma norm
    return (np.sqrt(1+eps*np.sum(z**2,axis=2,keepdims=True))-1)/eps
     
def sig_grad(z,norm=None): # Gradient of sigma norm
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


def differences(q): # Returns array of pairwise differences 
    return q[:,None,:]-q

def uUpdate(q,p):
        diff=differences(q)
        norms = sig_norm(diff)
        diffp=differences(p)
        return np.sum(phi_a(norms)*sig_grad(diff,norms),axis=0) + np.sum(rho_h(norms/r_a)*diffp,axis=0)
    
    
def differentiate(v): # Differentiates vector
    dv = v.copy()
    dv[1:]-=v[:-1]
    return dv/dt

# Gamma agent
x=np.cos(np.linspace(0,2*np.pi*num_iters/200.,num_iters))
y=np.cos(np.linspace(0,4*np.pi*num_iters/200.,num_iters))
z=np.sin(np.linspace(0,8*np.pi*num_iters/200.,num_iters))

q_g=np.stack((x,y,y),axis=1)
p_g=np.stack((differentiate(x),differentiate(y),differentiate(z)),axis=1)

# Init
q=q_init 
p=p_init 

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

np.save(fname,[X,V])

###########
# Animation
###########

flock = ScatterAnimation(ran = 2.0)
flock.params = ani_params
flock.setQ(X)
flock.setP(V)
flock.initAnimation()
flock.animate()

