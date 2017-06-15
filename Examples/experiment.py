# Authors: Ari Herman & Taiyo Terada

# Simplified version of exploding behavior

import sys
sys.path.insert(0,"../ClassDefinitions")
import numpy as np
from FlockParameters import SimulationParams, AnimationParams
from FlockSimulation import OlfatiFlockingSimulation
from FlockAnimation import ScatterAnimation
from matplotlib import pyplot as plt
from numpy.linalg import norm as l2_norm

# Set simulation parameters
a = 10
b = 15 #20
c = 1 #0.2
d = 1
e = 5 #1

dt = 0.01
fps = 30
num_iters = 500
num_boids = 250 #400
dim = 3
fname = "experiment_array"

# Set animation parameters
ani_params = AnimationParams()
ani_params.set_fps(fps)
ani_params.set_show(True)
ani_params.set_save(False)
ani_params.set_quiver(True)

###########
# Functions
###########

<<<<<<< HEAD
def sig_norm(d):
    return (np.sqrt(1+eps*d**2)-1)/eps

def sig_grad(d):
    return d/(np.sqrt(1+eps*d**2))

def rho_h(d):
    return np.logical_and(d>=0,d<h)+np.logical_and(d<=1,d>=h)*(0.5*(1+np.cos(np.pi*(d-h)/(1-h))))

def phi_a(d):
    return 0.5*rho_h(d/r_a)*((a+b)*sig_grad(d-d_a+c)+(a-b))

def f(dist):
    norm = sig_norm(dist)

    dq = phi_a(norm)/(1+eps*norm) + c_q/num_boids
    dp = rho_h(norm/r_a) + c_p/num_boids

=======
def f(dist):
    dp = 1.-1./(1+np.exp(b*(d-dist)))
    dq = (-c+e/(1+np.exp(a*(d-dist))))*dp
>>>>>>> simple
    return dq,dp

def uUpdate(q,p,i):
    diff = q[:,None,:] - q
    diffp = p[:,None,:] - p
    dist = l2_norm(diff,axis=2,keepdims=True)
    dq,dp = f(dist)
    return np.sum(diff*dq+dp*diffp,axis=0)

# Plot inter-agent forces
X = np.linspace(0,2.5,100)
Y,Z = f(X)
plt.plot(X,Y)
plt.plot(X,Z)
plt.show()

###########
# Animation
###########

# Init
q = np.random.normal(0.0,0.8,size=(num_boids,dim))
p = np.random.normal(0.0,0.5,size=(num_boids,dim))

# Main
X = np.zeros((num_iters,num_boids,dim))
V = np.zeros((num_iters,num_boids,dim))
for i in range(num_iters):
    q , p = q+p*dt , p + uUpdate(q,p,i)*dt
    X[i,:,:] = q
    V[i,:,:] = p

# Save array
#np.save(fname,[X,V])

# Animation
flock = ScatterAnimation(ran = 2.0)
flock.params = ani_params
flock.setQ(X)
flock.setP(V)
flock.initAnimation()
flock.animate()
