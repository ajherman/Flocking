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

params = SimulationParams()
params.set_dim(3)
params.c_p = 5
params.c_q = 10
params.set_num_boids(550)
params.set_gamma_path('wild')
params.set_d(0.7) #7.0 # 0.8
params.set_r(1.2*params.d)
params.set_num_iters(2500)
params.get_save()
# Init points
params.set_q_init('random')
params.set_p_init('random')

####################################
# Get animation parameters from user
####################################

ani_params = AnimationParams()
ani_params.set_show(True)
ani_params.get_save()
ani_params.get_quiver()


def sig_norm(z): # Sigma norm
    return (np.sqrt(1+params.eps*np.sum(z**2,axis=2,keepdims=True))-1)/params.eps
     
def sig_grad(z,norm=None): # Gradient of sigma norm
    if type(norm) == "NoneType":
        return z/(1+params.eps*sig_norm(z))
    else:
        return z/(1+params.eps*norm)
    
def rho_h(z):
    return  np.logical_and(z>=0,z<params.h)+np.logical_and(z<=1,z>=params.h)*(0.5*(1+np.cos(np.pi*(z-params.h)/(1-params.h))))

def phi(z):
    return 0.5*((params.a+params.b)*sig_grad(z+params.c,1)+(params.a-params.b))

def phi_a(z):
    return rho_h(z/params.r_a)*phi(z-params.d_a)


def differences(q,b=None): # Returns array of pairwise differences 
    if b is None:
        return q[:,None,:] - q
    else:
        return q[:,None,:]-b

def uUpdate(q,p):
        diff=differences(q)
        norms = sig_norm(diff)
        diffp=differences(p)
        return params.c_qa*np.sum(phi_a(norms)*sig_grad(diff,norms),axis=0)+params.c_pa*np.sum(rho_h(norms/params.r_a)*diffp,axis=0)
    
    
def differentiate(v): # Differentiates vector
    dv = v.copy()
    dv[1:]-=v[:-1]
    return dv/params.dt
    
x=np.cos(np.linspace(0,2*np.pi*params.num_iters/200.,params.num_iters))
y=np.cos(np.linspace(0,4*np.pi*params.num_iters/200.,params.num_iters))
z=np.sin(np.linspace(0,8*np.pi*params.num_iters/200.,params.num_iters))

q_g=np.stack((x,y,y),axis=1)
p_g=np.stack((differentiate(x),differentiate(y),differentiate(z)),axis=1)

q=params.q_init 
p=params.p_init 

X = np.zeros((params.num_iters,params.num_boids,params.dim))
V = np.zeros((params.num_iters,params.num_boids,params.dim))
for i in range(params.num_iters):
    z = uUpdate(q,p)
    q+=p*params.dt
    p+=(z-params.c_q*(q-q_g[i])-params.c_p*(p-p_g[i]))*params.dt
    X[i,:,:] = q
    V[i,:,:] = p

# Add the gamma agent
X = np.concatenate((X,q_g[:,None,:]),axis=1) 
V = np.concatenate((V,p_g[:,None,:]),axis=1)

# Save array
if params.save:
    np.save(self.params.fname,[X,V])

###########
# Animation
###########

flock = ScatterAnimation()
flock.params = ani_params
flock.setQ(X)
flock.setP(V)
flock.initAnimation()
flock.animate()

