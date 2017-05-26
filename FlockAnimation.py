# -*- coding: utf-8 -*-
"""
Created on Tue May 16 21:46:44 2017

@author: Ari
"""

import numpy as np
from sklearn.preprocessing import normalize
from numpy.linalg import norm
from AnimateFunc import ScatterAnimation as SA
from AnimateFunc import QuiverAnimation as QA
from UserInput import SimulationParams, AnimationParams
from FlockFuncs import OlfatiFlockingSimulation

#####################################
# Get simulation parameters from user
#####################################

sim_params = SimulationParams()
sim_params.getUserInput()

#eps,d,r,d_a,r_a,a,b,c,h,dt,c_q,c_p,num_boids,num_iters,dim = sim_params.eps,sim_params.d,sim_params.r,sim_params.d_a,sim_params.r_a,sim_params.a,sim_params.b,sim_params.c,sim_params.h,sim_params.dt,sim_params.c_q,sim_params.c_p,sim_params.num_boids,sim_params.num_iters,sim_params.dim
#

####################################
# Get animation parameters from user
####################################

ani_params = AnimationParams()
ani_params.getUserInput()

###########################
# Setup flocking simulation
###########################

flock_sim = OlfatiFlockingSimulation()

flock_sim.eps,flock_sim.num_boids,flock_sim.a,flock_sim.b,flock_sim.c,flock_sim.h,flock_sim.r_a,flock_sim.d_a,flock_sim.dt = sim_params.eps,sim_params.num_boids,sim_params.a,sim_params.b,sim_params.c,sim_params.h,sim_params.r_a,sim_params.d_a,sim_params.dt

#####################
# Generate trajectory
#####################
if sim_params.dim == 2:
    # Pick a path for gamme agent
    gamma_path = input("Select path for gamma agent ['circle','eight']: ")
    
    if gamma_path == "circle":
        x=np.cos(np.linspace(0,2*np.pi,sim_params.num_iters))
        y=np.sin(np.linspace(0,2*np.pi,sim_params.num_iters))

    elif gamma_path == "eight":
        x=np.cos(np.linspace(0,2*np.pi,sim_params.num_iters))
        y=np.sin(np.linspace(0,4*np.pi,sim_params.num_iters))
    
    else:
        assert(False)

    q_g=np.stack((x,y),axis=1) 
    p_g=np.stack((flock_sim.differentiate(x),flock_sim.differentiate(y)),axis=1)

if sim_params.dim ==3:
    # Pick a path for gamma agent
    gamma_path = input("Select path for gamma agent ['circle','wild']: ")
    
    if gamma_path == "circle":    
        # Gamma agent (moves in a circle)
        x=np.cos(np.linspace(0,2*np.pi,sim_params.num_iters))
        y=np.sin(np.linspace(0,2*np.pi,sim_params.num_iters))
        z=np.zeros(sim_params.num_iters)
    
    elif gamma_path == "wild":
        x=np.cos(np.linspace(0,2*np.pi,sim_params.num_iters))
        y=np.cos(np.linspace(0,4*np.pi,sim_params.num_iters))
        z=np.sin(np.linspace(0,8*np.pi,sim_params.num_iters))
    
    else:
        assert(False)

    q_g=np.stack((x,y,y),axis=1)
    p_g=np.stack((flock_sim.differentiate(x),flock_sim.differentiate(y),flock_sim.differentiate(z)),axis=1)


################
# Run simulation
################

# Random init boids 
q=np.random.normal(0.0,1.0,size=(sim_params.num_boids,sim_params.dim))
p=0.01*np.random.rand(sim_params.num_boids,sim_params.dim)

# Run simulation
X = np.zeros((sim_params.num_iters,sim_params.num_boids,sim_params.dim))
V = np.zeros((sim_params.num_iters,sim_params.num_boids,sim_params.dim))
for i in range(sim_params.num_iters):
    z=flock_sim.uUpdate(q,p)
    q+=p*sim_params.dt
    X[i,:,:] = q
    V[i,:,:] = p
    p+=(z-sim_params.c_q*(q-q_g[i])-sim_params.c_p*(p-p_g[i]))*sim_params.dt

# Add the gamma agent
X = np.concatenate((X,q_g[:,None,:]),axis=1) 
V = np.concatenate((V,p_g[:,None,:]),axis=1)

#########
# Animate
#########

if ani_params.save:
    np.save(ani_params.fname,X)

if ani_params.show:
    if ani_params.quiver:
        norm_V = 0.01*V/norm(V,axis=2,keepdims=True)
        flock = QA(X,norm_V)
        flock.animate(fname=ani_params.fname,show=ani_params.show)
    else:
        flock = SA(X)
        flock.animate(fname=ani_params.fname,show=ani_params.show)


