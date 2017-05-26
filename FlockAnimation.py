# -*- coding: utf-8 -*-
"""
Created on Tue May 16 21:46:44 2017

@author: Ari
"""

import numpy as np
from sklearn.preprocessing import normalize
from numpy.linalg import norm
from AnimateFunc import ScatterAnimation 
from AnimateFunc import QuiverAnimation 
from UserInput import SimulationParams, AnimationParams
from FlockFuncs import OlfatiFlockingSimulation

#####################################
# Get simulation parameters from user
#####################################

sim_params = SimulationParams()
sim_params.getUserInput()

####################################
# Get animation parameters from user
####################################

ani_params = AnimationParams()
ani_params.getUserInput()

###########################
# Setup flocking simulation
###########################

flock_sim = OlfatiFlockingSimulation()

# Set simulation parameters
flock_sim.eps,flock_sim.num_boids,flock_sim.a,flock_sim.b,flock_sim.c,flock_sim.h,flock_sim.r_a,flock_sim.d_a,flock_sim.dt,flock_sim.num_iters,flock_sim.gamma_path,flock_sim.dim,flock_sim.c_q,flock_sim.c_p = sim_params.eps,sim_params.num_boids,sim_params.a,sim_params.b,sim_params.c,sim_params.h,sim_params.r_a,sim_params.d_a,sim_params.dt,sim_params.num_iters,sim_params.gamma_path,sim_params.dim,sim_params.c_q,sim_params.c_p

################
# Run simulation
################

# Init simulation
flock_sim.initSim()

# Run simulation
X,V = flock_sim.runSim()

# Save simulation array?
save_array = input("Do you want to save the simulation array? [y/n]: ") != 'n'
if save_array:
    np.save(ani_params.fname,X)


#########
# Animate
#########

if ani_params.quiver:
    flock = QuiverAnimation(X,0.01*V/norm(V,axis=2,keepdims=True))
    flock.animate(show=ani_params.show,save=ani_params.save,fname=ani_params.fname)
else:
    flock = ScatterAnimation(X)
    flock.animate(show=ani_params.show,save=ani_params.save,fname=ani_params.fname)


