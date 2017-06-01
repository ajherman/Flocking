# Authors: Ari Herman & Taiyo Terada

# Displays counter example to lemma 3 from Olfati paper

import sys
sys.path.insert(0,"../ClassDefinitions")
import numpy as np
from FlockParameters import SimulationParams, AnimationParams
from FlockSimulation import OlfatiFlockingSimulation
from FlockAnimation import ScatterAnimation,QuiverAnimation
from matplotlib import pyplot as plt

###########################
# Set simulation parameters
###########################

sim_params = SimulationParams()
sim_params.set_dim(3)
sim_params.c_p = 5
sim_params.c_q = 10
sim_params.set_num_boids(550)
sim_params.set_gamma_path('wild')
sim_params.set_d(0.7) #7.0 # 0.8
sim_params.set_r(1.2*sim_params.d)
#sim_params.d_a = (np.sqrt(1+sim_params.eps*sim_params.d**2)-1)/sim_params.eps
#sim_params.r_a = (np.sqrt(1+sim_params.eps*sim_params.r**2)-1)/sim_params.eps
sim_params.set_num_iters(2500)

# Init points

#sim_params.q_init = np.random.normal(0.0,1.0,size=(sim_params.num_boids,sim_params.dim))
#sim_params.p_init = np.random.normal(0.0,0.1,size=(sim_params.num_boids,sim_params.dim))
sim_params.set_q_init('random')
sim_params.set_p_init('random')

####################################
# Get animation parameters from user
####################################

ani_params = AnimationParams()
ani_params.getUserInput()

##################
# Setup simulation
##################

# Make simulation object
flock_sim = OlfatiFlockingSimulation()

# Set simulation parameters
flock_sim.params = sim_params

################
# Run simulation
################

# Init sim
flock_sim.initSim()

# Run sim
X,V = flock_sim.runSim()

# Save simulation array
save_array = input("Do you want to save the simmulation array? [y/n]: ")
if save_array == 'y':
    np.save(ani_params.fname,X)

###########
# Animation
###########

if ani_params.quiver:
    flock = QuiverAnimation(X[:,:-1,:],V[:,:-1,:])
    
else:
    flock = ScatterAnimation(X[:,:-1,:])

flock.animate(show=ani_params.show,save=ani_params.save,fname=ani_params.fname)


