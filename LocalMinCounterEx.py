# Authors: Ari Herman & Taiyo Terada

# Displays counter example to lemma 3 from Olfati paper

import numpy as np
from scipy.linalg import norm
from FlockParameters import SimulationParams, AnimationParams
from FlockSimulation import OlfatiFlockingSimulation
from FlockAnimation import ScatterAnimation,QuiverAnimation

###########################
# Set simulation parameters
###########################

sim_params = SimulationParams()
sim_params.dim = 2
sim_params.c_p = 0
sim_params.c_q = 0
sim_params.num_boids = 4
sim_params.gamma_path = 'circle'
sim_params.num_iters = 5000

# Init points
d=sim_params.d
sim_params.q_init = np.array([[0,0],[0,d*np.sqrt(3)/3],[-d/2,-d*np.sqrt(3)/6],[d/2,-d*np.sqrt(3)/6]])
sim_params.p_init = np.zeros((sim_params.num_boids,2))

# Add noise
sim_params.q_init += np.random.normal(0.0,0.001,size=(sim_params.num_boids,sim_params.dim))
sim_params.p_init += np.random.normal(0.0,0.001,size=(sim_params.num_boids,sim_params.dim))

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
print(np.shape(X))
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



