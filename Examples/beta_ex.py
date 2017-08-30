# Beta agent example with sphere obstacle

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

sim_params = SimulationParams()
sim_params.set_dim(3)
sim_params.c_p = 5
sim_params.c_q = 10
sim_params.set_num_boids(500)
sim_params.set_gamma_path('wild')
sim_params.set_d(0.7) #7.0 # 0.8
sim_params.set_r(1.2*sim_params.d)
sim_params.set_num_iters(300)
sim_params.get_save()

# Init points
sim_params.set_q_init('random')
sim_params.set_p_init('random')

####################################
# Get animation parameters from user
####################################

ani_params = AnimationParams()
ani_params.set_show(True)
ani_params.get_save()
ani_params.get_quiver()

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
X,V = flock_sim.runSim(beta=True)


###########
# Animation
###########

flock = ScatterAnimation()
flock.params = ani_params
flock.setQ(X)
flock.setP(V)
flock.initAnimation()
flock.animate()
