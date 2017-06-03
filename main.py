# Authors: Ari Herman & Taiyo Terada

# Prompts user for input and runs flocking simulation using Algorithm 2 from Olfati paper

import sys
sys.path.insert(0,"ClassDefinitions")
import numpy as np
from FlockAnimation import ScatterAnimation
from FlockParameters import SimulationParams, AnimationParams
from FlockSimulation import OlfatiFlockingSimulation, OlfatiFlockingSimulationTF


#####################################
# Get parameters from user
#####################################

# Simulation
sim_params = SimulationParams()
sim_params.get_num_boids()
sim_params.get_num_iters()
sim_params.get_dim()
sim_params.get_gamma_path()
sim_params.get_save()
sim_params.set_q_init('random')
sim_params.set_p_init('random')
run_method = input("Do you want to run this simulation with Numpy or Tensorflow? ['NP'/'TF']: ")

# Animation
ani_params = AnimationParams()
ani_params.get_save()
ani_params.get_show()
ani_params.get_quiver()


##################
# Flock simulation
##################

# Setup
if run_method == 'NP':
    flock_sim = OlfatiFlockingSimulation()

elif run_method == 'TF':
    flock_sim = OlfatiFlockingSimulationTF()

else:
    print("Invalid run method.  Must select Numpy or Tensorflow. ['NP'/'TF']")
    assert(False)

# Set parameters
flock_sim.params = sim_params

# Init 
flock_sim.initSim()

# Run 
X,V = flock_sim.runSim(beta=True)


#################
# Flock animation
#################

# Setup
flock = ScatterAnimation()
flock.params = ani_params
flock.setQ(X)
flock.setP(V)

# Init
flock.initAnimation()

# Run
flock.animate()


