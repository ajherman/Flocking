# Make animation from array


# Authors: Ari Herman & Taiyo Terada

# Prompts user for input and runs flocking simulation using Algorithm 2 from Olfati paper

import sys
sys.path.insert(0,"ClassDefinitions")
import numpy as np
from FlockAnimation import ScatterAnimation
from FlockParameters import AnimationParams


#####################################
# Get parameters from user
#####################################
options=["\\Users\\tster\\Dropbox\\Flocking\\example_arrays\\circle3darray100.npy"]
print("Choose a number to run an animation \n 0: 3d, circle, 100 boids, beta at origin")
n = int(input("Enter one of the numbers above: "))
if n not in range(len(options)):
    assert False
f=options[n]

X,V = np.load(f)
num_iter,num_boids,dim = np.shape(X)

# Animation
ani_params = AnimationParams()
ani_params.get_save()
ani_params.get_show()
ani_params.get_quiver()

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



