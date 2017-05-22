# -*- coding: utf-8 -*-
"""
Created on Tue May 16 21:46:44 2017

@author: Ari
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from scipy.spatial.distance import pdist,squareform
import matplotlib
############
# Parameters
############

eps = 0.1
d=0.7 #25
r=1.2*d
d_a = (np.sqrt(1+eps*d**2)-1)/eps
r_a = (np.sqrt(1+eps*r**2)-1)/eps
a=5
b=5
c=.5*np.abs(a-b)/np.sqrt(a*b)
h=.2 #0<h<1
dt=0.01
c_q=10
c_p=5

# Get parameters from user
num_boids = input("Enter number of boids: ")
num_iters = input("Enter number of iterations: ")
dim = input("Enter number of dimensions [2/3]: ")
num_boids,num_iters,dim = int(num_boids),int(num_iters),int(dim)
save = input("Do you want to save this animation [y/n]: ")
if save=='y':
    fname = input("Type file name [no extension]: ")


#####################
# Useful tf functions
#####################

def sig_norm(z):
    return tf.reshape( (tf.sqrt(1+eps*tf.reduce_sum(z**2,axis=2))-1)/eps , (num_boids,num_boids,1))

def sig_grad(z,norm=None,eps=eps):
    if norm is None:
        return z/(1+eps*sig_norm(z))
    else:
        return z/(1+eps*norm)

def rho_h(z):
    return tf.to_float(tf.logical_and(z>=0,z<h))+tf.to_float(tf.logical_and(z<=1,z>=h))*(0.5*(1+tf.cos(np.pi*(z-h)/(1-h))))

def phi(z):
    return 0.5*((a+b)*sig_grad(z+c,1)+(a-b))

def phi_a(z):
    return rho_h(z/r_a)*phi(z-d_a)
    
def differences(q):
    return q[:,None,:] - q

def uUpdate(q,p):
    diff=differences(q)
    norms = sig_norm(diff)
    diffp=differences(p)
    return tf.reduce_sum(phi_a(norms)*sig_grad(diff,norms),axis=0)+tf.reduce_sum(rho_h(norms/r_a)*diffp,axis=0)

def differentiate(v):
    dv = tf.identity(v)
    dv = dv[1:] - v[:-1]
    zeros = tf.zeros((dv[:1]).get_shape())
    dv = tf.concat(0,[zeros,dv])  # I think this the argument order is reversed in newer tf
    return dv/dt


#####################
# Generate trajectory
#####################
if dim == 2:
    # Pick a path for gamme agent
    gamma_path = input("Select path for gamma agent ['circle','eight']: ")
    
    if gamma_path == "circle":
        x=tf.cast(tf.cos(np.linspace(0,2*np.pi,num_iters)),tf.float32)
        y=tf.cast(tf.sin(np.linspace(0,2*np.pi,num_iters)),tf.float32)
    elif gamma_path == "eight":
        x=tf.cast(tf.cos(np.linspace(0,2*np.pi,num_iters)),tf.float32)
        y=tf.cast(tf.sin(np.linspace(0,4*np.pi,num_iters)),tf.float32)
    
    else:
        assert(False)

    q_g=tf.stack((x,y),axis=1) 
    p_g=tf.stack((differentiate(x),differentiate(y)),axis=1)

if dim ==3:
    # Pick a path for gamma agent
    gamma_path = input("Select path for gamma agent ['circle','wild']: ")
    
    if gamma_path == "circle":    
        # Gamma agent (moves in a circle)
        x=tf.cast(tf.cos(np.linspace(0,2*np.pi,num_iters)),tf.float32)
        y=tf.cast(tf.sin(np.linspace(0,2*np.pi,num_iters)),tf.float32)
        z=tf.cast(tf.zeros(num_iters),tf.float32)
    
    elif gamma_path == "wild":
        x=tf.cast(tf.cos(np.linspace(0,2*np.pi,num_iters)),tf.float32)
        y=tf.cast(tf.cos(np.linspace(0,4*np.pi,num_iters)),tf.float32)
        z=tf.cast(tf.sin(np.linspace(0,8*np.pi,num_iters)),tf.float32)
    
    else:
        assert(False)

    q_g=tf.stack((x,y,y),axis=1)
    p_g=tf.stack((differentiate(x),differentiate(y),differentiate(z)),axis=1)


####################
# Compute trajectory
####################



###########
# Animation
###########

# Random init boids 
q=tf.Variable(tf.random_uniform((num_boids,dim)))
p=tf.Variable(0.01*tf.random_uniform((num_boids,dim)))

# Run simulation
X = tf.zeros((0,num_boids,dim))
for i in range(num_iters):
    z=uUpdate(q,p)
    q+=p*dt
    X = tf.concat(0,[X,tf.expand_dims(q,axis=0)])
    p+=(z-c_q*(q-q_g[i])-c_p*(p-p_g[i]))*dt
# Add the gamma agent
print(X)
X = tf.concat(1,[X,q_g[:,None,:]]) 
print(X)

"""
boid traj longer than gamma traj by 1
"""

##################
# Begin tf session
##################

sess = tf.Session() # Start session
sess.run(tf.global_variables_initializer()) # Initialize variables


def animate(X,save=False,show=True):
    num_iters,num_points,dim = np.shape(X)
    fig = plt.figure()

    if dim == 2:
        # Update function for animation
        def update(num):
            sc.set_offsets(X[num])
            

        # Init figure
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(-2, 2), ax.set_xticks([])
        ax.set_ylim(-2,2), ax.set_yticks([])
        sc = plt.scatter(X[0,:,0],X[0,:,1],s=5)

    if dim == 3:

        X=np.swapaxes(X,1,2) # Make data right shape for 3D animation

        def update(num):
            sc._offsets3d = X[num]
        
        # Set axes
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim3d([-2,2])
        ax.set_ylim3d([-2,2])
        ax.set_zlim3d([-2,2])
        
        # Init points
        sc = ax.scatter(X[0,0],X[0,1],X[0,2],s=5)

    # Animate
    ani = matplotlib.animation.FuncAnimation(fig,update,frames=range(num_iters),interval=20)
    if save:
        ani.save(fname+".mp4",fps=20)
        np.save(fname,X)
    if show:
        plt.show()

#animate(X,save=='y')
