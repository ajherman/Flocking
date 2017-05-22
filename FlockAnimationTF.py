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
#num_boids = input("Enter number of boids: ")
#num_iters = input("Enter number of iterations: ")
#dim = input("Enter number of dimensions [2/3]: ")
#num_boids,num_iters,dim = int(num_boids),int(num_iters),int(dim)
#save = input("Do you want to save this animation [y/n]: ")
#if save=='y':
#    fname = input("Type file name [no extension]: ")

num_boids,num_iters,dim = 3,4,2

X = tf.Variable(tf.zeros((num_iters,num_boids,dim)))
diffs = tf.Variable(tf.random_uniform((num_boids,num_boids,dim)))
q = tf.Variable(tf.random_uniform((num_boids,dim)))
p = tf.Variable(tf.random_uniform((num_boids,dim)))
g = tf.Variable(tf.random_uniform((num_iters,dim)))
init_vars = tf.global_variables_initializer()

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
    return tf.reduce_sum(phi_a(norms)*sig_grad(diff,norms),axis=0)+np.sum(rho_h(norms/r_a)*diffp,axis=0)

def differentiate(v):
    dv = tf.identity(v)
    dv = dv[1:] - v[:-1]
    zeros = tf.zeros((dv[:1]).get_shape())
    dv = tf.concat(0,[zeros,dv])  # I think this the argument order is reversed in newer tf
    return dv/dt


####################
# Compute trajectory
####################
norm = sig_norm(diffs)
grad = sig_grad(diffs,norm)
r=rho_h(diffs)
p=phi(diffs)
pa = phi_a(diffs)
d = differences(diffs)
u = uUpdate(q,p)
deriv = differentiate(g)

##################
# Begin tf session
##################

sess = tf.Session() # Start session
sess.run(init_vars) # Initialize variables

tf_diff = sess.run(diffs)
tf_q = sess.run(q)
tf_p = sess.run(p)
tf_g = sess.run(g)
tf_norm = sess.run(norm)
tf_grad = sess.run(grad)
tf_r = sess.run(r)
tf_p = sess.run(p)
tf_pa = sess.run(pa)
tf_d = sess.run(d)
tf_u = sess.run(u)
tf_deriv = sess.run(deriv)

##################
# Useful functions
##################

def np_sig_norm(z): # Sigma norm
    return (np.sqrt(1+eps*np.sum(z**2,axis=2).reshape((num_boids,num_boids,1)))-1)/eps

def np_sig_grad(z,norm=None,eps=eps): # Gradient of sigma norm
    if type(norm) == "NoneType":
        return z/(1+eps*sig_norm(z))
    else:
        return z/(1+eps*norm)

def np_rho_h(z):
    return  np.logical_and(z>=0,z<h)+np.logical_and(z<=1,z>=h)*(0.5*(1+np.cos(np.pi*(z-h)/(1-h))))

def np_phi(z):
    return 0.5*((a+b)*sig_grad(z+c,1)+(a-b))


def np_phi_a(z):
    return rho_h(z/r_a)*phi(z-d_a)
    
def np_differences(q):
    return q[:,None,:] - q

def np_uUpdate(q,p):
    diff=differences(q)
    norms = sig_norm(diff)
    diffp=differences(p)
    return np.sum(phi_a(norms)*sig_grad(diff,norms),axis=0)+np.sum(rho_h(norms/r_a)*diffp,axis=0)

def np_differentiate(v):
    dv = v.copy()
    dv[1:]-=v[:-1]
    return dv/dt


np_diff = tf_diff
np_q = tf_q
np_p = tf_p
np_g = tf_g
np_norm = np_sig_norm(np_diff)
np_grad = np_sig_grad(np_diff,np_norm)
np_r = np_rho_h(np_diff)
np_p = np_phi(np_diff)
np_pa = np_phi_a(np_diff)
np_d = np_differences(np_diff)
np_u = np_uUpdate(np_q,np_p)
np_deriv = np_differentiate(tf_g)

print("diff")
print(np_diff-tf_diff)
print("norm")
print(np_norm-tf_norm)
print("grad")
print(np_grad-tf_grad)
print("rho")
print(np_r-tf_r)
print("phi")
print(np_p-tf_p)
print("phi_a")
print(np_pa-tf_pa)
print("differences")
print(np_d-tf_d)
print("update")
print(np_u-tf_u)
print("derivs")
print(np_deriv-tf_deriv)

######################

def differentiate(v):
    dv = v.copy()
    dv[1:]-=v[:-1]
    return dv/dt

## Generate trajectory
######################
#if dim == 2:
#    # Pick a path for gamme agent
#    gamma_path = input("Select path for gamma agent ['circle','eight']: ")
#    
#    if gamma_path == "circle":
#        x=tf.cos(np.linspace(0,2*np.pi,num_iters))
#        y=tf.sin(np.linspace(0,2*np.pi,num_iters))
#
#    elif gamma_path == "eight":
#        x=tf.cos(np.linspace(0,2*np.pi,num_iters))
#        y=tf.sin(np.linspace(0,4*np.pi,num_iters))
#    
#    else:
#        assert(False)
#
#    q_g=tf.stack((x,y),axis=1) 
#    p_g=tf.stack((differentiate(x),differentiate(y)),axis=1)
#
#if dim ==3:
#    # Pick a path for gamma agent
#    gamma_path = input("Select path for gamma agent ['circle','wild']: ")
#    
#    if gamma_path == "circle":    
#        # Gamma agent (moves in a circle)
#        x=tf.cos(np.linspace(0,2*np.pi,num_iters))
#        y=tf.sin(np.linspace(0,2*np.pi,num_iters))
#        z=tf.zeros(num_iters)
#    
#    elif gamma_path == "wild":
#        x=tf.cos(np.linspace(0,2*np.pi,num_iters))
#        y=tf.cos(np.linspace(0,4*np.pi,num_iters))
#        z=tf.sin(np.linspace(0,8*np.pi,num_iters))
#    
#    else:
#        assert(False)
#
#    q_g=tf.stack((x,y,y),axis=1)
#    p_g=tf.stack((differentiate(x),differentiate(y),differentiate(z)),axis=1)
#
#
############
## Animation
############
#
## Random init boids 
#q=tf.Variable(tf.random.normal(0.0,1.0,size=(num_boids,dim)))
#p=tf.Variable(0.01*tf.random.rand(num_boids,dim))
#
## Run simulation
#X = tf.Variable(tf.zeros((num_iters,num_boids,dim)))
#for i in range(num_iters):
#    z=uUpdate(q,p)
#    q+=p*dt
#    X[i,:,:] = q
#    p+=(z-c_q*(q-q_g[i])-c_p*(p-p_g[i]))*dt
#
## Add the gamma agent
#X = np.concatenate((X,q_g[:,None,:]),axis=1) 
#
#def animate(X,save=False,show=True):
#    num_iters,num_points,dim = np.shape(X)
#    fig = plt.figure()
#
#    if dim == 2:
#        # Update function for animation
#        def update(num):
#            sc.set_offsets(X[num])
#            
#
#        # Init figure
#        ax = fig.add_axes([0, 0, 1, 1])
#        ax.set_xlim(-2, 2), ax.set_xticks([])
#        ax.set_ylim(-2,2), ax.set_yticks([])
#        sc = plt.scatter(X[0,:,0],X[0,:,1],s=5)
#
#    if dim == 3:
#
#        X=np.swapaxes(X,1,2) # Make data right shape for 3D animation
#
#        def update(num):
#            sc._offsets3d = X[num]
#        
#        # Set axes
#        ax = fig.add_subplot(111, projection='3d')
#        ax.set_xlim3d([-2,2])
#        ax.set_ylim3d([-2,2])
#        ax.set_zlim3d([-2,2])
#        
#        # Init points
#        sc = ax.scatter(X[0,0],X[0,1],X[0,2],s=5)
#
#    # Animate
#    ani = matplotlib.animation.FuncAnimation(fig,update,frames=range(num_iters),interval=20)
#    if save:
#        ani.save(fname+".mp4",fps=20)
#        np.save(fname,X)
#    if show:
#        plt.show()
#
#animate(X,save=='y')
