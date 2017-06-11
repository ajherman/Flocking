# Stuff

import numpy as np
import sys
sys.path.insert(0,"../ClassDefinitions")
from FlockAnimation import *
from FlockParameters import *

ani_params = AnimationParams()
ani_params.set_save(False)
ani_params.set_quiver(False)
ani_params.set_show(True)
flock = ScatterAnimation()
flock.params = ani_params

eps = 0.1
h = 0.2

d_a = 0.242070103255
r_a = 0.346786940882
c_q = 10
c_p = 5
num_iters = 800
dt = 0.01
num_boids = 600

def sig_norm(z): # Sigma norm
    return (np.sqrt(1+eps*z**2)-1)/eps
     
def rho_h(z):
    return  np.logical_and(z>=0,z<h)+np.logical_and(z<=1,z>=h)*(0.5*(1+np.cos(np.pi*(z-h)/(1-h))))

def phi(z):
    return 5*z/(1+eps*sig_norm(z))

def phi_a(z):
    return rho_h(z/r_a)*phi(z-d_a)

def f(d):
    n = sig_norm(d)
    f_q = phi_a(n)/(1+eps*n)
    f_p = rho_h(n/r_a)
    return f_q,f_p

def uUpdate(dq,dp):
    d = np.linalg.norm(v) 
    f_q,f_p = f(d)
    return f_q*dq + f_p*dp

def differentiate(v): # Differentiates vector
    dv = v.copy()
    dv[1:]-=v[:-1]
    return dv/dt

x=np.cos(np.linspace(0,2*np.pi*num_iters/200.,num_iters))
y=np.cos(np.linspace(0,4*np.pi*num_iters/200.,num_iters))
z=np.sin(np.linspace(0,8*np.pi*num_iters/200.,num_iters))

q_g=np.stack((x,y,y),axis=1)
p_g=np.stack((differentiate(x),differentiate(y),differentiate(z)),axis=1)

A = np.zeros((num_iters,num_boids,3))

q = np.random.rand(num_boids,3)
p = np.random.rand(num_boids,3)

 

for k in range(num_iters):
    print("Iteration: "+str(k))
    A[k] = q
    for i in range(num_boids):
        u = -c_q*(q[i]-q_g[k]) - c_p*(p[i]-p_g[k])
        for j in range(num_boids):
            dq,dp = q[i]-q[j], p[i]-p[j]
            u += uUpdate(dq,dp) 
    q += dt*p
    p += dt*u


flock.setQ(A)
flock.initAnimation()
flock.animate()
