# User input# User input

import numpy as np

def getUserInput():

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
    quiver = False
    show = False

    # Get parameters from user
    num_boids = input("Enter number of boids: ")
    num_iters = input("Enter number of iterations: ")
    dim = input("Enter number of dimensions [2/3]: ")
    num_boids,num_iters,dim = int(num_boids),int(num_iters),int(dim)
    save = input("Do you want to save this animation [y/n]: ")

    if save=='y':
        fname = input("Type file name [no extension]: ")
    else:
        fname = None

    display = input("Do you want to show this animation [y/n]: ")
    if display == 'y':
        show = True
        quiver_display = input("Do you want a quiver animation [y/n]: ")
        if quiver_display == 'y':
            quiver = True
    return eps,d,r,d_a,r_a,a,b,c,h,dt,c_q,c_p,quiver,show,num_boids,num_iters,dim,save,fname
