# Authors: Ari Herman & Taiyo Terada

import numpy as np
import tensorflow as tf
from numpy.linalg import norm

####################################
# Class for running flock simulation
####################################
class FlockingSimulation(): 
    def __init__(self):
        pass
    
    def initSim(self):
        pass

    def runSim(self):
        pass

############################################################
# Simulates flock with algorithm 2 from Olfati paper (Numpy)
############################################################
class OlfatiFlockingSimulation(FlockingSimulation):
    def __init__(self):
        self.params = None
        self.q=None
        self.p=None
        self.q_g=None
        self.p_g=None

# FUNCTIONS FROM PAPER
#########################################################################################################################
    
    def sig_norm(self,z): # Sigma norm
        return (np.sqrt(1+self.params.eps*np.sum(z**2,axis=2).reshape((self.params.num_boids,self.params.num_boids,1)))-1)/self.params.eps
    
    def sig_grad(self,z,norm=None): # Gradient of sigma norm
        if type(norm) == "NoneType":
            return z/(1+self.params.eps*self.sig_norm(z))
        else:
            return z/(1+self.params.eps*norm)
        
    def rho_h(self,z):
        return  np.logical_and(z>=0,z<self.params.h)+np.logical_and(z<=1,z>=self.params.h)*(0.5*(1+np.cos(np.pi*(z-self.params.h)/(1-self.params.h))))

    def phi(self,z):
        return 0.5*((self.params.a+self.params.b)*self.sig_grad(z+self.params.c,1)+(self.params.a-self.params.b))

    def phi_a(self,z):
        return self.rho_h(z/self.params.r_a)*self.phi(z-self.params.d_a)
    
    def sig_1(self,z): # sigma_1     
        return (np.sqrt(1+np.sum(z**2,axis=2).reshape((self.params.num_boids,self.params.num_boids,1)))-1)

    def phi_b(self,z):
        return self.rho_h(z/self.params.r_b)*(sig_1(z-self.params.d_b)-1)
    
#########################################################################################################################

    def diffp(self,q,p):
        diffs = self.differences(self.params.beta_pos,q)
        diffs = diffs/norm(diffs,axis=2,keepdims=True)
        proj_len =  np.sum(diffs*p[None,:,:],axis=2,keepdims=True)      
        proj = proj_len*diffs
        return proj

    def differences(self,q,b=None): # Returns array of pairwise differences 
        if b is None:
            return q[:,None,:] - q
        else:
            return q[:,None,:]-b
    
    def normalize(self,Z):
        return Z/norm(Z,axis=2,keepdims=True)
    
    def qbetadifferences(self,q):
        dqbeta=self.differences(self.beta_pos,q)
        diffqbeta=self.dqbeta-self.r_p*normalize(dqbeta)
        normqbeta=self.sig_norm(diffqbeta)/self.d_b
        rhoqbeta=self.rho_h(normqbeta)
        nhatbeta=self.sig_grad(diffqbeta,normqbeta)
        return 0 

    def uUpdate(self,q,p):
        diff=self.differences(q)
        norms = self.sig_norm(diff)
        diffp=self.differences(p)
        return self.params.c_qa*np.sum(self.phi_a(norms)*self.sig_grad(diff,norms),axis=0)+self.params.c_pa*np.sum(self.rho_h(norms/self.params.r_a)*diffp,axis=0)
    
    def bUpdate(self,q,p):
        dqbeta=self.differences(self.params.beta_pos,q)
        diffqbeta=dqbeta-self.r_p*normalize(dqbeta)
        normqbeta=self.sig_norm(diffqbeta)
        nhatbeta=self.sig_grad(diffqbeta,normqbeta)
        rhoqbeta=self.rho_h(normqbeta/self.d_b)
        
        diffs = self.differences(self.params.beta_pos,q)
        diffs = diffs/norm(diffs,axis=2,keepdims=True)
        proj_len =  np.sum(diffs*p[None,:,:],axis=2,keepdims=True)      
        proj = proj_len*diffs
        
        return self.params.c_qb*np.sum(self.phi_b(normqbeta)*nhatbeta,axis=0)+self.params.c_pb*np.sum(rhoqbeta*proj,axis=0)
        
    
    def differentiate(self,v): # Differentiates vector
        dv = v.copy()
        dv[1:]-=v[:-1]
        return dv/self.params.dt

    def makeGamma(self): # Generates/sets trajectory for gamma agent
        if self.params.dim == 2:
            if self.params.gamma_path == "circle":
                x=np.cos(np.linspace(0,2*np.pi*self.params.num_iters/200.,self.params.num_iters))
                y=np.sin(np.linspace(0,2*np.pi*self.params.num_iters/200.,self.params.num_iters))

            elif self.params.gamma_path == "eight":
                x=np.cos(np.linspace(0,2*np.pi*self.params.num_iters/200.,self.params.num_iters))
                y=np.sin(np.linspace(0,4*np.pi*self.params.num_iters/200.,self.params.num_iters))
            
            else:
                print("Not a valid gamma agent path for dimension 2")
                assert(False)

            self.q_g=np.stack((x,y),axis=1) 
            self.p_g=np.stack((self.differentiate(x),self.differentiate(y)),axis=1)

        elif self.params.dim ==3:
            if self.params.gamma_path == "circle":    
                x=np.cos(np.linspace(0,2*np.pi*self.params.num_iters/200.,self.params.num_iters))
                y=np.sin(np.linspace(0,2*np.pi*self.params.num_iters/200.,self.params.num_iters))
                z=np.zeros(self.params.num_iters)
            
            elif self.params.gamma_path == "wild":
                x=np.cos(np.linspace(0,2*np.pi*self.params.num_iters/200.,self.params.num_iters))
                y=np.cos(np.linspace(0,4*np.pi*self.params.num_iters/200.,self.params.num_iters))
                z=np.sin(np.linspace(0,8*np.pi*self.params.num_iters/200.,self.params.num_iters))

            else:
                print("Not a valid gamma agent path for dimension 3")
                assert(False)

            self.q_g=np.stack((x,y,y),axis=1)
            self.p_g=np.stack((self.differentiate(x),self.differentiate(y),self.differentiate(z)),axis=1)
        else:
            print("Invalid dimension")
            assert("False")

    def initSim(self): # Must be called before runSim       
        # Random init boids 
        self.q=self.params.q_init 
        self.p=self.params.p_init 

        # Init gamma agent
        self.makeGamma()        

    def runSim(self): # Runs simulation and returns pos, vel data arrays
        X = np.zeros((self.params.num_iters,self.params.num_boids,self.params.dim))
        V = np.zeros((self.params.num_iters,self.params.num_boids,self.params.dim))
        for i in range(self.params.num_iters):
            z=self.uUpdate(self.q,self.p)
            self.q+=self.p*self.params.dt
            self.p+=(z-self.params.c_q*(self.q-self.q_g[i])-self.params.c_p*(self.p-self.p_g[i]))*self.params.dt
            X[i,:,:] = self.q
            V[i,:,:] = self.p

        # Add the gamma agent
        X = np.concatenate((X,self.q_g[:,None,:]),axis=1) 
        V = np.concatenate((V,self.p_g[:,None,:]),axis=1)
        
        # Save array
        if self.params.save:
            np.save(self.params.fname,[X,V])

        return X,V


#################################################################
# Simulates flock with algorithm 2 from Olfati paper (Tensorflow)
#################################################################
class OlfatiFlockingSimulationTF(FlockingSimulation):
    def __init__(self):
        self.params=None
        self.q=None
        self.p=None
        self.q_g=None
        self.p_g=None


# FUNCTIONS FROM PAPER
#########################################################################################################################

    def sig_norm(self,z):
        return tf.reshape( (tf.sqrt(1+self.params.eps*tf.reduce_sum(z**2,axis=2))-1)/self.params.eps , (self.params.num_boids,self.params.num_boids,1))

    def sig_grad(self,z,norm=None):
        if norm is None:
            return z/(1+self.params.eps*self.sig_norm(z))
        else:
            return z/(1+self.params.eps*norm)

    def rho_h(self,z):
        return tf.to_float(tf.logical_and(z>=0,z<self.params.h)) +tf.to_float(tf.logical_and(z<=1,z>=self.params.h))*(0.5*(1+tf.cos(np.pi*(z-self.params.h)/(1-self.params.h))))

    def phi(self,z):
        return 0.5*((self.params.a+self.params.b)*self.sig_grad(z+self.params.c,1)+(self.params.a-self.params.b))

    def phi_a(self,z):
        return self.rho_h(z/self.params.r_a)*self.phi(z-self.params.d_a)
        
#########################################################################################################################

    def differences(self,q): # Returns array of pairwise differences
        return q[:,None,:] - q

    def uUpdate(self,q,p):
        diff=self.differences(q)
        norms = self.sig_norm(diff)
        diffp=self.differences(p)
        return tf.reduce_sum(self.phi_a(norms)*self.sig_grad(diff,norms),axis=0)+tf.reduce_sum(self.rho_h(norms/self.params.r_a)*diffp,axis=0)

    def differentiate(self,v): # Differentiates vector
        dv = tf.identity(v)
        dv = dv[1:] - v[:-1]
        zeros = tf.zeros((dv[:1]).get_shape())
        dv = tf.concat([zeros,dv],0)  # I think this the argument order is reversed in newer tf
        return dv/self.params.dt

    def makeGamma(self): # Generates/sets trajectory for gamma agent

        if self.params.dim == 2:            
            if self.params.gamma_path == "circle":
                x=tf.cast(tf.cos(np.linspace(0,2*np.pi*self.params.num_iters/200.,self.params.num_iters)),tf.float32)
                y=tf.cast(tf.sin(np.linspace(0,2*np.pi*self.params.num_iters/200.,self.params.num_iters)),tf.float32)
            elif self.params.gamma_path == "eight":
                x=tf.cast(tf.cos(np.linspace(0,2*np.pi*self.params.num_iters/200.,self.params.num_iters)),tf.float32)
                y=tf.cast(tf.sin(np.linspace(0,4*np.pi*self.params.num_iters/200.,self.params.num_iters)),tf.float32)
            else:
                print("Not a valid gamma agent path for dimensions 2")
                assert(False)

            self.q_g=tf.stack((x,y),axis=1) 
            self.p_g=tf.stack((self.differentiate(x),self.differentiate(y)),axis=1)

        elif self.params.dim == 3:
            if self.params.gamma_path == "circle":    
                # Gamma agent (moves in a circle)
                x=tf.cast(tf.cos(np.linspace(0,2*np.pi*self.params.num_iters/200.,self.params.num_iters)),tf.float32)
                y=tf.cast(tf.sin(np.linspace(0,2*np.pi*self.params.num_iters/200.,self.params.num_iters)),tf.float32)
                z=tf.cast(tf.zeros(self.params.num_iters),tf.float32)
            elif self.params.gamma_path == "wild":
                x=tf.cast(tf.cos(np.linspace(0,2*np.pi*self.params.num_iters/200.,self.params.num_iters)),tf.float32)
                y=tf.cast(tf.cos(np.linspace(0,4*np.pi*self.params.num_iters/200.,self.params.num_iters)),tf.float32)
                z=tf.cast(tf.sin(np.linspace(0,8*np.pi*self.params.num_iters/200.,self.params.num_iters)),tf.float32)
        
            else:
                print("Not a vaild gamma agent path for dimension 3")
                assert(False)

            self.q_g=tf.stack((x,y,y),axis=1)
            self.p_g=tf.stack((self.differentiate(x),self.differentiate(y),self.differentiate(z)),axis=1)

        else:
            print("Invalid dimension")
            assert(False)

    def initSim(self): # Must call before runSim
        # Random init boids 
        self.q=tf.Variable(self.params.q_init,dtype=tf.float32) 
        self.p=tf.Variable(self.params.p_init,dtype=tf.float32) 
        # Init gamma agent
        self.makeGamma()

    def runSim(self): # Runs simulations and returns pos, vel data arrays
        # Run simulation
        X = tf.zeros((0,self.params.num_boids,self.params.dim))
        V = tf.zeros((0,self.params.num_boids,self.params.dim))
        for i in range(self.params.num_iters):
            z=self.uUpdate(self.q,self.p)
            self.q+=self.p*self.params.dt
            self.p+=(z-self.params.c_q*(self.q-self.q_g[i])-self.params.c_p*(self.p-self.p_g[i]))*self.params.dt
            X = tf.concat([X,tf.expand_dims(self.q,axis=0)],0)
            V = tf.concat([V,tf.expand_dims(self.p,axis=0)],0)
        
        # Add the gamma agent
        X = tf.concat([X,self.q_g[:,None,:]],1) 
        V = tf.concat([V,self.p_g[:,None,:]],1)

        # Begin tf session
        sess = tf.Session() # Start session
        sess.run(tf.global_variables_initializer()) # Initialize variables
        resX = sess.run(X)
        resV = sess.run(V)
        sess.close()
        
        # Save
        if self.params.save:
            np.save(self.params.fname,[resX,resV])

        return resX,resV







