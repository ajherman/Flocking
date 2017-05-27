# Flocking simulation objects store all of the functions/data needed to run a simulation

import numpy as np
import tensorflow as tf

class FlockingSimulation():
    
    def __init__(self):
        pass
    
    def initSim(self):
        pass

    def runSim(self):
        pass


class OlfatiFlockingSimulation(FlockingSimulation):

    def __init__(self):

        self.eps=None
        self.num_boids=None
        self.num_iters=None
        self.gamma_path=None
        self.dim=None
        self.a=None
        self.b=None
        self.c=None
        self.h=None
        self.r_a=None
        self.d_a=None
        self.dt=None
        self.q=None
        self.p=None
        self.q_g=None
        self.p_g=None
        self.c_q=None
        self.c_p=None
    def sig_norm(self,z): # Sigma norm
        return (np.sqrt(1+self.eps*np.sum(z**2,axis=2).reshape((self.num_boids,self.num_boids,1)))-1)/self.eps
    
    def sig_grad(self,z,norm=None): # Gradient of sigma norm
        if type(norm) == "NoneType":
            return z/(1+self.eps*self.sig_norm(z))
        else:
            return z/(1+self.eps*norm)
        
    def rho_h(self,z):
        return  np.logical_and(z>=0,z<self.h)+np.logical_and(z<=1,z>=self.h)*(0.5*(1+np.cos(np.pi*(z-self.h)/(1-self.h))))

    def phi(self,z):
        return 0.5*((self.a+self.b)*self.sig_grad(z+self.c,1)+(self.a-self.b))

    def phi_a(self,z):
        return self.rho_h(z/self.r_a)*self.phi(z-self.d_a)

    def differences(self,q):
        return q[:,None,:] - q

    def uUpdate(self,q,p):
        diff=self.differences(q)
        norms = self.sig_norm(diff)
        diffp=self.differences(p)
        return np.sum(self.phi_a(norms)*self.sig_grad(diff,norms),axis=0)+np.sum(self.rho_h(norms/self.r_a)*diffp,axis=0)

    def differentiate(self,v):
        dv = v.copy()
        dv[1:]-=v[:-1]
        return dv/self.dt

    def makeGamma(self):

        # Generate trajectory
        if self.dim == 2:
            if self.gamma_path == "circle":
                x=np.cos(np.linspace(0,2*np.pi,self.num_iters))
                y=np.sin(np.linspace(0,2*np.pi,self.num_iters))

            elif self.gamma_path == "eight":
                x=np.cos(np.linspace(0,2*np.pi,self.num_iters))
                y=np.sin(np.linspace(0,4*np.pi,self.num_iters))
            
            else:
                print("Not a valid gamma agent path for dimension 2")
                assert(False)

            self.q_g=np.stack((x,y),axis=1) 
            self.p_g=np.stack((self.differentiate(x),self.differentiate(y)),axis=1)

        elif self.dim ==3:
            if self.gamma_path == "circle":    
                # Gamma agent (moves in a circle)
                x=np.cos(np.linspace(0,2*np.pi,self.num_iters))
                y=np.sin(np.linspace(0,2*np.pi,self.num_iters))
                z=np.zeros(self.num_iters)
            
            elif self.gamma_path == "wild":
                x=np.cos(np.linspace(0,2*np.pi,self.num_iters))
                y=np.cos(np.linspace(0,4*np.pi,self.num_iters))
                z=np.sin(np.linspace(0,8*np.pi,self.num_iters))
            else:
                print("Not a valid gamma agent path for dimension 3")
                assert(False)

            self.q_g=np.stack((x,y,y),axis=1)
            self.p_g=np.stack((self.differentiate(x),self.differentiate(y),self.differentiate(z)),axis=1)
        else:
            print("Invalid dimension")
            assert("False")

    def initSim(self):        
        # Random init boids 
        self.q=np.random.normal(0.0,1.0,size=(self.num_boids,self.dim))
        self.p=0.01*np.random.rand(self.num_boids,self.dim)
        # Init gamma agent
        self.makeGamma()        

    def runSim(self):
        X = np.zeros((self.num_iters,self.num_boids,self.dim))
        V = np.zeros((self.num_iters,self.num_boids,self.dim))
        for i in range(self.num_iters):
            z=self.uUpdate(self.q,self.p)
            self.q+=self.p*self.dt
            self.p+=(z-self.c_q*(self.q-self.q_g[i])-self.c_p*(self.p-self.p_g[i]))*self.dt
            X[i,:,:] = self.q
            V[i,:,:] = self.p
#            self.p+=(z-self.c_q*(self.q-self.q_g[i])-self.c_p*(self.p-self.p_g[i]))*self.dt

        # Add the gamma agent
        X = np.concatenate((X,self.q_g[:,None,:]),axis=1) 
        V = np.concatenate((V,self.p_g[:,None,:]),axis=1)

        return X,V
            

class OlfatiFlockingSimulationTF(FlockingSimulation):

    def __init__(self):

        self.eps=None
        self.num_boids=None
        self.num_iters=None
        self.gamma_path=None
        self.dim=None
        self.a=None
        self.b=None
        self.c=None
        self.h=None
        self.r_a=None
        self.d_a=None
        self.dt=None
        self.q=None
        self.p=None
        self.q_g=None
        self.p_g=None
        self.c_q=None
        self.c_p=None

    def sig_norm(self,z):
        return tf.reshape( (tf.sqrt(1+self.eps*tf.reduce_sum(z**2,axis=2))-1)/self.eps , (self.num_boids,self.num_boids,1))

    def sig_grad(self,z,norm=None):
        if norm is None:
            return z/(1+self.eps*self.sig_norm(z))
        else:
            return z/(1+self.eps*norm)

    def rho_h(self,z):
        return tf.to_float(tf.logical_and(z>=0,z<self.h))+tf.to_float(tf.logical_and(z<=1,z>=self.h))*(0.5*(1+tf.cos(np.pi*(z-self.h)/(1-self.h))))

    def phi(self,z):
        return 0.5*((self.a+self.b)*self.sig_grad(z+self.c,1)+(self.a-self.b))

    def phi_a(self,z):
        return self.rho_h(z/self.r_a)*self.phi(z-self.d_a)
        
    def differences(self,q):
        return q[:,None,:] - q

    def uUpdate(self,q,p):
        diff=self.differences(q)
        norms = self.sig_norm(diff)
        diffp=self.differences(p)
        return tf.reduce_sum(self.phi_a(norms)*self.sig_grad(diff,norms),axis=0)+tf.reduce_sum(self.rho_h(norms/self.r_a)*diffp,axis=0)

    def differentiate(self,v):
        dv = tf.identity(v)
        dv = dv[1:] - v[:-1]
        zeros = tf.zeros((dv[:1]).get_shape())
        dv = tf.concat([zeros,dv],0)  # I think this the argument order is reversed in newer tf
        return dv/self.dt

    def makeGamma(self):

        if self.dim == 2:            
            if self.gamma_path == "circle":
                x=tf.cast(tf.cos(np.linspace(0,2*np.pi,self.num_iters)),tf.float32)
                y=tf.cast(tf.sin(np.linspace(0,2*np.pi,self.num_iters)),tf.float32)
            elif self.gamma_path == "eight":
                x=tf.cast(tf.cos(np.linspace(0,2*np.pi,self.num_iters)),tf.float32)
                y=tf.cast(tf.sin(np.linspace(0,4*np.pi,self.num_iters)),tf.float32)
            else:
                print("Not a valid gamma agent path for dimensions 2")
                assert(False)

            self.q_g=tf.stack((x,y),axis=1) 
            self.p_g=tf.stack((self.differentiate(x),self.differentiate(y)),axis=1)

        elif self.dim == 3:
            if self.gamma_path == "circle":    
                # Gamma agent (moves in a circle)
                x=tf.cast(tf.cos(np.linspace(0,2*np.pi,self.num_iters)),tf.float32)
                y=tf.cast(tf.sin(np.linspace(0,2*np.pi,self.num_iters)),tf.float32)
                z=tf.cast(tf.zeros(self.num_iters),tf.float32)
            elif self.gamma_path == "wild":
                x=tf.cast(tf.cos(np.linspace(0,2*np.pi,self.num_iters)),tf.float32)
                y=tf.cast(tf.cos(np.linspace(0,4*np.pi,self.num_iters)),tf.float32)
                z=tf.cast(tf.sin(np.linspace(0,8*np.pi,self.num_iters)),tf.float32)
            else:
                print("Not a vaild gamma agent path for dimension 3")
                assert(False)

            self.q_g=tf.stack((x,y,y),axis=1)
            self.p_g=tf.stack((self.differentiate(x),self.differentiate(y),self.differentiate(z)),axis=1)

        else:
            print("Invalid dimension")
            assert(False)

    def initSim(self):        
        # Random init boids 
        self.q=tf.Variable(tf.random_uniform((self.num_boids,self.dim)))
        self.p=tf.Variable(0.01*tf.random_uniform((self.num_boids,self.dim)))
        # Init gamma agent
        self.makeGamma()

    def runSim(self):
        # Run simulation
        X = tf.zeros((0,self.num_boids,self.dim))
        V = tf.zeros((0,self.num_boids,self.dim))
        for i in range(self.num_iters):
            z=self.uUpdate(self.q,self.p)
            self.q+=self.p*self.dt
            self.p+=(z-self.c_q*(self.q-self.q_g[i])-self.c_p*(self.p-self.p_g[i]))*self.dt
            X = tf.concat([X,tf.expand_dims(self.q,axis=0)],0)
            V = tf.concat([V,tf.expand_dims(self.p,axis=0)],0)
    #        self.p+=(z-self.c_q*(self.q-self.q_g[i])-self.c_p*(self.p-self.p_g[i]))*self.dt
        # Add the gamma agent
        X = tf.concat([X,self.q_g[:,None,:]],1) 
        V = tf.concat([V,self.p_g[:,None,:]],1)

        # Begin tf session
        sess = tf.Session() # Start session
        sess.run(tf.global_variables_initializer()) # Initialize variables
        resX = sess.run(X)
        resV = sess.run(V)
        sess.close()

        return resX,resV





