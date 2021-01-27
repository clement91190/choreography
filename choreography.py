import jax.numpy as jnp
np = jnp
from jax import grad, jit, vmap
from jax import random
import jax
import matplotlib.pyplot as plt
key = jax.random.PRNGKey(0)  # key to make sure everything is repetible
import numpy as np
import numpy as np
from jax import value_and_grad
from matplotlib.pyplot import figure

class Choreography():
    def __init__(self, n_body=3, n_points=1000, D=10, potential=2):
        """
        @params : n_body : number of bodies for the problem 
        @params : n_points : number of timestep on the trajectory 
        @params : D : degree of cos/sin polynomial of trajectory
        @params : potential : gravity factor : attraction = (1/r) ** potential (Newtonian = 2 )
        """
        self.D = D
        self.n_points = n_points
        self.n_body = n_body
        self.potential = potential
        
        self.dt = 2 * jnp.pi / self.n_points
        self.t = jnp.arange(0, 2 * jnp.pi, self.dt)

    def get_random_weights(self):
        return random.uniform(key,(4, self.D), minval=-1.) / (2 ** jnp.arange(1, self.D + 1) )

    def get_8_starting_weights(self, degree=1):
        """ return a 8 loop (or triple or quadruple ... loop) 
        @param degree int : 1 for the classical 8, more if you want."""
        # Init 8 loop

        weights = np.zeros((4, self.D))
        weights[0, 0] = 1
        weights[3, 0] = 1
        weights[0, degree] = 5.
        return weights

    def plot_weight(self, weights):
            x,y = self.traj(weights, self.t)
            return plt.plot(x,y)

    def cleanup_weights(self, w):
        """cleanups index of D that divide n_bodys for symmetry reason """
        return jax.ops.index_update(w, jax.ops.index[:, self.n_body - 1::self.n_body], 0.)  # symmetry 


    def traj(self, w, t):
        """ @params: w : weights as a matrix ( 4, D) 
        @params t : timesteps """ 

        # choregraphy represented in the sin/cos space
        # w = w / (2 ** jnp.arange(1, D + 1) )
        [a,b,c,d] = self.cleanup_weights(w)
        fourier = (jnp.arange(1 , self.D + 1, 1)) #[ 1, 2,3,4, ..., D]
        tfourier= jnp.outer(t, fourier) # outer product fourier * t
        sins, coss = jnp.sin(tfourier), jnp.cos(tfourier)
        y = jnp.sum(a * sins + b * coss , axis=1)
        x = jnp.sum(c * sins + d * coss, axis=1) 
        return (x, y)

    def dtraj(self, w,t):
        """ derivative of the trajectory"""
        # w = w / (2 ** jnp.arange(1, D + 1) )

        [a,b,c,d] = self.cleanup_weights(w)
        fourier = (jnp.arange(1 , self.D + 1, 1))
        tfourier= jnp.outer(t, fourier)
        sins, coss = jnp.sin(tfourier), jnp.cos(tfourier)
        y = jnp.sum(a * fourier * coss - b * fourier * sins , axis=1)
        x = jnp.sum(c * fourier * coss - d * fourier * sins, axis=1) 
        return (x, y)

    def lagrangian(self, w, t):
        # t : arange [0, 2pi[ 
        x, y = self.traj(w, t)
        
        roll_val = int(len(t) / self.n_body)
        L = 0
        
        # U
        for i in range(self.n_body):
            xi = jnp.roll(x, roll_val * i)
            yi = jnp.roll(y, roll_val * i)
            for j in range(i + 1, self.n_body):
                xj = jnp.roll(x, roll_val * j)
                yj = jnp.roll(y, roll_val * j)
            
                rij = jnp.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                L += 1. / rij
        
        # K 
        vx, vy = self.dtraj(w, t)
        L += 0.5 * (vx ** 2 + vy ** 2)
        return L
    
    def action(self, w):
        # integral of the lagrangian over [0 : 2pi[
        dt = 2 * jnp.pi / self.n_points
        t = jnp.arange(0, 2 * jnp.pi + dt, dt)
        L = self.lagrangian(w, t)
        return np.sum(L) * dt

    def optimize(self, weights, n_steps=None, delta_stop=0.001, optimization_step=0.005, verbose=True):
        """ @params : weights (see above)
        @params, delta_stop : difference between steps of optimization
        @params optimization_step for the gradient descent """
        W = weights 
        plt.figure(figsize=(8, 8))
        los_val = None
        stop = False
        i=0
        #for i in range(400):
        while not stop:
            if n_steps is not None:
                stop = i > n_steps
            i += 1
            L, W_grad = value_and_grad(self.action)(W)
            if n_steps is None and los_val is not None and los_val - L < delta_stop:
                print(los_val, L)
                stop = True
            los_val = L
            W = W - optimization_step * W_grad
            if i % 20 == 0:
                x, y = (self.traj(W, self.t))
                if verbose:
                    plt.plot(x,y)
                    print("Action : ", L)
        return W
