# N-body choreographies
Orbital Choreography of same-weight planets 

Sub-space of the n-body problem where the goal is to find trajectories where planets of the same weight follow eachother. 

Examples of choreographies (from this [paper](https://arxiv.org/abs/1305.0470)): 

![alt text](chor1.png)


![alt text](chor2.png)

More on n-body choreographies [here](https://en.wikipedia.org/wiki/N-body_choreography)

## On the implementation

This implemetation leverage [JAX](https://github.com/google/jax) to perform gradient descent. Starting from random shapes the Action ( integral of the lagragian IE : kinetic energy - potential energy over the trajectory) is minimized. Minima of the action represents solutions of the problem. The trajectories are built in a trigonometrical polynomial space : 
![alt text](formula.png)


Here is an example of the optimization process.


![alt text](88.png)
