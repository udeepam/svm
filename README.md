# SVM: from scratch
This repository contains python code for training and testing a multiclass soft-margin kernelised SVM implemented using NumPy. 

## Overview
The loss functions used are
* L1-SVM: standard hinge loss ,
* L2-SVM: squared hinge loss. 

The constrained optimisation problems are solved using  
* Log barrier Interior point method with the feasible start Newton method,
* Sequenital Minimal Optimisation  (SMO) method.
* `CVXOPT` python package: https://cvxopt.org/userguide/coneprog.html

Generalisation to the multiclass setting can be achieved using
* One vs One (OVO)
* One vs Rest (OVR)

An example Jupyter notebook is provided for training and testing a support vector classifier (SVC) on a reduced version of the MNIST dataset.

## Code
A list of optimisation algorithms that have been coded up in this repository include
* Log barrier Interior point
* Feasible Newton
* SMO
* Backtracking linesearch obeying Armijo conditions
* Linesearch obeying strong Wolfe conditions
* Descent algorithm using steepest descent direction and Newton direction

Kernel functions available
* Gaussian radial basis function (RBF) kernel
* Polynomial kernel

## Resources
Useful resources
* http://cs229.stanford.edu/materials/smo.pdf (psuedocode for the simplified SMO algorithm by Andrew Ng)
* https://web.stanford.edu/~boyd/cvxbook/ (psuedocode for the barrier method and feasible Newton method)
* https://link.springer.com/book/10.1007/978-0-387-40065-5 (for linesearch methods)
* https://github.com/ywteh/advml2020 
