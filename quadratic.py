"""
For the Function f(x) = x^4, this calculates the gradient descent for a given
step size and starting x0 
"""

import matplotlib.pyplot as plt
import numpy as np

# Functions
def F(x):
    """Function for f(x) = x^4"""
    return x**4

def grad_F(x):
    """Gradient of f(x)"""
    return 4*x**3

# Gradient descent
def descent(t, X0):
    """Function for plotting gradient descent for a given step size and x0"""
    npts = 100  # number of points in f(x)the graph
    niter = 100  # Number of iterations
    fig, axs = plt.subplots(1,3, figsize = (10, 5)) # Plots for the descent analysis
    axs[0].title.set_text("f(x) = x^4")
    axs[1].title.set_text("Convergence")
    axs[2].title.set_text("loglog conergence")
    
    # Graph f(x)
    x = np.linspace(0, 2, npts)
    y = np.zeros((npts, npts))
    for i in range(npts):
        y[i] = F(x[i])
    axs[0].plot(x, y)


    # Variables for the values of the gradient descent
    X = np.zeros((niter))
    X[0] = X0
    Y = np.zeros((niter))
    Y[0] = F(X[0])
    convergence = np.zeros(niter)
    convergence[0] = np.linalg.norm(X[0])

    # Perform the iterations can calculate the descent
    for i in range(niter-1):
        X[i+1] = X[i] - t * grad_F(X[i])
        convergence[i+1] = np.linalg.norm(X[i+1])
        Y[i+1] = F(X[i+1])

    # Plot the gradient descent and convergence
    axs[0].plot(X,Y,  'ro-')
    axs[1].plot(np.log10(convergence))

    # Plot the loglog of convergence
    k = np.arange(niter)
    axs[2].loglog(k, X)

    plt.show()

descent(0.125, 1) # Gradient descent for step size=0.125 and x0=1 
descent(0.12, 2)
