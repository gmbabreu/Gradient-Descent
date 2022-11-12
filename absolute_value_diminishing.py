"""
For the Function f(x) = |x|, this calculates the gradient descent for a given
diminishing beta for step size and starting x0 
"""

import matplotlib.pyplot as plt
import numpy as np

# Functions
def F(x):
    """Function for f(x) = |x|"""
    return abs(x)

def grad_F(x):
    """Gradient of f(x)"""
    return x/abs(x)

# Gradient diminishing descent
def diminishing_descent(beta, X0):
    """
    Function for plotting gradient descent for a given beta and x0
    It terminates if x = 0, since that point is not diffrentiable
    """
    npts = 100  # number of points in f(x)the graph
    niter = 100  # Number of iterations
    fig, axs = plt.subplots(1,3, figsize = (10, 5)) # Plots for the descent analysis
    axs[0].title.set_text("f(x) = |x|")
    axs[1].title.set_text("Convergence")
    axs[2].title.set_text("loglog conergence")

    # Graph f(x)
    x = np.linspace(-4, 4, npts)
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
    k = 0
    while k <niter-1:
        st = 1.1/((k+1)**beta)
        
        X[k+1] = X[k] - st * grad_F(X[k])
        convergence[k+1] = np.linalg.norm(X[k+1])
        Y[k+1] = F(X[k+1])
        k+=1
        # Terminate if gradient does not exist
        if (X[k]==0):
            tmp = k
            k = niter    
            X= X[:tmp]
            Y= Y[:tmp]
            convergence = convergence[:tmp]

    # Plot the gradient descent and convergence
    axs[0].plot(X,Y,  'ro-')
    axs[1].plot(np.log10(convergence))

    # Plot the loglog of convergence
    k = np.arange(niter)
    axs[2].loglog(k, X)

    plt.show()

diminishing_descent(0.5, 1)  # Gradient descent for beta=0.5 and x0=1 
diminishing_descent(1, 1)