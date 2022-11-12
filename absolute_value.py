"""
For the Function f(x) = |x|, this calculates the gradient descent for a given
step size and starting x0 
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

# Gradient descent
def descent(t, X0):
    """
    Function for plotting gradient descent for a given step size and x0
    It terminates if x = 0, since that point is not diffrentiable
    """
    npts = 100  # number of points in f(x)the graph
    niter = 100  # Number of iterations
    fig, axs = plt.subplots(1,2, figsize = (10, 5)) # Plots for the descent analysis
    axs[0].title.set_text("f(x) = |x|")
    axs[1].title.set_text("Convergence")
    
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
    i = 0
    while i <niter-1:
        X[i+1] = X[i] - t * grad_F(X[i])
        convergence[i+1] = np.linalg.norm(X[i+1])
        Y[i+1] = F(X[i+1])
        i+=1
        # Terminate if gradient does not exist
        if (X[i]==0):
            tmp = i
            i = niter    
            X= X[:tmp]
            Y= Y[:tmp]
            convergence = convergence[:tmp]

    # Plot the gradient descent and convergence
    axs[0].plot(X,Y,  'ro-')
    axs[1].plot(np.log10(convergence))

    plt.show()

descent(0.2, 1)  # Gradient descent for step size=0.2 and x0=1 
descent(1.5, 2)