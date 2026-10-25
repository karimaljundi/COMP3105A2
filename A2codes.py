import numpy as np
from scipy.optimize import minimize

def minSqHinge(X, y, lamb):



    d = X.shape[1]

 
    def obj_func(u):

        w0 = u[-1]  # last unknown is w0
        w = u[:-1]  # the first d dimensions are w
        w = w[:, None]  # make it d-by-1

        m = y * (X @ w + w0)
        loss = np.sum(np.square(np.maximum(0, 1 - m)))
        reg = 0.5 * lamb * float(w.T @ w)

        return loss+ reg

    u0 = np.ones(d + 1)  # first d dimension for w, and last for w0

    sol = minimize(obj_func, u0)
    # Get the solution
    w = sol.x[:-1][:, None]  # make it d-by-1
    w0 = sol.x[-1]

    return w, w0