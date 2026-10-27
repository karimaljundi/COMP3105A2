from turtle import st

import numpy as np
from scipy.optimize import minimize
import cvxopt
from A2helpers import generateData, linearKernel, polyKernel, gaussKernel

def minSqHinge(X, y, lamb):
    d = X.shape[1]
    def obj_func(u):

        w0 = u[-1]  # last unknown is w0
        w = u[:-1]  # the first d dimensions are w
        w = w[:, None]  # make it d-by-1

        m = y * (X @ w + w0)
        loss = np.sum(np.square(np.maximum(0, 1 - m)))
        reg = 0.5 * lamb * np.sum(np.square(w))

        return loss+ reg

    u0 = np.ones(d + 1)  # first d dimension for w, and last for w0

    sol = minimize(obj_func, u0)
    # Get the solution
    w = sol.x[:-1][:, None]  # make it d-by-1
    w0 = sol.x[-1]

    return w, w0
def minHinge(X, y, lamb, stabilizer=1e-5):
    d = X.shape[1]
    n = X.shape[0]
    q = np.vstack([np.zeros((d+1, 1)), np.ones((n,1))])


    #matrix of zeros
    P = np.zeros((d+1+n, d+1+n))
    #top left of matrix needs to be replaced by lamba
    P[:d, :d] = lamb * np.eye(d)
    #adding stabilizer to every diagonal entry
    P = P + stabilizer * np.eye(d+1+n)

    # xi >= 0 is -xi <= 0
    G1 = np.hstack([np.zeros((n,d)),  np.zeros((n,1)), -np.eye(n)])
    h1 = np.zeros((n,1))

    # 1 - y_i (x_i ^ T w + w0) <= xi_i is written as -Delta(y) X w - y w0 - xi <= -1
    G2 = np.hstack([-y * X, -y, -np.eye(n)])
    h2 = -np.ones((n,1))

    G = np.vstack([G1, G2])
    h = np.vstack([h1, h2])

    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(cvxopt.matrix(P), cvxopt.matrix(q),
                            cvxopt.matrix(G), cvxopt.matrix(h))
    u = np.array(sol['x'])
    w = u[:d]
    w0 = u[d,0]
    return w, w0

def classify(Xtest, w, w0):
    #need np.where to dictate if the linear prediction is zero or greater we predict +1
    ytest = np.where(((Xtest @ w) + w0) >=0, 1, -1)
    return ytest
def synExperimentsRegularize():
    n_runs = 100
    n_train = 100
    n_test = 1000
    lamb_list = [0.001, 0.01, 0.1, 1.]
    gen_model_list = [1, 2, 3]
    train_acc_sq_hinge = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    test_acc_sq_hinge = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    train_acc_hinge = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    test_acc_hinge = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    # TODO: Change the following random seed to your GROUP ID
    # (the XX in A2-XX on Brightspace)
    np.random.seed(176)
    for r in range(n_runs):
        for i, lamb in enumerate(lamb_list):
            for j, gen_model in enumerate(gen_model_list):
                Xtrain, ytrain = generateData(n=n_train, gen_model=gen_model)
                Xtest, ytest = generateData(n=n_test, gen_model=gen_model)
                w, w0 = minSqHinge(Xtrain, ytrain, lamb)


                train_acc_sq_hinge[i, j, r] = np.mean(classify(Xtrain, w, w0) == ytrain)# TODO: compute accuracy on training set
                test_acc_sq_hinge[i, j, r] = np.mean(classify(Xtest, w, w0)== ytest)
                # TODO: compute accuracy on test set
                w, w0 = minHinge(Xtrain, ytrain, lamb)
                train_acc_hinge[i, j, r] = np.mean(classify(Xtrain, w, w0) == ytrain)# TODO: compute accuracy on training set
                test_acc_hinge[i, j, r] = np.mean(classify(Xtest, w, w0)==ytest)# TODO: compute accuracy on test set
    
    train_accuracy = np.hstack([np.mean(train_acc_sq_hinge, axis=2),
                                np.mean(train_acc_hinge, axis=2)])
    test_accuracy = np.hstack([np.mean(test_acc_sq_hinge, axis=2),
                               np.mean(test_acc_hinge, axis=2)])
    return train_accuracy, test_accuracy
def adjSqHinge(X, y, lamb, kernel_func):
    # w = X^T * a => x_i ^ T * w becomes k_i^T * a
    K = kernel_func(X,X)
    n = X.shape[0]
    def obj_func(u):
            a0 = u[-1]  
            a = u[:-1]  
            a = a[:, None]  
    
            m = y * (K @ a + a0) # n by 1
            loss = np.sum(np.square(np.maximum(0, 1 - m)))
            reg = 0.5 * lamb * (a.T @ K @ a) # (lambda / 2 )* a^T * K * a
    
            return loss+ reg
    
    u0 = np.ones(n + 1)  
    
    sol = minimize(obj_func, u0)

    a = sol.x[:-1][:, None]  
    a0 = sol.x[-1]
    
    return a, a0
def adjHinge(X, y, lamb, kernel_func, stabilizer=1e-5):
    n = X.shape[0]
    K = kernel_func(X, X)

    q = np.vstack([np.zeros((n+1, 1)), np.ones((n,1))])

    # (1/2) * u^T * P * u = (lambda/2) * a^T * K * a
    P = np.zeros((2 * n + 1, 2*n + 1))
    #top left of matrix needs to be replaced by lamba
    P[:n, :n] = lamb * K
    #adding stabilizer to every diagonal entry
    P = P + stabilizer * np.eye(2 * n  +1)

    # xi >= 0 rewritten as -xi <= 0 
    G1 = np.hstack([np.zeros((n,n)),  np.zeros((n,1)), -np.eye(n)])
    h1 = np.zeros((n,1))

    # 1 - y_i (k_i ^ T * a + a0) <= xi_i
    # rewritten as -Delta(y) * K * a - y * a0 - xi <= -1
    G2 = np.hstack([-y * K, -y, -np.eye(n)])
    h2 = -np.ones((n,1))

    G = np.vstack([G1, G2])
    h = np.vstack([h1, h2])

    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(cvxopt.matrix(P), cvxopt.matrix(q),
                            cvxopt.matrix(G), cvxopt.matrix(h))
    u = np.array(sol['x'])
    a = u[:n]
    a0 = u[n,0]
    return a, a0    
def adjClassify(Xtest, a, a0, X, kernel_func):
    scores = kernel_func(Xtest, X) @ a + a0
    return np.where(scores >=0, 1, -1)
def synExperimentsKernel():
    n_runs = 10
    n_train = 100
    n_test = 1000
    lamb = 0.001
    kernel_list = [linearKernel,
    lambda X1, X2: polyKernel(X1, X2, 2),
    lambda X1, X2: polyKernel(X1, X2, 3),
    lambda X1, X2: gaussKernel(X1, X2, 1.0),
    lambda X1, X2: gaussKernel(X1, X2, 0.5)]
    gen_model_list = [1, 2, 3]
    train_acc_sq_hinge = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    test_acc_sq_hinge = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    train_acc_hinge = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    test_acc_hinge = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    # TODO: Change the following random seed to your GROUP ID
    # (the XX in A2-XX on Brightspace)
    np.random.seed(176)
    for r in range(n_runs):
        for i, kernel in enumerate(kernel_list):
            for j, gen_model in enumerate(gen_model_list):
                Xtrain, ytrain = generateData(n=n_train, gen_model=gen_model)
                Xtest, ytest = generateData(n=n_test, gen_model=gen_model)
                a, a0 = adjSqHinge(Xtrain, ytrain, lamb, kernel)


                train_acc_sq_hinge[i, j, r] = np.mean(adjClassify(Xtrain, a, a0, Xtrain, kernel) == ytrain)# TODO: compute accuracy on training set
                test_acc_sq_hinge[i, j, r] = np.mean(adjClassify(Xtest, a, a0, Xtrain, kernel)== ytest)
                # TODO: compute accuracy on test set
                a, a0 = adjHinge(Xtrain, ytrain, lamb, kernel)
                train_acc_hinge[i, j, r] = np.mean(adjClassify(Xtrain, a, a0, Xtrain, kernel) == ytrain)# TODO: compute accuracy on training set
                test_acc_hinge[i, j, r] = np.mean(adjClassify(Xtest, a, a0, Xtrain, kernel)==ytest)# TODO: compute accuracy on test set
    
    train_accuracy = np.hstack([np.mean(train_acc_sq_hinge, axis=2),
                                np.mean(train_acc_hinge, axis=2)])
    test_accuracy = np.hstack([np.mean(test_acc_sq_hinge, axis=2),
                               np.mean(test_acc_hinge, axis=2)])
    return train_accuracy, test_accuracy    
def dualHinge(X, y, lamb, kernel_func, stabilizer=1e-5):
    n = X.shape[0]
    y = y.astype(float)
    K = kernel_func(X, X)

    # D(y) * K D(y) is the same as Y* Y^T * K
    P = (1 / lamb) * ( y @ y.T) * K + stabilizer * np.eye(n)
    q = -np.ones((n,1))


    #Box constraints 0<= a <= 1 translated to -a <=0 and a <=1
    G = np.vstack([-np.eye(n), np.eye(n)])
    h = np.vstack([np.zeros((n,1)), np.ones((n,1))])

    #euqlity constraint y^T * a = 0
    A = y.reshape(1, -1)
    b_eq = np.zeros((1,1))

    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(cvxopt.matrix(P), cvxopt.matrix(q),
                            cvxopt.matrix(G), cvxopt.matrix(h),
                            cvxopt.matrix(A), cvxopt.matrix(b_eq))
    a = np.array(sol['x'])


    # use the a_i closes to 0.5 
    i = np.argmin(np.abs(a-0.5))
    b = (y[i,0] - (1 / lamb) * K[i:i + 1] @ (y * a)).item()
    
    return a,b     