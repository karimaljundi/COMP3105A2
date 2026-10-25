import numpy as np
from scipy.optimize import minimize
import cvxopt
from A2helpers import generateData

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
    # TODO: compute the average accuracies over runs
    # TODO: combine accuracies (sq_hinge and hinge)
    # TODO: return 4-by-6 train accuracy and 4-by-6 test accuracy