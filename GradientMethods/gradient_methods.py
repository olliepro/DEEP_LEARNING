# gradient_methods.py
"""Volume 2: Gradient Descent Methods.
<Name>
<Class>
<Date>
"""
import numpy as np
from scipy import optimize as opt
import matplotlib.pyplot as plt
# Problem 1
def steepest_descent(f, Df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the exact method of steepest descent.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    #initialize
    i=0
    x = x0
    #run while not exceeded max iter and not converged
    while i<maxiter and np.linalg.norm(Df(x),np.inf)>tol:
        #find exact step size
        f_ = lambda a: f(x-a*Df(x).T)
        alpha = opt.minimize_scalar(f_)['x']
        #take gradient step
        x0 = x
        x = x0-alpha*Df(x0).T
        i += 1
    if i==maxiter:
        return x, False, i
    return x, True, i

# Problem 2
def conjugate_gradient(Q, b, x0, tol=1e-4,force = False):
    """Solve the linear system Qx = b with the conjugate gradient algorithm.

    Parameters:
        Q ((n,n) ndarray): A positive-definite square matrix.
        b ((n, ) ndarray): The right-hand side of the linear system.
        x0 ((n,) ndarray): An initial guess for the solution to Qx = b.
        tol (float): The convergence tolerance.

    Returns:
        ((n,) ndarray): The solution to the linear system Qx = b.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    #intialize
    r0 = Q@x0 - b
    d0 = -r0
    k = 0
    n = len(r0)
    #run while not converged
    while (np.linalg.norm(r0,np.inf)>=tol):
        #generate conjugate direction
        a = (r0.T@r0)/(d0.T@Q@d0)
        #take step
        x0 += a*d0
        r_ = r0 + (a*Q@d0)
        B0 = (r_.T@r_)/(r0.T@r0)
        #take approximate step for gradient
        d0 = -r_+(B0*d0)
        k+=1
        r0=r_
        #break if max iter exceeded
        if k>=n and not force:
            break
    if np.linalg.norm(r0,np.inf)<tol:
        return x0, True, k
    return x0, False, k

# Problem 3
def nonlinear_conjugate_gradient(f, df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the nonlinear conjugate gradient
    algorithm.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    #initialize
    r0 = -df(x0).T
    d0 = r0
    #find exact step size
    f_ = lambda a: f(x0+a*d0)
    alpha = opt.minimize_scalar(f_)['x']
    k = 1
    x0 += alpha*d0
    #Run while not converged and not reached maxiter
    while np.linalg.norm(r0)>=tol and k < maxiter:
        #calculate gradient
        r_ = -df(x0).T
        #get conjugate direction
        B0 = (r_.T@r_)/(r0.T@r0)
        d0 = r_+(B0*d0)
        f_ = lambda a: f(x0+a*d0)
        #get step size
        alpha = opt.minimize_scalar(f_)['x']
        x0 += alpha*d0
        k+=1
        r0=r_
    if np.linalg.norm(r0)<tol:
        return x0, True, k
    return x0, False, k

# Problem 4
def prob4(filename="linregression.txt",
          x0=np.array([-3482258., 15, 0, -2, -1, 0, 1829])):
    """Use conjugate_gradient() to solve the linear regression problem with
    the data from the given file, the given initial guess, and the default
    tolerance. Return the solution to the corresponding Normal Equations.
    """
    #intialize and set up least squares
    x0 = np.array([-3482258., 15, 0, -2, -1, 0, 1829])
    data = np.loadtxt('linregression.txt')
    Y = data[:,0]
    X = data
    X[:,0] = X[:,0]*0+1
    Q = X.T@X
    b = X.T@Y
    #get hyperplane normal
    return conjugate_gradient(Q, b, x0, force=True)[0]

# Problem 5
class LogisticRegression1D:
    """Binary logistic regression classifier for one-dimensional data."""

    def fit(self, x, y, guess):
        """Choose the optimal beta values by minimizing the negative log
        likelihood function, given data and outcome labels.

        Parameters:
            x ((n,) ndarray): An array of n predictor variables.
            y ((n,) ndarray): An array of n outcome variables.
            guess (array): Initial guess for beta.
        """
        #initialize negative log function
        neglog = lambda b: np.sum([np.log(1+np.exp(-(b[0]+b[1]*x[i])))+(1-y[i])*(b[0]+b[1]*x[i]) for i in range(len(x))])
        #optimize negative log function
        b = opt.fmin_cg(neglog,guess)
        #set as attributes
        self.b0 = b[0]
        self.b1 = b[1]

    def predict(self, x):
        """Calculate the probability of an unlabeled predictor variable
        having an outcome of 1.

        Parameters:
            x (float): a predictor variable with an unknown label.
        """
        return 1/(1+np.exp(-(self.b0 + self.b1*x)))

# Problem 6
def prob6(filename="challenger.npy", guess=np.array([20., -1.])):
    """Return the probability of O-ring damage at 31 degrees Farenheit.
    Additionally, plot the logistic curve through the challenger data
    on the interval [30, 100].

    Parameters:
        filename (str): The file to perform logistic regression on.
                        Defaults to "challenger.npy"
        guess (array): The initial guess for beta.
                        Defaults to [20., -1.]
    """
    #get DATA
    data = np.load("/Users/Oliver/Desktop/ACME_STUFF/volume_2/GradientMethods/challenger.npy")
    x,y = data.T
    #Create fit
    FIT = LogisticRegression1D()
    FIT.fit(x,y,np.array([20,-1]))
    #PLOT
    domain = np.linspace(30,100,200)
    plt.figure(figsize = (8,5))
    plt.title("Probability of O-ring Damage")
    plt.plot(domain,FIT.predict(domain),color = "purple")
    plt.scatter(x,y,label = "Previous Damage")
    plt.scatter(31,1,label = "P(Damage) at launch")
    plt.xlabel("Temperature")
    plt.ylabel("O-Ring Damage")
    plt.legend()
    plt.show()
    #return probability of o-ring damage at 31F
    return FIT.predict(31)
