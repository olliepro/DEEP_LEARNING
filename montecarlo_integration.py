# montecarlo_integration.py
"""Volume 1: Monte Carlo Integration.
<Name>
<Class>
<Date>
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

# Problem 1
def ball_volume(n, N=10000):
    """Estimate the volume of the n-dimensional unit ball.

    Parameters:
        n (int): The dimension of the ball. n=2 corresponds to the unit circle,
            n=3 corresponds to the unit sphere, and so on.
        N (int): The number of random points to sample.

    Returns:
        (float): An estimate for the volume of the n-dimensional unit ball.
    """
    #get uniform sample
    points = np.random.rand(int(N), 2)

    #run samples through function
    points_combo = points[:,0]**2+points[:,1]**2

    #use mask to calculate points inside and outside
    mask = points_combo<=1
    return 4*sum(mask)/len(mask)


# Problem 2
def mc_integrate1d(f, a, b, N=10000):
    """Approximate the integral of f on the interval [a,b].

    Parameters:
        f (function): the function to integrate. Accepts and returns scalars.
        a (float): the lower bound of interval of integration.
        b (float): the lower bound of interval of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over [a,b].

    Example:
        >>> f = lambda x: x**2
        >>> mc_integrate1d(f, -4, 2)    # Integrate from -4 to 2.
        23.734810301138324              # The true value is 24.
    """
    #use the approximation formula with a translate and scale of unit interval
    points = (b-a)*np.random.rand(N)+a
    y = f(points)
    return (b-a)*sum(y)/N

# Problem 3
def mc_integrate(f, mins, maxs, N=10000):
    """Approximate the integral of f over the box defined by mins and maxs.

    Parameters:
        f (function): The function to integrate. Accepts and returns
            1-D NumPy arrays of length n.
        mins (list): the lower bounds of integration.
        maxs (list): the upper bounds of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over the domain.

    Example:
        # Define f(x,y) = 3x - 4y + y^2. Inputs are grouped into an array.
        >>> f = lambda x: 3*x[0] - 4*x[1] + x[1]**2

        # Integrate over the box [1,3]x[-2,1].
        >>> mc_integrate(f, [1, -2], [3, 1])
        53.562651072181225              # The true value is 54.
    """
    #get dim
    n = len(mins)

    #Get samples
    points = np.random.rand(n,int(N))
    vol = 1
    #Loop through max and min to translate samples
    for i in range(n):
        points[i] *= maxs[i]-mins[i]
        points[i] += mins[i]
        #get vol
        vol *= maxs[i]-mins[i]
    #evaluate samples
    evaluated = f(points)
    #use formula to calculate approximation
    return vol/N*sum(evaluated)

# Problem 4
def prob4():
    """Let n=4 and Omega = [-3/2,3/4]x[0,1]x[0,1/2]x[0,1].
    - Define the joint distribution f of n standard normal random variables.
    - Use SciPy to integrate f over Omega.
    - Get 20 integer values of N that are roughly logarithmically spaced from
        10**1 to 10**5. For each value of N, use mc_integrate() to compute
        estimates of the integral of f over Omega with N samples. Compute the
        relative error of estimate.
    - Plot the relative error against the sample size N on a log-log scale.
        Also plot the line 1 / sqrt(N) for comparison.
    """
    #get Omega, F, and initialize
    Omega = np.array([[-3/2,3/4],[0,1],[0,1/2],[0,1]])
    f = lambda x: np.exp(np.sum(-x**2/2,axis=0))/(2*np.pi)**(2)
    means, cov = np.zeros(4), np.eye(4)
    truth = scipy.stats.mvn.mvnun(list(Omega[:,0]),list(Omega[:,1]), means, cov)[0]
    domain = np.logspace(1,5,20)
    approxs = []
    error = []
    for N in domain:
        #calculate approx for various sizes of samples
        approx = mc_integrate(f,Omega[:,0],Omega[:,1],N)
        approxs.append(approx)
        #calculate relative err.
        error.append(np.abs((truth-approx)/truth))
    #PLOT it all
    plt.title("Error vs Sample Size")
    plt.plot(domain,1/np.sqrt(domain),label = "1/sqrt(N)")
    plt.plot(domain,error,label = "Error")
    plt.loglog()
    plt.xlabel("N")
    plt.ylabel("Relative Error")
    plt.legend()
    plt.show()
