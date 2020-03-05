# polynomial_interpolation.py
"""Volume 2: Polynomial Interpolation.
Oliver Proudfoot
Math 326
Jan 16, 2020
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BarycentricInterpolator as BI
import numpy.linalg as la
# Problems 1 and 2
def lagrange(xint, yint, points):
    """Find an interpolating polynomial of lowest degree through the points
    (xint, yint) using the Lagrange method and evaluate that polynomial at
    the specified points.

    Parameters:
        xint ((n,) ndarray): x values to be interpolated.
        yint ((n,) ndarray): y values to be interpolated.
        points((m,) ndarray): x values at which to evaluate the polynomial.

    Returns:
        ((m,) ndarray): The value of the polynomial at the specified points.
    """
    #produce the Lagrange polynomials in an array
    L = lambda j,X,x: np.prod([x-xi for xi in X if xi!=X[j]],axis=0)/ np.prod([X[j]-xi for xi in X if xi!=X[j]])

    #return the result of the approximation
    return np.sum(np.array([yint[i]*L(i,xint,points) for i in range(len(xint))]),axis=0)

# Problems 3 and 4
class Barycentric:
    """Class for performing Barycentric Lagrange interpolation.

    Attributes:
        w ((n,) ndarray): Array of Barycentric weights.
        n (int): Number of interpolation points.
        x ((n,) ndarray): x values of interpolating points.
        y ((n,) ndarray): y values of interpolating points.
    """

    def __init__(self, xint, yint):
        """Calculate the Barycentric weights using initial interpolating points.

        Parameters:
            xint ((n,) ndarray): x values of interpolating points.
            yint ((n,) ndarray): y values of interpolating points.
        """
        #initialize
        self.x = xint
        self.y = yint

        n = len(xint)                   # Number of interpolating points.
        w = np.ones(n)                  # Array for storing barycentric weights.

        # Calculate the capacity of the interval.
        C = (np.max(xint) - np.min(xint)) / 4
        shuffle = np.random.permutation(n-1)
        for j in range(n):
            temp = (xint[j] - np.delete(xint, j)) / C
            temp = temp[shuffle]        # Randomize order of product.
            w[j] /= np.product(temp)
        self.weights = w

    def __call__(self, points):
        """Using the calcuated Barycentric weights, evaluate the interpolating polynomial
        at points.

        Parameters:
            points ((m,) ndarray): Array of points at which to evaluate the polynomial.

        Returns:
            ((m,) ndarray): Array of values where the polynomial has been computed.
        """
        #Evaluate the Barycentric Interpolation at points using formula
        return sum([self.y[j]*self.weights[j]/(points-self.x[j]) for j in range(len(self.x))])  /  sum([self.weights[j]/(points-self.x[j]) for j in range(len(self.x))])

    # Problem 4
    def add_weights(self, xint, yint):
        """Update the existing Barycentric weights using newly given interpolating points
        and create new weights equal to the number of new points.

        Parameters:
            xint ((m,) ndarray): x values of new interpolating points.
            yint ((m,) ndarray): y values of new interpolating points.
        """
        #initialize
        old_n = len(self.x)

        #update x
        self.x = np.concatenate((self.x,xint))

        #store old weights
        n = len(self.x)
        self.weights **= -1
        old_wi = self.weights

        #initialize new weights list
        self.weights = np.zeros(n)
        self.weights[:old_n] = old_wi

        #iteratively add and update
        for j in range(old_n,n):
            self.weights[:j] *= (self.x[j]-self.x[:j])
            self.weights[j] = np.multiply.reduce(self.x[:j]-self.x[j])

        #get weights back to correct form
        self.weights **= -1

        #update y
        self.y = np.append(self.y,yint)

        """    f = lambda x: 1/(1+25 * x**2)
            x = np.linspace(-1,1,5)
            Bary = Barycentric(x,f(x))
            domain = np.linspace(-.8,.8,200)
            plt.plot(domain,Bary(domain))
            plt.plot(domain,f(domain))
            toadd = np.linspace(-.9,.9,100)[np.abs(np.linspace(-.9,.9,100))>.7]
            Bary.add_weights(toadd,f(toadd))
            plt.plot(domain,Bary(domain))
            plt.ylim(-.2,1.2)
            plt.show()"""



# Problem 5
def prob5():
    """For n = 2^2, 2^3, ..., 2^8, calculate the error of intepolating Runge's
    function on [-1,1] with n points using SciPy's BarycentricInterpolator
    class, once with equally spaced points and once with the Chebyshev
    extremal points. Plot the absolute error of the interpolation with each
    method on a log-log plot.
    """
    #initialize
    f = lambda x: 1/(1+25 * x**2)
    polyerr=[]
    cheberr=[]
    N = [2**i for i in range(2,9)]

    #calculate approximation error at 2**i for i in range 2-8
    for n in N:
        pts = np.linspace(-1, 1, n)
        cheb = np.array([np.cos((2*i-1)*np.pi/(2*n+2)) for i in range(1,n+1)])
        domain = np.linspace(-1, 1, 400)
        poly = BI(pts,f(pts))
        poly_cheb = BI(cheb,f(cheb))

        #use the inf norm to get error
        cheberr.append(la.norm(f(domain)-poly_cheb(domain),ord=np.inf))
        polyerr.append(la.norm(f(domain)-poly(domain),ord=np.inf))

    #plot results
    plt.plot(N, cheberr,marker='o')
    plt.plot(N, polyerr,marker='o')
    plt.loglog()
    plt.show()
# Problem 6
def chebyshev_coeffs(f, n):
    """Obtain the Chebyshev coefficients of a polynomial that interpolates
    the function f at n points.

    Parameters:
        f (function): Function to be interpolated.
        n (int): Number of points at which to interpolate.

    Returns:
        coeffs ((n+1,) ndarray): Chebyshev coefficients for the interpolating polynomial.
    """
    #Calculate the zeros
    y=np.cos((np.pi*np.arange(2*n))/n)
    samples = f(y)

    #calculate coeffs from zeros
    coeffs = np.real(np.fft.fft(samples))[:n+1]/n
    coeffs[0] /= 2
    coeffs[n] /= 2
    return coeffs

# Problem 7
def prob7(n):
    """Interpolate the air quality data found in airdata.npy using
    Barycentric Lagrange interpolation. Plot the original data and the
    interpolating polynomial.

    Parameters:
        n (int): Number of interpolating points to use.
    """
    #load data
    data = np.load("airdata.npy")

    #initialize
    fx = lambda a, b, n: .5*(a+b + (b-a) * np.cos(np.arange(n+1) * np.pi / n))
    a, b = 0, 366 - 1/24
    domain = np.linspace(0, b, 8784)
    points = fx(a, b, n)

    #get data close to interpolating points
    temp = np.abs(points - domain.reshape(8784, 1))
    temp2 = np.argmin(temp, axis=0)

    #interpolate
    poly = BI(domain[temp2], data[temp2])

    #PLOT
    plt.plot(domain,data)
    plt.plot(domain,poly(domain))
    plt.show()
