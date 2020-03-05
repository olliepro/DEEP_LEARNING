# condition_stability.py
"""Volume 1: Conditioning and Stability.
<Name>
<Class>
<Date>
"""

import numpy as np
import sympy as sy
import scipy.linalg as la
from matplotlib import pyplot as plt

# Problem 1
def matrix_cond(A):
    """Calculate the condition number of A with respect to the 2-norm."""
    #get singular values
    s = la.svdvals(A)

    #if min eigval is 0, return inf
    if s[-1]> 1e-10:
        return s[0]/s[-1]
    else: return np.inf

def test_matrix_cond():
    #get orthonormal matrix for test
    Q = np.linalg.qr(np.random.randint(0,5,(2,2)))[0]
    print(np.linalg.cond(Q))
    print(matrix_cond(Q))
    print(matrix_cond(np.array([[1,2],[1,2]])))

# Problem 2
def prob2():
    """Randomly perturb the coefficients of the Wilkinson polynomial by
    replacing each coefficient c_i with c_i*r_i, where r_i is drawn from a
    normal distribution centered at 1 with standard deviation 1e-10.
    Plot the roots of 100 such experiments in a single figure, along with the
    roots of the unperturbed polynomial w(x).

    Returns:
        (float) The average absolute condition number.
        (float) The average relative condition number.
    """
    #initialize standard roots
    w_roots = np.arange(1, 21)

    # Get the exact Wilkinson polynomial coefficients using SymPy.
    x, i = sy.symbols('x i')
    w = sy.poly_from_expr(sy.product(x-i, (i, 1, 20)))[0]
    w_coeffs = np.array(w.all_coeffs())
    #initialize
    new_roots = []
    perturbations = []

    #run 100 pertubations checking roots at each step
    for _ in range(100):
        p = np.random.normal(1,1e-10,len(w_coeffs))
        perturbations.append(p)
        new_coeffs = w_coeffs * p
        # Use NumPy to compute the roots of the perturbed polynomial.
        new_roots.append(np.roots(np.poly1d(new_coeffs)))

    #PLOT EVERYTHING
    plt.scatter(np.real(new_roots),np.imag(new_roots),marker = '$.$',s=2.3,label = 'perturbed')
    plt.scatter(w_roots,np.zeros_like(w_roots),marker = 'o',label = 'unperturbed')
    plt.legend()
    plt.title("True wilkinson polynomial roots and slightly perturbed roots")
    plt.show()
    #Calculate the conditioning numbers with inf norm
    k= la.norm(new_roots - w_roots, np.inf) / la.norm(perturbations, np.inf)
    return k,k * la.norm(w_coeffs, np.inf) / la.norm(w_roots, np.inf)

# Problem 3
def eig_cond(A):
    """Approximate the condition numbers of the eigenvalue problem at A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) The absolute condition number of the eigenvalue problem at A.
        (float) The relative condition number of the eigenvalue problem at A.
    """
    #calculate eigenvalues
    A_eigs = la.eigvals(A)
    #calc norms with 2 norm
    A_eigs_norm = la.norm(A_eigs)
    A_norm = la.norm(A)
    k_hat,k=0,0

    #average a bunch of pertubations to get most accurate condition number
    for _ in range(100):
        #perturb by imag #
        reals = np.random.normal(0, 1e-10, A.shape)
        imags = np.random.normal(0, 1e-10, A.shape)
        H = reals + 1j*imags
        k_hat_ = la.norm(A_eigs - la.eigvals(A + H)) / la.norm(H)
        k_hat += k_hat_
        k += k_hat_ * A_eigs_norm / A_norm
    return k_hat/100,k/100

def test_eig_cond():
    print(eig_cond(np.array([[1,2],[1,2.0000001]])))

# Problem 4
def prob4(domain=[-100, 100, -100, 100], res=200):
    """Create a grid [x_min, x_max] x [y_min, y_max] with the given resolution. For each
    entry (x,y) in the grid, find the relative condition number of the
    eigenvalue problem, using the matrix   [[1, x], [y, 1]]  as the input.
    Use plt.pcolormesh() to plot the condition number over the entire grid.

    Parameters:
        domain ([x_min, x_max, y_min, y_max]):
        res (int): number of points along each edge of the grid.
    """
    #initialize
    x0,x1,y0,y1=domain
    x = np.linspace(x0,x1,res)
    y = np.linspace(y0,y1,res)
    X,Y = np.meshgrid(x,y)
    J = np.empty_like(X)

    #iterate through domain
    for i in range(res):
        for j in range(res):
            #initialize matrix and true eigs
            M = np.array([[1, X[i,j]],[Y[i,j],1]])
            eigs = la.eigvals(M)

            #repeat with perturbations
            perturb = np.random.normal(0, 1e-6, M.shape) + np.random.normal(0,1e-6, M.shape)*1j
            eigsp = la.eigvals(M+perturb)

            #calculate the condition numbers
            k = la.norm(eigs-eigsp)/la.norm(perturb)
            J[i,j] = k*la.norm(M)/la.norm(eigs)

    #plot everything
    plt.pcolormesh(X,Y,J, cmap='gray_r')
    plt.title("Condition numbers for symmetricly varied matricies")
    plt.colorbar()
    plt.show()

# Problem 5
def prob5(n):
    """Approximate the data from "stability_data.npy" on the interval [0,1]
    with a least squares polynomial of degree n. Solve the least squares
    problem using the normal equation and the QR decomposition, then compare
    the two solutions by plotting them together with the data. Return
    the mean squared error of both solutions, ||Ax-b||_2.

    Parameters:
        n (int): The degree of the polynomial to be used in the approximation.

    Returns:
        (float): The forward error using the normal equations.
        (float): The forward error using the QR decomposition.
    """
    #get data
    xk, yk = np.load("stability_data.npy").T
    #initialize and run for n+1 degree least squares
    A = np.vander(xk, n+1)
    coeffs_unstable = np.dot(la.inv(np.dot(A.T,A)),np.dot(A.T,yk))

    #initialize and run QR method for solving system
    Q,R = la.qr(A,mode='economic')
    coeffs_stable = np.dot(la.inv(R),np.dot(Q.T,yk))

    #PLOT
    domain = np.linspace(0,1,100)
    domain_ = np.vander(domain,n+1)
    plt.plot(domain,np.dot(domain_,coeffs_stable),color = "orange", label = "QR",lw=2)
    plt.plot(domain,np.dot(domain_,coeffs_unstable),color = "red",label = 'Normal Eqts',lw = 2)
    plt.scatter(xk,yk,label = "Data",alpha = .5)
    plt.title("Unstability of Normal Equations for Least Squares approximation")
    plt.legend()
    plt.show()

# Problem 6
def prob6():
    """For n = 5, 10, ..., 50, compute the integral I(n) using SymPy (the
    true values) and the subfactorial formula (may or may not be correct).
    Plot the relative forward error of the subfactorial formula for each
    value of n. Use a log scale for the y-axis.
    """
    #initialize
    x,n,k = sy.symbols('x_,n,k')
    summand = (-1)**k/sy.factorial(k)
    approx = (-1)**n*(sy.subfactorial(n)-sy.factorial(n)/sy.E)
    I_n = sy.integrate(x**n*sy.exp(x-1.),(x,0,1))

    #for values of n, evaluate integral
    domain, truth,approxs = list(range(5,51,5)),[],[]
    for N in domain:
        truth.append(float(I_n.subs(n,N)))
        approxs.append(float(approx.subs(n,N)))

    #PLOT ERROR
    plt.yscale('log')
    plt.plot(domain,np.abs(np.array(truth)-np.array(approxs)))
    plt.title("Forward Error for subfactorial method of calculating I(n)")
    plt.xlabel("n")
    plt.ylabel("Error")
    plt.show()
