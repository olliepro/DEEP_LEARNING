# solutions.py
"""Volume 1: The SVD and Image Compression."""
import numpy as np
from numpy import linalg as la
import scipy
import matplotlib.pyplot as plt

# Problem 1
def compact_svd(A,k=None, tol=1e-6):
    """Compute the truncated SVD of A.

    Parameters:
        A ((m,n) ndarray): The matrix (of rank r) to factor.
        tol (float): The tolerance for excluding singular values.

    Returns:
        ((m,r) ndarray): The orthonormal matrix U in the SVD.
        ((r,) ndarray): The singular values of A as a 1-D array.
        ((r,n) ndarray): The orthonormal matrix V^H in the SVD.
    """
    AHA = A.conj().T@A                          # Compute A_H @ A

    eig_vals,V = la.eigh(AHA)                   # Compute Eigenvalues and associated vectors
    eig_vals[eig_vals<tol] = 0
    single_vals = np.sqrt(eig_vals)             # Compute Singular Vals for Eigenvalues

    V = V[:,::-1]                               # Reorder Eigenvalues and Eigenvectors
    single_vals = np.sort(single_vals)[::-1]

    r = sum(single_vals>tol)                    # Count # of singular Vals > tol
    single_vals = single_vals[:r]               # Keep only non-zero vals
    V = V[:,:r]                                 # keep only corresponding Eigenvectors

    U = A @ V / single_vals                     # create U
    return U, single_vals, V.conj().T

# Problem 2
def visualize_svd(A):
    """Plot the effect of the SVD of A as a sequence of linear transformations
    on the unit circle and the two standard basis vectors.
    """
    #CREATE CIRCLE and BASIS VECTORS
    circle = np.array([[np.cos(theta) for theta in np.linspace(0,2*np.pi,200)],[np.sin(theta) for theta in np.linspace(0,2*np.pi,200)]])
    E = np.array([[1,0,0],[0,0,1]])

    #CALC SVD
    U,sigma,V_H = la.svd(A)
    sigma = np.diag(sigma)

    #CREATE PLOTS
    plt.suptitle("Plots of Circle (S) Transformations by SVD of A", fontsize = 18)
    plt.subplot(221)
    plt.title("S")
    plt.plot(circle[0],circle[1])
    plt.plot(E[0],E[1])
    plt.subplot(222)
    plt.title("V.H @ S")
    coords = V_H@circle
    Ecoords = V_H@E
    plt.plot(coords[0],coords[1])
    plt.plot(Ecoords[0],Ecoords[1])
    plt.subplot(223)
    plt.title("Σ @ V.H @ S")
    plt.ylim(-1,1)
    coords = sigma@V_H@circle
    Ecoords = sigma@V_H@E
    plt.plot(coords[0],coords[1])
    plt.plot(Ecoords[0],Ecoords[1])
    plt.subplot(224)
    coords = U@sigma@V_H@circle
    Ecoords = U@sigma@V_H@E
    plt.title("U @ Σ @ V.H @ S")
    plt.plot(coords[0],coords[1])
    plt.plot(Ecoords[0],Ecoords[1])
    plt.show()

# Problem 3
def svd_approx(A, s):
    """Return the best rank s approximation to A with respect to the 2-norm
    and the Frobenius norm, along with the number of bytes needed to store
    the approximation via the truncated SVD.

    Parameters:
        A ((m,n), ndarray)
        s (int): The rank of the desired approximation.

    Returns:
        ((m,n), ndarray) The best rank s approximation of A.
        (int) The number of entries needed to store the truncated SVD.
    """
    #Caclulate SVD
    U, sigma, V_H = la.svd(A, full_matrices=False)

    #Check Valid Compression
    if s > len(sigma):raise ValueError("s cannot be greater than rank(A)")

    #truncate U, Σ, V.H
    U = U[:,:s]
    sigma = sigma[:s]
    V_H = V_H[:s]

    entries = sum((U.size,sigma.size,V_H.size))
    return U@np.diag(sigma)@V_H, entries

# Problem 4
def lowest_rank_approx(A, err):
    """Return the lowest rank approximation of A with error less than 'err'
    with respect to the matrix 2-norm, along with the number of bytes needed
    to store the approximation via the truncated SVD.

    Parameters:
        A ((m, n) ndarray)
        err (float): Desired maximum error.

    Returns:
        A_s ((m,n) ndarray) The lowest rank approximation of A satisfying
            ||A - A_s||_2 < err.
        (int) The number of entries needed to store the truncated SVD.
    """
    #Caclulate SVD
    U,sigma,V_H = la.svd(A, full_matrices=False)

    #Check Valid Compression
    if err <= sigma[-1]: raise ValueError("error bound too low for matrix approximation")

    #Get Truncate Point
    k = np.where(sigma<err)[0][0]

    #Truncate
    U = U[:,:k]
    sigma = sigma[:k]
    V_H = V_H[:k]

    entries = sum((U.size,sigma.size,V_H.size))
    return U@np.diag(sigma)@V_H, entries

# Problem 5
def compress_image(filename, s):
    """Plot the original image found at 'filename' and the rank s approximation
    of the image found at 'filename.' State in the figure title the difference
    in the number of entries used to store the original image and the
    approximation.

    Parameters:
        filename (str): Image file path.
        s (int): Rank of new image.
    """
    #Initialize Images to Display
    OG = plt.imread(filename).astype(float)
    img = np.zeros(OG.shape)
    color = False

    #check Dim for color
    if len(OG.shape) == 3:
        color = True
        r = OG[:,:,0]
        g = OG[:,:,1]
        b = OG[:,:,2]
        #set truncated color segments
        img[:,:,0],num_r = svd_approx(r,s)
        img[:,:,1],num_g = svd_approx(g,s)
        img[:,:,2],num_b = svd_approx(b,s)
        num = num_b+num_g+num_r
    else:
        #calculate compressed img
        img,num = svd_approx(OG,s)

    #bring in values to be in displayable range
    img = np.round(img)/255
    OG = np.round(OG)/255
    img[img<0] = 0.
    img[img>1] = 1.

    #PLOT IMAGES
    plt.suptitle("Prob 5: Entries Saved = {}".format(np.prod(img.shape)-num))
    plt.subplot(121)
    plt.title("OG Image")
    if color:
        plt.imshow(OG)
    else:
        plt.imshow(OG, cmap = "gray")
    plt.subplot(122)
    plt.title("Rank " + str(s) + " Approx. of Image")
    if color:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap = "gray")
    plt.show()
