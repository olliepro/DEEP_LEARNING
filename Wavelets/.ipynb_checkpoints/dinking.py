import numpy as np
import matplotlib.pyplot as plt
from sympy import dsolve,symbols,lambdify
from sympy import Function, Derivative, Eq
X = np.arange(-3, 3,.3)
T = X
domain = np.meshgrid(T,X)
x_prime = lambda x,t: t/x
U = x_prime(domain[1],domain[0])
V = np.ones(domain[1].shape)
norm = np.linalg.norm(np.vstack((U.flatten(),V.flatten())).T,axis=1).reshape(U.shape)
U /= norm
V /= norm
plt.figure(figsize=(10,10))
plt.quiver(T,X,V,U)
t,C= symbols('t,C')
f = Function("f")(t)
f_ = Derivative(f,t)
eq = Eq(f_, t/f)
print(dsolve(eq))
expr = -(C + t**2)**(1/2)
expr2 = -expr
g = lambdify((C,t),expr)
g2 = lambdify((C,t),expr2)
for c in np.linspace(-2,2,6):
    domain = np.linspace(-3,3,100)
    plt.plot(domain,g(c,domain))
    plt.plot(domain,g2(c,domain))
    plt.ylim(-3,3)
plt.show()

X = np.arange(-3, 3,.3)
T = X
domain = np.meshgrid(T,X)
x_prime = lambda x: 2*x*(1-x/2)
U = x_prime(domain[1])
V = np.ones(domain[1].shape)
norm = np.linalg.norm(np.vstack((U.flatten(),V.flatten())).T,axis=1).reshape(U.shape)
U /= norm
V /= norm
plt.figure(figsize=(10,10))
plt.quiver(T,X,V,U)
t,C= symbols('t,C')
f = Function("f")(t)
f_ = Derivative(f,t)
eq = Eq(f_, 2*f*(1-f/2))
print(dsolve(eq))
expr = 2/(C * np.e**(-2*t) + 1)
g = lambdify((C,t),expr)
for c in np.linspace(-2,2,6):
    domain = np.linspace(-3,3,100)
    plt.plot(domain,g(c,domain))
    plt.ylim(-3,3)
plt.show()
