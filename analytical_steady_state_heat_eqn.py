# Heat Equation with Analytical Comparison
# In this part, we solve a steady-state heat equation with a sinusoidal
# forcing function and compare the numerical solution with the known
# analytical solution. This comparison is important to verify the accuracy
# of the numerical method.
#
# The steady-state heat equation is given by:
#
# .. math::
#
#    -\Delta u = f
#
# on a unit square domain with zero Dirichlet boundary conditions.
# The forcing function \( f \) and its corresponding analytical solution are:
#
# .. math::
#
#    f(x, y) = 2\pi^2 \sin(\pi x) \sin(\pi y)
#    u_{\text{analytic}}(x, y) = \sin(\pi x) \sin(\pi y)
#
# After solving the equation numerically, we compute the \( L_2 \) norm
# of the error between the numerical and analytical solutions.

from firedrake import *
import matplotlib.pyplot as plt
import os

# Create a mesh
mesh = UnitSquareMesh(50, 50)

# Define function space
V = FunctionSpace(mesh, "CG", 2)

# Define boundary condition
bc = DirichletBC(V, 0, "on_boundary")

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Function(V)
x, y = SpatialCoordinate(mesh)
f.interpolate(2*pi**2*sin(pi*x)*sin(pi*y))
a = dot(grad(u), grad(v)) * dx
L = f * v * dx

# Compute solution
u = Function(V)
solve(a == L, u, bcs=bc)

# Define analytical solution
u_analytic = Function(V)
u_analytic.interpolate(sin(pi*x)*sin(pi*y))

# Compute the L2 norm of the error
error = sqrt(assemble(dot(u - u_analytic, u - u_analytic) * dx))
print(f"L2 Norm of the Error: {error}")
