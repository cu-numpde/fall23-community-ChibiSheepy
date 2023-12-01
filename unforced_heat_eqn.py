# 2D Unforced Heat Equation
# ======================
#
# In this demo, we explore the heat equation, first simulating it on a
# two dimensional square domain. The heat equation describes how heat 
# diffuses over time, first, in 2d:
# .. math::
#
#    \frac{\partial u}{\partial t} = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)
#
# where \( u(x, y, t) \) is the temperature at location \((x, y)\) and time \(t\),
# and \( \alpha \) is the thermal diffusivity of the material.
#
# The heat equation can also be forced, which will be shown later in the demo.
# We use Dirichlet boundary conditions with the temperature held constant
# at the boundaries of the domain. The initial condition is a peak in temperature
# at the center of the domain.
#
# The numerical solution is obtained using the Firedrake library, and the results
# are partially visualized using Matplotlib. Full time-stepped animations can be 
# viewed in ParaView.

from firedrake import *
import matplotlib.pyplot as plt
import os

# Parameters
alpha = 1.0e-1       # thermal diffusivity
dt = 0.01         # time step size
time_end = 1.0    # end time
nx, ny = 50, 50   # number of cells in x and y directions

# Create a 2D square mesh
mesh = UnitSquareMesh(nx, ny)

# Define function space - continuous Galerkin space of degree 1
V = FunctionSpace(mesh, "CG", 1)

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Initial condition: peak in the center of the domain
u_n = Function(V)
x, y = SpatialCoordinate(mesh)
u_n.interpolate(exp(-((x - 0.5)**2 + (y - 0.5)**2)/0.02))

# Boundary condition: temperature is zero on all boundaries
bc = DirichletBC(V, 0.0, "on_boundary")

# Variational problem
F = (u - u_n) * v * dx + dt * alpha * (dot(grad(u), grad(v))) * dx

# Create bilinear and linear forms
a, L = lhs(F), rhs(F)

# Create a subfolder for output files
output_folder = "unforced_heat_eqn_output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Setup file for exporting results
vtkfile = File(os.path.join(output_folder,'heat_diffusion_no_forcing.pvd'))

# Time-stepping
u = Function(V)
t = 0
while t < time_end:
    # Solve the variational problem
    solve(a == L, u, bcs=[bc])

    # Update previous solution
    u_n.assign(u)

    # Increment time
    t += dt

    vtkfile.write(u_n, time=t)

# Export results (for visualization)
vtkfile.write(u_n)

# Matplotlib Pseudo-color Plot
fig, axes = plt.subplots()
colors = tripcolor(u, axes=axes)
fig.colorbar(colors)
plt.show()

# Matplotlib Contour Plot
fig, axes = plt.subplots()
contours = tricontour(u, axes=axes)
fig.colorbar(contours)
plt.show()

# As is, this only shows one time-step as proof of concept, so need ParaView
