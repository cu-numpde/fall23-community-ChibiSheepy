# 3D Heat Equation on the Surface of a Sphere
# ===========================================
#
# This demo simulates the time-dependent heat equation on the surface of a sphere.
# The heat equation in this context represents how heat (or a similar scalar quantity)
# diffuses over the surface of the sphere over time.
#
# The heat equation on a curved surface like a sphere is given by:
#
# .. math::
#
#    \frac{\partial u}{\partial t} = \alpha \Delta_{\mathbb{S}^2} u + f
#
# where \( u \) is the temperature, \( \alpha \) is the thermal diffusivity,
# \( \Delta_{\mathbb{S}^2} \) is the Laplace-Beltrami operator on the sphere,
# and \( f \) is a forcing function.
#
# We will consider a simple sinusoidal forcing function and use Firedrake
# to solve this PDE on the surface of a sphere. The results will be saved
# for visualization in Paraview.


from firedrake import *
import os

# Parameters
alpha = 1.0       # Thermal diffusivity
dt = 0.05         # Time step size
time_end = 5.0    # End time for simulation

# Create a mesh of a unit sphere
mesh = UnitIcosahedralSphereMesh(refinement_level=2)
x, y, z = SpatialCoordinate(mesh)

# Correcting the orientation of cells
normal = as_vector((x, y, z))
mesh.init_cell_orientations(normal)

# Define function space
V = FunctionSpace(mesh, "CG", 1)

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Initial condition - non-uniform temperature
u_n = Function(V)
u_n.interpolate(100 * sin(pi * x) * cos(pi * y))

# Forcing function - varying over space and time
def forcing_function(t):
    return 100 * sin(-2 * pi * x) * cos(2 * pi * y) * cos(0.01*t)

f = Function(V)

# Variational problem
F = (u - u_n) * v * dx + dt * alpha * dot(grad(u), grad(v)) * dx - dt * f * v * dx
a, L = lhs(F), rhs(F)

# Time-stepping
u = Function(V)

# Create a directory for output files
output_directory = "spherical_heat_eqn_output"
os.makedirs(output_directory, exist_ok=True)

# Create Paraview file for saving results
pvd = File(os.path.join(output_directory, "heat_equation_sphere.pvd"))

# Assign the initial condition to 'u' and write it to the file
u.assign(u_n)
pvd.write(u, time=0)

for t in range(int(time_end / dt)):
    current_time = t / dt

    # Update the forcing function
    f.interpolate(forcing_function(current_time))

    # Solve the variational problem
    solve(a == L, u, bcs=[])

    # Update previous solution
    u_n.assign(u)

    # Write solution to file
    pvd.write(u, time=current_time)
    
# The solution can be visualized using ParaView