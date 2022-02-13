# 2d-coulombgas
A Monte Carlo implementation of the 2D Coulomb gas model using the Metropolis algorithm. Both periodic and open boundary conditions are allowed, the latter for mirror configurations used with the method of images.

The 2D Coulomb gas (2D CG) is a two-dimensional lattice (a grid with dimension Lx by Ly) where each lattice site is populated by a electrical charge. The charges can have values 0, -1, 1, -2, 2, and so on. To each configuration, one can calculate an energy H which is calculated by the Coulomb potential in 2D (proportional to <img src="https://render.githubusercontent.com/render/math?math={\color{grey} -\ln(r)}"> where <img src="https://render.githubusercontent.com/render/math?math={\color{gray}r}"> is the distance).

The Metropolis algorithm simulates this model in a thermal equilibrium where the probability of each configuration is proportional to its Boltzmann weight exp(-\beta H). A new configuration is randomly selected by trying a pair charge insertion at a location, from a given previous configuration. The general outline of the algorithm is:
1. Generate a new proposed configuration of the lattice model.
2. Calculate the resulting change in energy <img src="https://render.githubusercontent.com/render/math?math={\color{gray}\ \Delta H}"> of the proposed change.
3. Accept if <img src="https://render.githubusercontent.com/render/math?math={\color{gray}\ \Delta H \leq 0}"> or with probability <img src="https://render.githubusercontent.com/render/math?math={\color{gray}\ e^{-\beta \Delta H}}">. Reject otherwise.
4. If accepted, change to the new proposed lattice configuration and update values.
5. Repeat.

Most difficulties in simulating the 2D CG are the energy calculations which are costly and might diverge due to the potential V ~ -ln(r). This implementation does the following:
1. Construct the potential V as a finite Fourier sum with a screening length lambda.
2. Calculate energy differences with optimized derived formulas instead of calculating the total energy.

The mirror configuration with open boundary condition is somewhat more difficult rather than a regular configuration with periodic boundary conditions.
1. All new configurations are mirror images.
2. All energy calculations use corresponding formulas for mirror configurations.
