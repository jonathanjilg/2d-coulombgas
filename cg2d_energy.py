# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 15:57:36 2022

@author: jonat
"""

import numpy as np
from numba import njit # used as a decorator on the functions for maximum efficiency

# This file contains all functions for energy calculations in the 2D CG model.

@njit
def calc_V(Lx,Ly,lambd,x_period_factor = 1,y_period_factor = 1):
    '''
    Returns an array of the 2D screened periodic Coulomb potential
    indexed by (positive) x and y distance. The computation is by a Fourier sum.

            Parameters:
                    Lx (int): The length in x
                    Ly (int): The length in y
                    lambda (float): The London screening length
                    x_period_factor (int): A multiple of system length that determines the Fourier period
                    y_period_factor (int): A multiple of system length that determines the Fourier period

            Returns:
                    V (array): The 2D Coulomb potential. Its shape is [Lx,Ly].
    '''
    lambd2 = 1/lambd**(2)
    V = np.zeros((Lx,Ly))
    
    Nx, Ny = x_period_factor*Lx, y_period_factor*Ly # Period lengths
    vol_fourier = Nx*Ny
    norm = 2*np.pi/vol_fourier
    # Take Fourier sum on each site
    for x in range(Lx):
        for y in range(Ly):
            s = 0 # real part of the Fourier sum
            for nx in range(Nx):
                kx = 2*np.pi*nx/Nx
                for ny in range(Ny):
                    ky = 2*np.pi*ny/Ny
                    s+= np.cos(kx*x + ky*y)/(lambd2 + 4*np.sin(kx/2)**2 + 4*np.sin(ky/2)**2)
            V[x,y] = norm*s
    return V

@njit
def calc_H(lattice,V):
    '''
    Returns the total energy H of a lattice configuration given a potential V
    by summing over all pair interactions, including self energy.
    It does not include current energy!

            Parameters:
                    lattice (array): The configuration of charges with shape [Lx,Ly].
                    V (array): The potential with shape [Lx,Ly].

            Returns:
                    H (float): The total energy of the configuration
    '''
    Lx = lattice.shape[0]
    Ly = lattice.shape[1]
    
    H = 0
    for x1 in range(Lx):
        for y1 in range(Ly):
            q1 = lattice[x1,y1]
            for x2 in range(Lx):
                for y2 in range(Ly):
                    q2 = lattice[x2,y2]
                    H += 0.5*q1*q2*V[abs(x2-x1),abs(y2-y1)] # configuration energy
    return H

@njit
def calc_dH_pair(dq,q1,q2,x1,y1,x2,y2,lattice,V):
    '''
    Returns the change in energy dH (float) of insertion of a vortex-antivortex pair
    at site 1 and site 2 given a lattice configuration and potential.

            Parameters:
                    dq (int/float): The change in charge at site (x1,y1) and negative change in charge at site (x2,y2)
                    q1 (int): The initial charge at site (x1,y1)
                    q2 (int): The initial charge at site (x1,y1)
                    x1 (int): The x-coordinate of site 1
                    y1 (int): The y-coordinate of site 1
                    x2 (int): The x-coordinate of site 2
                    y2 (int): The y-coordinate of site 2
                    lattice (array): The configuration of charges
                    V (array): The potential

            Returns:
                    dH (float): The calculated energy change of the pair insertion
    '''
    Lx = lattice.shape[0]
    Ly = lattice.shape[1]
    
    dH = 0
    dH = dq*(q1-q2+dq)*(V[0,0]-V[np.abs(x2-x1),np.abs(y2-y1)]) # pair energy of adding a vortex pair
    for x in range(Lx):
        for y in range(Ly):
            # Don't consider interaction w/ sites 1-2
            if (x!=x1 or y!=y1) and (x!=x2 or y!=y2):
                q = lattice[x,y]
                dH += q*dq*(V[np.abs(x1-x),np.abs(y1-y)]-V[np.abs(x2-x),np.abs(y2-y)])
    
    return dH

@njit
def calc_dH_single(dq,q1,x1,y1,lattice,V):
    '''
    Returns the change in energy dH (float) of insertion of a single vortex at site 1 given a configuration and potential.

            Parameters:
                    dq (int/float): The change in charge at site (x1,y1)
                    q1 (int): The initial charge at site (x1,y1)
                    x1 (int): The x-coordinate of site 1
                    y1 (int): The y-coordinate of site 1
                    lattice (array): The configuration of charges
                    V (array): The potential

            Returns:
                    dH (float): The calculated energy change of the pair insertion
    '''
    Lx = lattice.shape[0]
    Ly = lattice.shape[1]
    dH = 0
    dH = dq*(q1+0.5*dq)*V[0,0] # self-energy of adding a single vortex
    for x in range(Lx):
        for y in range(Ly):
            # Don't consider interaction w/ sites 1
            if (x!=x1 or y!=y1): #Don't consider self-energy here
                q = lattice[x,y]
                dH += q*dq*V[np.abs(x1-x),np.abs(y1-y)]
    return dH

@njit
def calc_dH_pair_mirror(dq,q1,q2,x1,y1,x2,y2,lattice,V):
    '''
    Returns the change in energy dH (float) of insertion of a vortex-antivortex pair
    at site 1 and site 2 given a lattice configuration and potential.

            Parameters:
                    dq (int/float): The change in charge at site (x1,y1) and negative change in charge at site (x2,y2)
                    q1 (int): The initial charge at site (x1,y1)
                    q2 (int): The initial charge at site (x1,y1)
                    x1 (int): The x-coordinate of site 1
                    y1 (int): The y-coordinate of site 1
                    x2 (int): The x-coordinate of site 2
                    y2 (int): The y-coordinate of site 2
                    lattice (array): The configuration of charges
                    V (array): The potential

            Returns:
                    dH (float): The calculated energy change of the pair insertion
    '''
    Lx = lattice.shape[0]
    Ly = lattice.shape[1]
    
    # Tabulate coordinates for sites 3 & 4 (site 1 & 2 are already given)
    x3, y3 = x1, Ly-y1-1 # mirror image of site 1 in y
    x4, y4 = x2, Ly-y2-1 # mirror image of site 2 in y
    
    # Pairwise tuple distances between sites 1 to 4
    dist_self = (0,0)
    dist12 = (np.abs(x2-x1),np.abs(y2-y1))
    dist13 = (0,np.abs(y3-y1))
    dist14 = (np.abs(x2-x1),np.abs(y4-y1))
    dist23 = dist14 # symmetry
    dist24 = (0,np.abs(y4-y2))
    dist34 = dist12 # symmetry
    
    dH = 0
    
    # Self-energy and pair energy between sites 1-4
    dH += (2 + 2*dq*(q1-q2))*V[dist_self] # self-energy of adding 4 vortex-antivortex pairs at location
    dH += ((-q1+q2)*dq-1)*V[dist12]
    dH += (-2*q1*dq-1)*V[dist13]
    dH += ((q1-q2)*dq+1)*V[dist14]
    dH += ((-q2+q1)*dq+1)*V[dist23]
    dH += (2*q2*dq-1)*V[dist24]
    dH += ((-q1+q2)*dq-1)*V[dist34]
    
    # Sum all other energy contributions
    s = 0
    for x in range(Lx):
        for y in range(Ly):
            # Don't consider interaction w/ sites 1-4
            if (x!=x1 or y!=y1) and (x!=x2 or y!=y2) and (x!=x3 or y!=y3) and (x!=x4 or y!=y4):
                q = lattice[x,y]
                disti1 = (np.abs(x1-x),np.abs(y1-y))
                disti2 = (np.abs(x2-x),np.abs(y2-y))
                disti3 = (np.abs(x3-x),np.abs(y3-y))
                disti4 = (np.abs(x4-x),np.abs(y4-y))
                s += q*(V[disti1] - V[disti2] - V[disti3] + V[disti4]) 
    dH += s*dq
    return dH

@njit
def calc_dH_single_mirror(dq,q1,x1,y1,lattice,V):
    '''
    Returns the change in energy dH (float) of insertion of a single vortex at site 1 given a configuration and potential.

            Parameters:
                    dq (int/float): The change in charge at site (x1,y1)
                    q1 (int): The initial charge at site (x1,y1)
                    x1 (int): The x-coordinate of site 1
                    y1 (int): The y-coordinate of site 1
                    lattice (array): The configuration of charges
                    V (array): The potential

            Returns:
                    dH (float): The calculated energy change of the pair insertion
    '''
    Lx = lattice.shape[0]
    Ly = lattice.shape[1]
    
    x2, y2 = x1, Ly-y1-1
    dH = 0
    dH = dq*(2*q1+dq)*(V[0,0]-V[0,y2-y1]) # pair energy of adding a vortex pair
    for x in range(Lx):
        for y in range(Ly):
            # Don't consider interaction w/ sites 1-2
            if (x!=x1 or y!=y1) and (x!=x2 or y!=y2):
                q = lattice[x,y]
                dH += q*dq*(V[np.abs(x1-x),np.abs(y1-y)]-V[np.abs(x2-x),np.abs(y2-y)])
    return dH