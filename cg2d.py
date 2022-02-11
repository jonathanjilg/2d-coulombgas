# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 15:33:03 2022

@author: jonat
"""

import numpy as np
from numba import njit # used as a decorator on the functions for maximum efficiency

# Import energy calculations
from cg2d_energy import calc_V, calc_dH_pair, calc_dH_pair_mirror, calc_dH_single_mirror

@njit
def set_up_config(Lx,Ly,lambd_factor = 4, mirror = False):
    '''
    Builds a set-up configuration and returns all quantities you need to do a Monte Carlo sweep.
    It doubles the y length for mirror effects.

            Parameters:
                    Lx (int): The length in x
                    Ly (int): The length in y
                    lambd_factor (int): Determines the screening length lambda as a multiple of L
                    mirror (boolean): If true, it gives a mirror configuration.
            Returns:
                    lattice (array): The starting configuration of the system.
                    V (array): The 2D Coulomb potential.
                    H (float): The total energy of the system.
                    N (int): The total particle number.
                    Qy (float): The particle flow in the y direction.
    '''
    L = np.maximum(Lx,Ly)
    lambd = lambd_factor*L
    if mirror:
        lattice = np.zeros((Lx,2*Ly))
        V = calc_V(Lx,2*Ly,lambd)
    else:
        lattice = np.zeros((Lx,Ly))
        V = calc_V(Lx,Ly,lambd)
    H = 0
    N = 0
    Qy = 0
    return lattice, V, H, N, Qy

@njit
def sweep(current,lattice,V,T,H,N,qy_flow):
    '''
    Performs a Monte Carlo sweep of the 2D Coulomb gas model with a mirror configuration.
    Uses functions calc_dH_pair and calc_dH_single.

            Parameters:
                    current (float): The current coupling
                    lattice (array): The configuration of charges
                    V (array): The potential
                    T (float): Temperature
                    H (float): Energy
                    N (float): Number of particles
            Returns:
                    lattice (array): The sweeped lattice
                    H: The updated energy
                    N: The updated particle number
    '''
    Lx,Ly = lattice.shape
    for step in range(Lx*Ly):
        # 1. Select at random a neighbouring pair in the lattice
        # or a single site if the selected neighbour is outside the lattice.
        # 1a. Select site 1
        x1,y1 = np.random.randint(Lx),np.random.randint(Ly)
        # 1b. Select site 2
        x2,y2 = x1,y1
        d = np.random.randint(4)
        dy = 0
        if d == 0:
            x2 = (x2 + 1) % Lx
        elif d == 1:
            x2 = (x2 - 1) % Lx
        elif d == 2:
            y2 = (y2 + 1) % Ly
            dy = 1
        elif d == 3:
            y2 = (y2 - 1) % Ly
            dy = -1
        else:
            raise Exception('Direction does not exist')
        
        # 2. Get charges of the neighbouring pair
        q1,q2 = lattice[x1,y1],lattice[x2,y2]
        dq = 2*np.random.randint(2)-1

        # 3. Calculate energy change
        dH = calc_dH_pair(dq,q1,q2,x1,y1,x2,y2,lattice,V)
        dH_curr = -current*dy*dq # current coupling

        # 4. Metropolis acceptance test
        if np.random.rand() < np.exp(-(dH+dH_curr)/T):
            # 5. Update quantities
            lattice[x1,y1],lattice[x2,y2] = q1+dq,q2-dq
            H += dH
            N += np.abs(q1+dq) + np.abs(q2-dq) - np.abs(q1)  - np.abs(q2)
            qy_flow += dq*dy
    return lattice, H, N, qy_flow

@njit
def sweep_mirror(current,lattice,V,T,H,N,qy_flow):
    '''
    Performs a Monte Carlo sweep of the 2D Coulomb gas model with a mirror configuration in y.
    Uses functions calc_dH_pair and calc_dH_single.

            Parameters:
                    current (float): The current coupling
                    lattice (array): The configuration of charges
                    V (array): The potential
                    T (float): Temperature
                    H (float): Energy
                    N (float): Number of particles
            Returns:
                    lattice (array): The sweeped lattice
                    H: The updated energy
                    N: The updated particle number
    '''
    Lx,Ly_tot = lattice.shape
    Ly = Ly_tot//2
    interval_y = range(Ly)
    for step in range(Lx*Ly):
        # 1. Select at random a neighbouring pair in the lattice
        # or a single site if the selected neighbour is outside the lattice.
        # 1a. Select site 1
        x1,y1 = np.random.randint(Lx),np.random.randint(Ly)
        # 1b. Select site 2
        x2,y2 = x1,y1
        dy = 0
        d = np.random.randint(4)
        if d == 0:
            x2 = (x2 + 1) % Lx
        elif d == 1:
            x2 = (x2 - 1) % Lx
        elif d == 2:
            y2 = y2 + 1
            dy = 1
        elif d == 3:
            y2 = y2 - 1
            dy = -1
        else:
            raise Exception('Direction does not exist')
        
        # 1c. Check if neighbour is in the lattice
        if y2 in interval_y: # neighbour is in lattice - pair move
            # 2. Get charges of the neighbouring pair
            q1,q2 = lattice[x1,y1],lattice[x2,y2]
            dq = 2*np.random.randint(2)-1

            # 3. Calculate energy change
            dH = calc_dH_pair_mirror(dq,q1,q2,current,x1,y1,x2,y2,lattice,V) # configuration
            dH_curr = -current*dy*dq # current coupling
            # 4. Metropolis acceptance test
            if np.random.rand() < np.exp(-(dH+dH_curr)/T):
                # 5. Update quantities
                lattice[x1,y1],lattice[x2,y2] = q1+dq,q2-dq
                lattice[x1,Ly_tot-y1-1],lattice[x2,Ly_tot-y2-1] = -q1-dq,-q2+dq
                H += dH
                N += np.abs(q1+dq) + np.abs(q2-dq) - np.abs(q1)  - np.abs(q2)
                qy_flow += dq*dy
        else: # neighbour is not in lattice - single move
            # 2. Get the charge of the single site
            q1 = lattice[x1,y1]
            dq = 2*np.random.randint(2)-1
            # 3. Calculate energy change
            dH = calc_dH_single_mirror(dq,q1,current,x1,y1,lattice,V) # configuration
            dH_curr = -current*dy*dq # current coupling

            # 4. Metropolis acceptance test
            if np.random.rand() < np.exp(-(dH+dH_curr)/T):
                # 5. Update quantities
                lattice[x1,y1] = q1+dq
                lattice[x1,Ly_tot-y1-1] = -q1-dq
                H += dH
                N += np.abs(q1+dq) - np.abs(q1)
                qy_flow += dq*dy
    return lattice, H, N, qy_flow