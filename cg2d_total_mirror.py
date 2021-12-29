# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 13:47:20 2021

@author: jonat
"""

import numpy as np
from numba import njit

@njit
def set_up_config(Lx,Ly):
    L = np.maximum(Lx,Ly)
    lambd = 2*L
    lattice = np.zeros((Lx,2*Ly))
    V = calc_V(Lx,2*Ly,lambd)
    H = 0
    N = 0
    return lattice, V, H, N

@njit
def calc_V(Lx,Ly,lambd,x_period_factor = 1,y_period_factor = 1):
    '''
    Returns an array of the 2D screened periodic Coulomb potential depending on distance.

            Parameters:
                    Lx (int): The length in x
                    Ly (int): The length in y
                    lambda (float): The London screening length
                    x_period_factor (int): A multiple of system length that determines the Fourier period
                    y_period_factor (int): A multiple of system length that determines the Fourier period

            Returns:
                    calc_V (np array): Lx by Ly numpy array.
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
def calc_H(J,lattice,V):
    '''
    Returns the total energy H (float) of a lattice configuration given a potential.

            Parameters:
                    J (float): Applied current
                    lattice (array): The configuration of charges
                    V (array): The potential

            Returns:
                    calc_H (float): The total energy of the configuration
    '''
    Lx = lattice.shape[0]
    Ly = lattice.shape[1]
    
    H = 0
    for x1 in range(Lx):
        for y1 in range(Ly):
            q1 = lattice[x1,y1]
            if y1 < Ly//2:
                H += J*q1*y1
            for x2 in range(Lx):
                for y2 in range(Ly):
                    q2 = lattice[x2,y2]
                    H += 0.5*q1*q2*V[abs(x2-x1),abs(y2-y1)] # configuration energy
    return H

@njit
def calc_dH_pair(dq,q1,q2,J,x1,y1,x2,y2,lattice,V):
    '''
    Returns the change in energy dH (float) of insertion of a vortex-antivortex pair at site 1 and site 2 given a lattice configuration and potential.

            Parameters:
                    dq (int/float): The change in charge at site (x1,y1) and negative change in charge at site (x2,y2)
                    q1 (int): The initial charge at site (x1,y1)
                    q2 (int): The initial charge at site (x1,y1)
                    J (float): The current coupling
                    x1 (int): The x-coordinate of site 1
                    y1 (int): The y-coordinate of site 1
                    x2 (int): The x-coordinate of site 2
                    y2 (int): The y-coordinate of site 2
                    lattice (array): The configuration of charges
                    V (array): The potential

            Returns:
                    calc_H (float): The total energy of the configuration
    '''
    Lx = lattice.shape[0]
    Ly = lattice.shape[1]
    
    dist_self = (0,0)
    dist12 = (np.abs(x2-x1),np.abs(y2-y1))
    dist13 = (0,np.abs(Ly-y1-1-y1))
    dist14 = (np.abs(x2-x1),np.abs(Ly-y2-1-y1))
    dist23 = dist14
    dist24 = (0,np.abs(Ly-y2-1-y2))
    dist34 = dist12
    
    dH = 0
    dH += (2 + 2*dq*(q1-q2))*V[dist_self] # self-energy of adding 4 vortex-antivortex pairs at location
    dH += ((-q1+q2)*dq-1)*V[dist12]
    dH += (-2*q1*dq-1)*V[dist13]
    dH += ((q1-q2)*dq+1)*V[dist14]
    dH += ((-q2+q1)*dq+1)*V[dist23]
    dH += (2*q2*dq-1)*V[dist24]
    dH += ((-q1+q2)*dq-1)*V[dist34]
    
    s = 0
    for x in range(Lx):
        for y in range(Ly):
            if (x!=x1 or y!=y1) and (x!=x2 or y!=y2) and (x!=x1 or y!=Ly-y1-1) and (x!=x2 or y!=Ly-y2-1): #Don't consider self-energy here
                q = lattice[x,y]
                disti1 = (np.abs(x1-x),np.abs(y1-y))
                disti2 = (np.abs(x2-x),np.abs(y2-y))
                disti3 = (np.abs(x1-x),np.abs(Ly-y1-1-y))
                disti4 = (np.abs(x2-x),np.abs(Ly-y2-1-y))
                s += q*(V[disti1] - V[disti2] - V[disti3] + V[disti4]) 
    dH += s*dq
    
    # dH = 0
    # dH += 2*dq*(q1-q2+dq)*(V[0,0]-V[np.abs(x2-x1),np.abs(y2-y1)]) # self-energy of adding vortex-antivortex pair at location
    # dH += dq*((2*q1+dq)*(V[np.abs(x1-x2),np.abs(Ly-y1-y2-1)]-V[0,np.abs(Ly-2*y1-1)]) + (2*q2+dq)*(V[np.abs(x1-x2),np.abs(Ly-y1-y2-1)]-V[0,np.abs(Ly-2*y2-1)]))
    # for x in range(Lx):
    #     for y in range(Ly):
    #         if (x!=x1 or y!=y1) and (x!=x2 or y!=y2) and (x!=x2 or y!=Ly-y1-1) and (x!=x2 or y!=Ly-y2-1): #Don't consider self-energy here
    #             q = lattice[x,y]
    #             dH += q*dq*(V[np.abs(x1-x),np.abs(y1-y)]-V[np.abs(x2-x),np.abs(y2-y)] - V[np.abs(x1-x),np.abs(Ly-y1-1-y)] + V[np.abs(x2-x),np.abs(Ly-y2-1-y)])
    
    dH += J*(y1-y2)*dq # current coupling
    
    return dH

@njit
def calc_dH_single(dq,q1,J,x1,y1,lattice,V):
    '''
    Returns the change in energy dH (float) of insertion of a single vortex at site 1 given a configuration and potential.

            Parameters:
                    dq (int/float): The change in charge at site (x1,y1)
                    q1 (int): The initial charge at site (x1,y1)
                    J (float): The current coupling
                    x1 (int): The x-coordinate of site 1
                    y1 (int): The y-coordinate of site 1
                    lattice (array): The configuration of charges
                    V (array): The potential

            Returns:
                    calc_H (float): The total energy of the configuration
    '''
    Lx = lattice.shape[0]
    Ly = lattice.shape[1]
    
    dH = 0
    dH = dq*(2*q1+dq)*(V[0,0]-V[0,Ly-2*y1-1]) # self-energy of adding a single vortex at location
    for x in range(Lx):
        for y in range(Ly):
            if (x!=x1 or y!=y1) and (x!=x1 or y!=Ly-y1-1): #Don't consider self-energy here
                q = lattice[x,y]
                dH += q*dq*(V[np.abs(x1-x),np.abs(y1-y)]-V[np.abs(x1-x),np.abs(Ly-y1-1-y)])
    dH += J*dq*y1 # current coupling
    return dH

@njit
def sweep(J,lattice,V,T,H,N):
    '''
    Performs a Monte Carlo sweep of the 2D Coulomb gas model.

            Parameters:
                    J (float): The current coupling
                    lattice (array): The configuration of charges
                    V (array): The potential
                    T (float): Temperature
            Returns:
                    sweep (numpy array): The sweeped lattice
    '''
    Lx,Ly_tot = lattice.shape
    Ly = Ly_tot//2
    interval_y = range(Ly)
    for step in range(Lx*Ly):
        # 1. Select at random a neighbouring pair in the lattice or a single site if the selected neighbour is outside the lattice.
        # 1a. Select site 1
        x1,y1 = np.random.randint(Lx),np.random.randint(Ly)
        # 1b. Select site 2
        x2,y2 = x1,y1
        d = np.random.randint(4)
        if d == 0:
            x2 = (x2 + 1) % Lx
        elif d == 1:
            x2 = (x2 - 1) % Lx
        elif d == 2:
            y2 = y2 + 1
        elif d == 3:
            y2 = y2 -1
        else:
            raise Exception('Direction does not exist')
        # 1c. Check if neighbour is in the lattice
        if y2 in interval_y: # if neighbour is in lattice - pair move
            # 2. Get charges of the neighbouring pair
            q1,q2 = lattice[x1,y1],lattice[x2,y2]
            dq = 2*np.random.randint(2)-1

            # 3. Calculate energy change
            dH = calc_dH_pair(dq,q1,q2,J,x1,y1,x2,y2,lattice,V)

            # 4. Metropolis acceptance test
            if np.random.rand() < np.exp(-dH/T):
                # 5. Update quantities
                lattice[x1,y1],lattice[x2,y2] = q1+dq,q2-dq
                lattice[x1,Ly_tot-y1-1],lattice[x2,Ly_tot-y2-1] = -q1-dq,-q2+dq
                H += dH
                N += np.abs(q1+dq) + np.abs(q2-dq) - np.abs(q1)  - np.abs(q2)
        else: # if neighbour is not in lattice - single move
            # 2. Get the charge of the single site
            q1 = lattice[x1,y1]
            dq = 2*np.random.randint(2)-1
            # 3. Calculate energy change
            dH = calc_dH_single(dq,q1,J,x1,y1,lattice,V)

            # 4. Metropolis acceptance test
            if np.random.rand() < np.exp(-dH/T):
                # 5. Update quantities
                lattice[x1,y1] = q1+dq
                lattice[x1,Ly_tot-y1-1] = -q1-dq
                H += dH
                N += np.abs(q1+dq) - np.abs(q1)
    return lattice, H, N

@njit
def sweep_god_jul(J,lattice,V,T):
    '''
    Performs a Monte Carlo sweep of the 2D Coulomb gas model.

            Parameters:
                    J (float): The current coupling
                    lattice (array): The configuration of charges
                    V (array): The potential
                    T (float): Temperature
            Returns:
                    sweep (numpy array): The sweeped lattice
    '''
    Lx,Ly_tot = lattice.shape
    dq = 2*np.random.randint(2)-1
    
    coord_list= [(1,0),(2,0),(0,1),(3,1),(0,2),(2,2),(3,2),(0,4),(1,4),(2,4),(3,4),(0,5),(3,5),(0,6),(1,6),(2,6),(3,6),(0,8),(1,8),(2,8),(3,8),(0,9),(3,9),(1,10),(2,10),(5,0),(8,0),(5,1),(6,1),(7,1),(8,1),(5,3),(6,3),(7,3),(8,3),(8,4),(5,5),(6,5),(7,5),(8,5),(5,7),(6,7),(7,7),(8,7),(8,8),(10,0),(11,0),(12,0),(13,0),(11,1),(12,2),(11,3),(10,4),(11,4),(12,4),(13,4),(11,6),(12,6),(13,6),(10,7),(12,7),(11,8),(12,8),(13,8),(10,10),(10,11),(11,11),(12,11),(13,11),(10,12),(10,14),(11,14),(13,14),(10,15),(12,15),(13,15)]
    
    for x1, y1 in coord_list:
        lattice[x1,y1] = dq
        lattice[x1,Ly_tot-y1-1] = -dq

    return lattice

@njit
def sweep_animate(J,lattice,V,T):
    '''
    Performs a Monte Carlo sweep of the 2D Coulomb gas model.

            Parameters:
                    J (float): The current coupling
                    lattice (array): The configuration of charges
                    V (array): The potential
                    T (float): Temperature
            Returns:
                    sweep (numpy array): The sweeped lattice
    '''
    Lx,Ly_tot = lattice.shape
    Ly = Ly_tot//2
    interval_y = range(Ly)
    for step in range(Lx*Ly):
        # 1. Select at random a neighbouring pair in the lattice or a single site if the selected neighbour is outside the lattice.
        # 1a. Select site 1
        x1,y1 = np.random.randint(Lx),np.random.randint(Ly)
        # 1b. Select site 2
        x2,y2 = x1,y1
        d = np.random.randint(4)
        if d == 0:
            x2 = (x2 + 1) % Lx
        elif d == 1:
            x2 = (x2 - 1) % Lx
        elif d == 2:
            y2 = y2 + 1
        elif d == 3:
            y2 = y2 -1
        else:
            raise Exception('Direction does not exist')
        # 1c. Check if neighbour is in the lattice
        if y2 in interval_y: # if neighbour is in lattice - pair move
            # 2. Get charges of the neighbouring pair
            q1,q2 = lattice[x1,y1],lattice[x2,y2]
            dq = 2*np.random.randint(2)-1

            # 3. Calculate energy change
            dH = calc_dH_pair(dq,q1,q2,J,x1,y1,x2,y2,lattice,V)

            # 4. Metropolis acceptance test
            if np.random.rand() < np.exp(-dH/T):
                # 5. Update quantities
                lattice[x1,y1],lattice[x2,y2] = q1+dq,q2-dq
                lattice[x1,Ly_tot-y1-1],lattice[x2,Ly_tot-y2-1] = -q1-dq,-q2+dq
        else: # if neighbour is not in lattice - single move
            # 2. Get the charge of the single site
            q1 = lattice[x1,y1]
            dq = 2*np.random.randint(2)-1
            # 3. Calculate energy change
            dH = calc_dH_single(dq,q1,J,x1,y1,lattice,V)

            # 4. Metropolis acceptance test
            if np.random.rand() < np.exp(-dH/T):
                # 5. Update quantities
                lattice[x1,y1] = q1+dq
                lattice[x1,Ly_tot-y1-1] = -q1-dq
    return lattice