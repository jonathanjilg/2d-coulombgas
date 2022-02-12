# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 00:22:04 2022

@author: jonat
"""
import numpy as np
from cg2d_energy import calc_V, calc_H, calc_dH_pair, calc_dH_single, calc_dH_pair_mirror, calc_dH_single_mirror

def test_pbc():
    Lx,Ly = 16, 16
    lambd = 4*np.maximum(Lx,Ly)
    lattice = np.zeros((Lx,Ly))
    V = calc_V(Lx,Ly,lambd)
    
    for i in range(1000):
        H0 = calc_H(lattice,V)
    
        x1,y1 = np.random.randint(Lx),np.random.randint(Ly)
        x2,y2 = x1,y1
        dy = 0
        d = np.random.randint(4)
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
    
        q1,q2 = lattice[x1,y1],lattice[x2,y2]
        dq = 2*np.random.randint(2)-1
    
        dH = calc_dH_pair(dq, q1, q2, x1, y1, x2, y2, lattice, V)
    
        lattice[x1,y1] += dq
        lattice[x2,y2] += -dq
    
        H1 = calc_H(lattice,V)
        dH_actual = H1 - H0
        err = np.round(dH - dH_actual,10)
        assert err == 0, "Error in pair PBC energy. Should be 0"
    for i in range(1000):
        H0 = calc_H(lattice,V)
    
        # 1a. Select site 1
        x1,y1 = np.random.randint(Lx),np.random.randint(Ly)
    
        # 2. Get charges of the vortex
        q1 = lattice[x1,y1]
        dq = 2*np.random.randint(2)-1
    
        dH = calc_dH_single(dq, q1, x1, y1,lattice, V)
    
        lattice[x1,y1] += dq
    
        H1 = calc_H(lattice,V)
        dH_actual = H1 - H0
        err = np.round(dH - dH_actual,8)
        assert err == 0, "Error in single PBC energy. Should be 0"
def test_obc():
    Lx,Ly = 16, 16
    lambd = 4*np.maximum(Lx,2*Ly)
    lattice = np.zeros((Lx,2*Ly))
    V = calc_V(Lx,2*Ly,lambd)
    
    Ly_tot = 2*Ly
    interval_y = range(Ly)
    for i in range(1000):
        H0 = calc_H(lattice,V)
    
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
        
        if y2 in interval_y: # neighbour is in lattice - pair move
            # Get charges of the neighbouring pair
            q1,q2 = lattice[x1,y1],lattice[x2,y2]
            dq = 2*np.random.randint(2)-1
    
            # Calculate energy change
            dH = calc_dH_pair_mirror(dq,q1,q2,x1,y1,x2,y2,lattice,V) # configuration
            # Update quantities
            lattice[x1,y1],lattice[x2,y2] = q1+dq,q2-dq
            lattice[x1,Ly_tot-y1-1],lattice[x2,Ly_tot-y2-1] = -q1-dq,-q2+dq
        else: # neighbour is not in lattice - single move
            # Get the charge of the single site
            q1 = lattice[x1,y1]
            dq = 2*np.random.randint(2)-1
            # Calculate energy change
            dH = calc_dH_single_mirror(dq,q1,x1,y1,lattice,V) # configuration
    
            lattice[x1,y1] = q1+dq
            lattice[x1,Ly_tot-y1-1] = -q1-dq
    
        H1 = calc_H(lattice,V)
        dH_actual = H1 - H0
        err = np.round(dH - dH_actual,8)
        assert err == 0, "Error in OBC energy. Should be 0"

if __name__ == "__main__":
    test_pbc()
    test_obc()
    print("Everything passed")