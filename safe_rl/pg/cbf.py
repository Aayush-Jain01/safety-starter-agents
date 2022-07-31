#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 14:17:49 2018
@author: rcheng
"""

import numpy as np
from cvxopt import matrix
from cvxopt import solvers
import dynamics_gp

#Build barrier function model
def build_barrier(self):
    N = self.action_size
    #self.P = matrix(np.eye(N), tc='d')
    self.P = matrix(np.diag([2, 2]), tc='d')
    self.q = matrix(np.zeros(N+1))
    self.H1 = np.array([1, 1])
    self.H2 = np.array([1, -1])
    self.H3 = np.array([-1, 1])
    self.H4 = np.array([-1, -1])

#Get compensatory action based on satisfaction of barrier function
def control_barrier(self, obs, u_rl, f, g, control_bound, distance, r1, r2):
    vel = f[0]
    #Set up Quadratic Program to satisfy Control Barrier Function
    G = np.array([[-np.dot(self.H1,g), -np.dot(self.H2,g), -np.dot(self.H3,g), -np.dot(self.H4,g), 1, -1, g[1], -g[1]], [-1, -1, -1, -1, 0, 0, 0, 0]])
    G = np.transpose(G)
    h = np.array([u_rl + 2*vel + distance - (r1 + r2),
                  u_rl + 2*vel + distance - (r1 + r2),
                  u_rl + 2*vel + distance - (r1 + r2),
                  u_rl + 2*vel + distance - (r1 + r2),
                  -u_rl + control_bound,
                  u_rl + control_bound,
                  -f[1] - g[1]*u_rl + self.max_speed,
                  f[1] + g[1]*u_rl + self.max_speed])
    h = np.squeeze(h).astype(np.double)
    
    #Convert numpy arrays to cvx matrices to set up QP
    G = matrix(G,tc='d')
    h = matrix(h,tc='d')

    solvers.options['show_progress'] = False
    sol = solvers.qp(self.P, self.q, G, h)
    u_bar = sol['x']

    if (np.add(np.squeeze(u_rl), np.squeeze(u_bar[0])) - 0.001 >= control_bound):
        u_bar[0] = control_bound - u_rl
        print("Error in QP")
    elif (np.add(np.squeeze(u_rl), np.squeeze(u_bar[0])) + 0.001 <= -control_bound):
        u_bar[0] = -control_bound - u_rl
        print("Error in QP")
    else:
        pass

    return np.expand_dims(np.array(u_bar[0]), 0)