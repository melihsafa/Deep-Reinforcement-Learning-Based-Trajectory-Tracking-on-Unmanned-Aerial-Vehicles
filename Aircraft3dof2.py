#Aircraft Dynamics
from argparse import Action
import numpy as np
from scipy.optimize import fsolve
from sympy import sympify
from sympy.geometry import Point2D, Segment2D, Circle

class Aircraft():
    def __init__(self,initial_val):
        self.dt = 0.1
        self.g = 9.81
        self.x = initial_val[0]
        self.y = initial_val[1]
        self.psi = initial_val[2]
        self.V = initial_val[3]
        self.xdot = 0
        self.ydot = 0
        self.aic = [self.x, self.y, self.psi, self.V]
        self.time = 0
        #States
        self.x_vals = np.array([])
        self.y_vals = np.array([])
        self.v_vals = np.array([])
        self.psi_vals = np.array([])
        self.time_vals = np.array([])
        #Action
        self.u_vals = np.array([])
        self.dv_vals = np.array([])

        self.psi_r = np.array([])
        self.v_r = np.array([])


    def Dynamics(self, action):
        # Aircraft Dynamic
        u = action[0]*np.pi/6
        dv = action[1]
        # Bank Angle
        for i in range(1):
            self.aic[0] +=  self.dt  * self.aic[3] * np.cos(self.aic[2])
            self.aic[1] +=  self.dt  * self.aic[3] * np.sin(self.aic[2])
            self.aic[2] +=  self.dt  * self.g * np.tan(u) / self.aic[3]
            self.aic[3] += self.dt * dv
            self.time += self.dt 
        self.aic[3] = np.clip(self.aic[3],5,12)
        self.aic[2] = self.wrap(self.aic[2])
        self.saveParameters(u,dv)
        
    def saveParameters(self,u,dv):

        self.x_vals = np.append(self.x_vals,self.aic[0])
        self.y_vals = np.append(self.y_vals,self.aic[1])
        self.psi_vals = np.append(self.psi_vals,self.aic[2]) 
        self.v_vals = np.append(self.v_vals, self.aic[3])
        self.u_vals = np.append(self.u_vals,u)
        self.dv_vals = np.append(self.dv_vals,dv)
        self.time_vals = np.append(self.time_vals,self.time)

    def saveParameters_ref(self,psi_r,v_r):
        self.psi_r = np.array(self.psi_r, psi_r)
        self.v_r = np.array(self.v_r, v_r)

    def wrap(self,aic_psi):
        while aic_psi > np.pi:
            aic_psi -= 2*np.pi
        while aic_psi < -np.pi:
            aic_psi += 2*np.pi
        return aic_psi