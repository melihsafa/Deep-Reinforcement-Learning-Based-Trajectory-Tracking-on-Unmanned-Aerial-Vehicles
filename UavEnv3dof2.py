import numpy as np
import gym
from gym import spaces
from stable_baselines3 import DQN, PPO, A2C
import time
import cv2
from funcs import *
from matplotlib import pyplot as plt
from Aircraft3dof2 import Aircraft

def terminalState(self,obs):
  if obs[0,1] > 200 or self.Aircraft.time > 3000:
    return True
  else:
    return False

def terminalState2(self):
  if np.sqrt((self.Aircraft.aic[0] - self.waypoints[-1][0])**2 + (self.Aircraft.aic[1] - self.waypoints[-1][1])**2) < 50:
    return True
  else:
    return False

def checkpoint(self):
  checkpoints = [np.array([self.waypoints[i*10][0],self.waypoints[i*10][1]]) for i in range(int(len(self.waypoints[:,0])/10))]
  if np.sqrt((self.Aircraft.aic[0] - checkpoints[self.a][0])**2 + (self.Aircraft.aic[1] - checkpoints[self.a][1])**2) < 5:
    self.a +=1
    return True

  else: 
    return False
def wrap(aic_psi):
  while aic_psi > np.pi:
    aic_psi -= 2*np.pi
  while aic_psi < -np.pi:
    aic_psi += 2*np.pi
  return aic_psi


class UavEnv(gym.Env):
  metadata = {'render.modes': ['console']}
  # Define constants for clearer code

  def __init__(self):
    super(UavEnv, self).__init__()
    gym.logger.set_level(40)
    # Define action and observation space
    self.success = 0 
    self.aic_vOld = 0
    self.aic_psiOld = 0
    self.v_r = np.array([])
    self.psi_r = np.array([])
    # Constants that used for normalization of observation states

    self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
    self.observation_space = spaces.Box(low=-np.array(4), 
                                        high=np.ones(4),dtype=np.float32)                                      

  def step(self, action, test=False):
    # Environmental parameters

    self.Aircraft.Dynamics(action)
    #Aircraft states
    aic_x = self.Aircraft.aic[0]
    aic_y = self.Aircraft.aic[1]
    aic_psi = wrap(self.Aircraft.aic[2])
    aic_v = self.Aircraft.aic[3]


    #Rewarding[]--------------------------------------------
    b1 = .1
    b2 = .05
    b3 = .05
    b4 = .1
    #observation 0: index, 1: r, 2: theta_t
    obs = observationWaypoint(aic_x, aic_y, self.waypoints[:,0], self.waypoints[:,1])

    V_ec = np.abs(aic_v - self.aic_vOld)
    W_ec = np.abs(aic_psi - self.aic_psiOld)
    d_e = np.abs(obs[0][1]*np.sin(self.waypoints[int(obs[0,0]),2]-obs[0][2]))/50
    phi_e = np.abs(self.waypoints[int(obs[0,0])][2] - aic_psi)/np.pi
    v_e = np.abs(self.waypoints[int(obs[0,0])][3] - aic_v)/5

    self.v_r = np.append(self.v_r, self.waypoints[int(obs[0,0])][3])
    self.psi_r = np.append(self.psi_r, self.waypoints[int(obs[0,0])][2])

    #t-1 values for linear and angular velocity, respectively
    self.aic_vOld = aic_v
    self.aic_psiOld = aic_psi

    #self.reward = -b1*v_e*phi_e*d_e - b2*V_ec - b3*W_ec -b4*d_e
    self.reward = -d_e - phi_e - v_e -.5*W_ec - .5*V_ec
    #Rewarding[]----------------------------------------------
    #Observation[]--------------------------------------------

    r_t = obs[:,1]
    theta_t = obs[:,2]
    alpha_t = [self.waypoints[int(obs[i,0])][2] for i in range(len(obs[:,0]))] 
    phi_t = aic_psi
    v = aic_v
    v_t = [self.waypoints[int(obs[i,0])][3] for i in range(len(obs[:,0]))] 

    obs_0 = r_t/np.max(r_t)
    obs_1 = (theta_t - phi_t)
    obs_2 = (alpha_t - phi_t)
    obs_3 = (v_t - v)/10

    observation = np.array([])
    observation = np.append(observation, obs_0[0])
    observation = np.append(observation, wrap(obs_1[0])/np.pi)
    observation = np.append(observation, wrap(obs_2[0])/np.pi)
    observation = np.append(observation, obs_3[0])



    #Observation[]--------------------------------------------
    if terminalState(self,obs):
      self.reward = -10
      self.done = True

    if terminalState2(self):
      self.reward = 100
      self.done = True

    if checkpoint(self):
      self.reward = 5


    info = {}
   
    """if test:
      self.visualize()"""

    if test and self.done:
      self.showPlots()
      #print(self.reward)

      
    return observation, self.reward, self.done, info

  def reset(self,init=0, final=0, train=True):
    self.a = 0
    self.waypoints = waypointGenerator(np.random.randint(4))
    self.waypoints = waypointGenerator(4)
    #self.waypoints = helicalWaypointGenerator()
    self.initial_conditions = self.waypoints[np.random.randint(10)+10] # Initial Conditions [x,y,psi,v]

    self.done = False
    self.turn = 0
    self.Aircraft = Aircraft(self.initial_conditions)

    #observation 0: index, 1: r, 2: theta_t
    aic_x = self.Aircraft.aic[0]
    aic_y = self.Aircraft.aic[1]
    aic_psi = self.Aircraft.aic[2]
    aic_v = self.Aircraft.aic[3]

    obs = observationWaypoint(aic_x, aic_y, self.waypoints[:,0], self.waypoints[:,1])

    #Observation[]--------------------------------------------

    r_t = obs[:,1]
    theta_t = obs[:,2]
    alpha_t = [self.waypoints[int(obs[i,0])][2] for i in range(len(obs[:,0]))] 
    phi_t = aic_psi
    v = aic_v
    v_t = [self.waypoints[int(obs[i,0])][3] for i in range(len(obs[:,0]))] 

    self.v_r = np.array([])
    self.psi_r = np.array([])

    obs_0 = r_t/np.max(r_t)
    obs_1 = (theta_t - phi_t)
    obs_2 = (alpha_t - phi_t)
    obs_3 = (v_t - v)/10

    observation = np.array([])
    observation = np.append(observation, obs_0[0])
    observation = np.append(observation, wrap(obs_1[0])/np.pi)
    observation = np.append(observation, wrap(obs_2[0])/np.pi)
    observation = np.append(observation, obs_3[0])

    return observation

  def showPlots(self):

    # Plot 1
    plt.figure(figsize=[8, 6])
    plt.plot(self.Aircraft.x_vals, self.Aircraft.y_vals,'r')
    plt.plot(self.Aircraft.x_vals[0],self.Aircraft.y_vals[0],'og', markersize=10)
    plt.plot(self.Aircraft.x_vals[-1],self.Aircraft.y_vals[-1],'ob', markersize=10)
    plt.scatter(self.waypoints[:,0], self.waypoints[:,1], s = 8)
    #plt.plot(self.waypoints[:,0], self.waypoints[:,1])
    plt.xlabel('X (m)', fontsize=10)
    plt.ylabel('Y (m)', fontsize=10)
    plt.legend(["Aircraft Trajectory", "Initial Point", "Final Point", "Waypoints"], loc='lower right',prop={'size': 8})
    plt.grid()
    #plt.savefig('xy.png') 
    plt.show()

    #plot 2
    plt.figure(figsize=[9,6])

    ax1 = plt.subplot(4, 1, 1)
    ax1.plot( self.Aircraft.time_vals, self.Aircraft.v_vals,'r', linewidth=.6)
    ax1.plot( self.Aircraft.time_vals, self.v_r,'b', linewidth=.6)
    ax1.legend(['V_aircraft', 'V_ref'],loc='upper right',prop={'size': 6})
    ax1.set_ylabel('v(m/s)', fontsize=8)
    ax1.set_xlabel('time(s)', fontsize=8)
    ax1.tick_params(axis='x', labelsize=0)
    ax1.tick_params(axis='y', labelsize=8)
    ax1.grid()

    ax2 = plt.subplot(4, 1, 2)
    ax2.plot(self.Aircraft.time_vals,self.Aircraft.dv_vals,'r', linewidth=.6)
    ax2.set_ylim([-1,1])
    ax2.grid()
    ax2.tick_params(axis='x', labelsize=0)
    ax2.tick_params(axis='y', labelsize=8)
    ax2.set_ylabel('a(m/s^2)', fontsize=8)
    ax2.set_xlabel('time(s)', fontsize=8)

    ax3 = plt.subplot(4, 1, 3)
    ax3.plot( self.Aircraft.time_vals, self.Aircraft.psi_vals,'r', linewidth=.6)
    ax3.plot( self.Aircraft.time_vals, self.psi_r,'b', linewidth=.6)
    ax3.legend(['psi_aircraft', 'psi_ref'],loc='upper right',prop={'size': 6})
    ax3.set_ylabel('psi', fontsize=8)
    ax3.set_xlabel('time(s)', fontsize=8)
    ax3.tick_params(axis='x', labelsize=0)
    ax3.tick_params(axis='y', labelsize=8)
    ax3.set_ylim([-np.pi,np.pi])
    ax3.grid()

    ax4 = plt.subplot(4, 1, 4)
    ax4.plot(self.Aircraft.time_vals,self.Aircraft.u_vals,'r', linewidth=.6)
    ax4.grid()
    ax4.set_ylabel('bank angle', fontsize=8)
    ax4.set_xlabel('time(s)', fontsize=8)
    ax4.set_ylim([-np.pi/6,np.pi/6])
    ax4.tick_params(axis='x', labelsize=8)
    ax4.tick_params(axis='y', labelsize=8)
    plt.show()