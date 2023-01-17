import math
from re import T
import numpy as np
from scipy.interpolate import CubicSpline

def sineWaypointGenerator():
    x = np.arange(0, 7*np.pi, np.pi/24)
    y = 150*np.sin(x)   
    alpha = [np.arctan2(y[i+1] - y[i], 150*x[i+1] - 150*x[i]) for i in range(len(x)-1)]
    alpha.append(0)
    v = np.random.uniform(low=5, high=7, size=(len(x),))
    waypoints = [np.array([150*x[i],y[i],alpha[i],v[i]]) for i in range(len(x))]
    return np.array(waypoints).reshape(len(x),4)

def helicalWaypointGenerator():
    h = np.linspace(0, 300, 120)
    x = h * np.sin(h/25)
    y = h * np.cos(h/25)
    alpha_1 = [np.arctan2(y[i+1] - y[i], x[i+1] - x[i]) for i in range(len(x)-1)]
    alpha_1.append(0)
    v = np.random.uniform(low=5, high=7, size=(len(x),))
    waypoints = [np.array([x[i],y[i],alpha_1[i],v[i]]) for i in range(len(x))]
    return np.array(waypoints).reshape(len(x),4)

def straightWaypointGenerator():
    x = np.linspace(0, 1500, 120)
    y = np.linspace(0, 1500, 120)
    alpha_1 = [np.arctan2(y[i+1] - y[i], x[i+1] - x[i]) for i in range(len(x)-1)]
    alpha_1.append(0)
    v = np.random.uniform(low=5, high=7, size=(len(x),))
    waypoints = [np.array([x[i],y[i],alpha_1[i],v[i]]) for i in range(len(x))]
    return np.array(waypoints).reshape(len(x),4)

def circularWaypointGenerator():
    h = np.linspace(0, 2*np.pi, 200)
    x = 1000 * np.sin(h)
    y = 1000 * np.cos(h)
    alpha_1 = [np.arctan2(y[i+1] - y[i], x[i+1] - x[i]) for i in range(len(x)-1)]
    alpha_1.append(0)
    v = np.random.uniform(low=5, high=7, size=(len(x),))
    waypoints = [np.array([x[i],y[i],alpha_1[i],v[i]]) for i in range(len(x))]
    return np.array(waypoints).reshape(len(x),4)    

def mixedWaypointGenerator():
    x = np.linspace(-400,400,200)
    y = x*np.sin(x/20) + ((x-100)/40)**3
    f = CubicSpline(x, y, bc_type='natural')
    x_new = np.linspace(-400,400,200)
    y_new = f(x_new)
    alpha = [np.arctan2(y_new[i+1] - y_new[i], x_new[i+1] - x_new[i]) for i in range(len(x_new)-1)]
    alpha.append(0)
    v = np.random.uniform(low=5, high=7, size=(len(x),))
    waypoints = [np.array([x_new[i],y_new[i],alpha[i],v[i]]) for i in range(len(x))]
    return np.array(waypoints).reshape(len(x),4)    

def waypointGenerator(decider):
    if decider == 0:
        return sineWaypointGenerator()
    elif decider == 1:
        return helicalWaypointGenerator()
    elif decider == 2:
        return straightWaypointGenerator()
    elif decider == 3:
        return circularWaypointGenerator()
    elif decider == 4:
        return mixedWaypointGenerator()



def observationWaypoint(x,y,waypoint_x, waypoint_y):
    distance = [np.sqrt((waypoint_y[i] - y)**2 + (waypoint_x[i] - x)**2) for i in range(len(waypoint_x))]
    theta = [np.arctan2(waypoint_y[i] - y, waypoint_x[i] - x) for i in range(len(waypoint_x))]
    observation = [np.array([int(i),distance[i],theta[i]]) for i in range(len(distance))]
    observation.sort(key = lambda x: x[1])
    return np.array(observation[:10]).reshape(10, 3)

