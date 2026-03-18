import pybullet as p
import time
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np

dt = 1/240 # pybullet simulation step
q0 = 0.5   # starting position (radian)
g = 10
L = 0.8
gui = False
if gui:
    physicsClient = p.connect(p.GUI) # or p.DIRECT for non-graphical version
else:
    physicsClient = p.connect(p.DIRECT)
p.setGravity(0,0,-g)
boxId = p.loadURDF("./simple.urdf.xml", useFixedBase=True)


# get rid of all the default damping forces
p.changeDynamics(boxId, 1, linearDamping=0, angularDamping=0)
p.changeDynamics(boxId, 2, linearDamping=0, angularDamping=0)

# go to the starting position
p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetPosition=q0, controlMode=p.POSITION_CONTROL)
for _ in range(1000):
    p.stepSimulation()

# turn off the motor for the free motion
p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetVelocity=0, controlMode=p.VELOCITY_CONTROL, force=0)

t = 0
logTime = [t]
logTheta = [q0]
maxTime = 5
while t < maxTime:
    p.stepSimulation()
    pos = p.getJointState(boxId, 1)[0]
    logTheta.append(pos)
    t += dt
    logTime.append(t)
    if gui:
        time.sleep(dt)
p.disconnect()

def right_part(x, t):
    return np.array([x[1],
                     -g/L * np.sin(x[0])])

# substitute with pybullet-based integration method
theta = odeint(func=right_part,
               y0=[q0, 0],
               t=logTime)

logThetaInt = theta[:,0]

# L2: ||logTheta - logThetaInt||
# diff = abs(logTheta-logThetaInt)
# L2 = 1/N * sqrt(diff[0]**2 + diff[1]**2 + ...)
# Linf = max(diff)

plt.plot(logTime, logTheta, label='pybullet')
plt.plot(logTime, logThetaInt, label='odeint')
plt.grid(True)
plt.legend()
plt.show()