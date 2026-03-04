import pybullet as p
import time
import matplotlib.pyplot as plt

dt = 1/240 # pybullet simulation step
q0 = 0.5   # starting position (radian)
physicsClient = p.connect(p.GUI) # or p.DIRECT for non-graphical version
p.setGravity(0,0,-10)
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
    time.sleep(dt)
p.disconnect()

plt.plot(logTime, logTheta)
plt.grid(True)
plt.show()