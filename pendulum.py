import pybullet as p
import time
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np

dt = 1/240 # pybullet simulation step
q0 = 0.5   # starting position (radian)
qd = 1.0   # desired position (radian)
kp = 40.0    # proportional coefficient
ki = 40.0    # integral coeffiecient
kd = 10.0   # differential coefficient
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
logCtrl = [0]
maxTime = 20
e_int = 0 # error integral
e_prev = 0
while t < maxTime:
    p.stepSimulation()
    pos = p.getJointState(boxId, 1)[0]
    e = pos - qd
    de = (e - e_prev)/dt
    e_prev = e
    e_int += e * dt
    # ПИД-регулятор
    # Пропорционально-интегрально-дифференциальный регулятор
    u = -kp*e -ki * e_int -kd * de
    # u=0

    # p.setJointMotorControl2(
    #     bodyIndex=boxId,
    #     jointIndex=1,
    #     targetVelocity=u,
    #     controlMode=p.VELOCITY_CONTROL
    # )

    p.setJointMotorControl2(
        bodyIndex=boxId,
        jointIndex=1,
        force=u,
        controlMode=p.TORQUE_CONTROL
    )

    logTheta.append(pos)
    logCtrl.append(u)
    t += dt
    logTime.append(t)
    if gui:
        time.sleep(dt)
p.disconnect()

def symp_euler(fun, x0, TT):
    x1 = np.array(x0)
    xx = np.array(x1)
    for i in range(len(TT)-1):
        dt = (TT[i+1] - TT[i])
        x1[1] += fun(x1, 0)[1]*dt
        x1[0] += x1[1]*dt
        xx = np.vstack((xx,x1))
    return xx

def right_part(x, t):
    return np.array([x[1],
                     -g/L * np.sin(x[0])])

# # substitute with pybullet-based integration method
# theta = odeint(func=right_part,
#                y0=[q0, 0],
#                t=logTime)

# logThetaInt = theta[:,0]

# theta = symp_euler(right_part, [q0, 0], logTime)
# logThetaEuler = theta[:,0]

# # L2: ||logTheta - logThetaInt||
# # diff = abs(logTheta-logThetaInt)
# # L2 = 1/N * sqrt(diff[0]**2 + diff[1]**2 + ...)
# # Linf = max(diff)

# traj_absdiff = np.abs(logTheta - logThetaInt)
# l2_norm = np.sqrt((traj_absdiff**2).sum())
# print("ODEINT")
# print(f'L_2 norm = {l2_norm}')
# l_inf = traj_absdiff.max()
# print(f'L_inf norm = {l_inf}')

# traj_absdiff = np.abs(logTheta - logThetaEuler)
# l2_norm = np.sqrt((traj_absdiff**2).sum())
# print("EULER")
# print(f'L_2 norm = {l2_norm}')
# l_inf = traj_absdiff.max()
# print(f'L_inf norm = {l_inf}')

# ddq = -g/L*sin(q) - k*dq + tau
# ddq = -g/L*sin(q) + tau



plt.subplot(2,1,1)
plt.plot(logTime, logTheta, label='theta')
plt.plot([logTime[0], logTime[-1]], [qd, qd], 'r--', 'desired')
plt.grid(True)
plt.legend()

plt.subplot(2,1,2)
plt.plot(logTime, logCtrl, label='ctrl')
plt.grid(True)
plt.legend()

plt.show()