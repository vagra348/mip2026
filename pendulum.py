import pybullet as p
import time
import matplotlib.pyplot as plt
import numpy as np


dt = 1 / 240
g = 10
L1 = 0.8
L2 = 0.8
baseXZ = np.array([0.0, 2.0])
jointIndices = [1, 3]
eefLinkIndex = 4
gui = True

kp = 8.0
maxJointSpeed = 8.0
segmentTime = 2.0

points = np.array([
    [-0.25, 0.75],
    [0.35, 0.95],
    [0.55, 1.35],
    [0.05, 1.55],
    [-0.45, 1.25],
    [-1.25, 1.05],
])


# Modern Robotics, 9.3: beta(t) = a0 + a1*t + a2*t^2 + a3*t^3, формула (9.25)
def cubic_coefficients(p0, p1, v0, v1, T):
    a0 = p0
    a1 = v0
    a2 = (3 * p1 - 3 * p0 - 2 * v0 * T - v1 * T) / T**2   # MR, формула (9.28)
    a3 = (2 * p0 + (v0 + v1) * T - 2 * p1) / T**3   # MR, формула (9.29)
    return a0, a1, a2, a3


# проверка выхода точек за границы достижимости
maxRadius = L1 + L2
pointsOk = True
for i in range(len(points)):
    dist = np.linalg.norm(points[i] - baseXZ)
    if dist > maxRadius:
        pointsOk = False
        print(f"Точка {i} недостижима: {points[i]}, расстояние = {dist:.3f}")

if not pointsOk:
    print(f"Интервалы достижимости: x: [{baseXZ[0] - maxRadius}, {baseXZ[0] + maxRadius}], "
          f"z: [{baseXZ[1] - maxRadius}, {baseXZ[1] + maxRadius}]")
    raise SystemExit("Недостижимая точка")


# скорости в проходимых точках: 0 в начале и конце, НЕ ноль в промеж-х, т.к. в задании - без остановок
waypointVelocities = np.zeros_like(points)
for i in range(1, len(points) - 1):
    waypointVelocities[i] = (points[i + 1] - points[i - 1]) / (2 * segmentTime)

segmentCount = len(points) - 1
maxTime = segmentCount * segmentTime
segments = []
for i in range(segmentCount):
    segments.append(cubic_coefficients(
        points[i],
        points[i + 1],
        waypointVelocities[i],
        waypointVelocities[i + 1],
        segmentTime
    ))


physicsClient = p.connect(p.GUI if gui else p.DIRECT)
p.setGravity(0, 0, -g)
p.setTimeStep(dt)
boxId = p.loadURDF("./two-link.urdf.xml", useFixedBase=True)

for joint in jointIndices:
    p.setJointMotorControl2(
        bodyIndex=boxId,
        jointIndex=joint,
        targetVelocity=0,
        controlMode=p.VELOCITY_CONTROL,
        force=0
    )

startAngles = p.calculateInverseKinematics(
    bodyUniqueId=boxId,
    endEffectorLinkIndex=eefLinkIndex,
    targetPosition=[points[0, 0], 0, points[0, 1]]
)
for joint, angle in zip(jointIndices, startAngles[:2]):
    p.resetJointState(boxId, joint, angle)

t = 0
logTime = [t]
logX = []
logZ = []
logXd = []
logZd = []
logErr = []

while t <= maxTime + 1.0:
    if t >= maxTime:
        segment = segmentCount - 1
        localTime = segmentTime
    else:
        segment = int(t // segmentTime)
        localTime = t - segment * segmentTime

    a0, a1, a2, a3 = segments[segment]
    desiredPos = a0 + a1 * localTime + a2 * localTime**2 + a3 * localTime**3
    desiredVel = a1 + 2 * a2 * localTime + 3 * a3 * localTime**2

    pos = p.getLinkState(boxId, eefLinkIndex)[0]
    actualPos = np.array([pos[0], pos[2]])
    error = desiredPos - actualPos

    q = [p.getJointState(boxId, joint)[0] for joint in jointIndices]
    qd = [p.getJointState(boxId, joint)[1] for joint in jointIndices]
    linearJacobian, _ = p.calculateJacobian(
        boxId,
        eefLinkIndex,
        localPosition=[0, 0, 0],
        objPositions=q,
        objVelocities=qd,
        objAccelerations=[0, 0]
    )
    jacobian = np.array(linearJacobian)[[0, 2], :]

    cartesianVelocity = desiredVel + kp * error
    jointVelocity = np.linalg.inv(jacobian) @ cartesianVelocity
    jointVelocity = np.clip(jointVelocity, -maxJointSpeed, maxJointSpeed)

    p.setJointMotorControlArray(
        bodyUniqueId=boxId,
        jointIndices=jointIndices,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocities=jointVelocity,
        forces=[80, 80]
    )

    p.stepSimulation()

    logX.append(actualPos[0])
    logZ.append(actualPos[1])
    logXd.append(desiredPos[0])
    logZd.append(desiredPos[1])
    logErr.append(np.linalg.norm(error))

    t += dt
    logTime.append(t)
    if gui:
        time.sleep(dt)

p.disconnect()

plt.subplot(2, 1, 1)
plt.plot(logXd, logZd, 'r--', label='desired')
plt.plot(logX, logZ, label='actual')
plt.scatter(points[:, 0], points[:, 1], color='black', label='points')
plt.axis('equal')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(logTime[:-1], logErr, label='error')
plt.grid(True)
plt.legend()

plt.show()
