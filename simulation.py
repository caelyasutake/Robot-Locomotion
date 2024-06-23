import pybullet as p
import time
import pybullet_data
import math

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0, 0, 0.2]
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
p.setAdditionalSearchPath("/cad")
robotId = p.loadURDF("myrobot.urdf", cubeStartPos, cubeStartOrientation)

mode = p.POSITION_CONTROL
'''
robotId -> joint
2 -> back right hip
4 -> front left hip

0 -> front right hip
6 -> back left hip

3 -> back right knee
5 -> front left knee

1 -> front right knee
7 -> back left knee
'''


for i in range(10000):
    p.stepSimulation(physicsClient)
    p.setJointMotorControl2(robotId, 2, controlMode=mode, targetPosition=math.sin(i * 0.1))
    p.setJointMotorControl2(robotId, 4, controlMode=mode, targetPosition=math.sin(i * 0.1))

    p.setJointMotorControl2(robotId, 0, controlMode=mode, targetPosition=math.sin(-i * 0.1))
    p.setJointMotorControl2(robotId, 6, controlMode=mode, targetPosition=math.sin(-i * 0.1))

    p.setJointMotorControl2(robotId, 3, controlMode=mode, targetPosition=math.sin(i * 0.1))
    p.setJointMotorControl2(robotId, 5, controlMode=mode, targetPosition=math.sin(i * 0.1))

    p.setJointMotorControl2(robotId, 1, controlMode=mode, targetPosition=math.sin(-i * 0.1))
    p.setJointMotorControl2(robotId, 7, controlMode=mode, targetPosition=math.sin(-i * 0.1))
    time.sleep(1./240.)

cubePos, cubeOrn = p.getBasePositionandOrientation(robotId)
print(cubePos, cubeOrn)

p.disconnect()