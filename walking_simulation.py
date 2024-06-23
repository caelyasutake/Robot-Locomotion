import pybullet as p
import time
import pybullet_data
import math

physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -98.1)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0, 0, 0.1]
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
p.setAdditionalSearchPath("/cad2")
robotId = p.loadURDF("shibot_inu.urdf", cubeStartPos, cubeStartOrientation)
mode = p.POSITION_CONTROL

'''
sphere_radius = 0.1
visualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                    radius=sphere_radius,
                                    rgbaColor=[1, 0, 0, 1])  # Red color with full opacity

# Positioning the sphere in front of the robot
# Assume it's 1 meter in front of the robot's initial position
sphere_position = [-1, 0, 0.1]

# Create the sphere as a multibody (since it doesn't need dynamics, mass=0)
sphereId = p.createMultiBody(baseMass=0,
                             baseVisualShapeIndex=visualShapeId,
                             basePosition=sphere_position)

'''
'''
robotId -> joint (ID)
0 -> l_hip_leg1 (1)
1 -> l_foot1 (10)
2 -> r_hip_leg1 (2)
3 -> r_foot1 (20)

4 -> l_hip_leg2 (3)
5 -> l_foot2 (30)
6 -> r_hip_leg2 (4)
7 -> r_foot2 (40)

ID: Motor Ranges
1: 120 -> 80
2: 120 -> 150
3: 120 -> 80
4: 120 -> 150

10: 120 -> 60
20: 120 -> 180
30: 120 -> 60
40: 120 -> 180
'''
'''
width, height = 640, 480
fov, aspect, nearplane, farplane = 60, width/height, 0.02, 5
cameraOffset = [-0.2, 0, 0.06]
'''
for i in range(10000):
    p.stepSimulation(physics_client)

    '''
    robotPos, robotOrient = p.getBasePositionAndOrientation(robotId)
    cameraPos = p.multiplyTransforms(robotPos, robotOrient, cameraOffset, p.getQuaternionFromEuler([0, 0, 0]))

    rot_matrix = p.getMatrixFromQuaternion(robotOrient)
    forward_vec = [rot_matrix[0], rot_matrix[3], rot_matrix[6]]  # Extracting the first column (x-axis)
    camera_target = [cameraPos[0][0] - forward_vec[0],
                     cameraPos[0][1] - forward_vec[1],
                     cameraPos[0][2] - forward_vec[2]]

    viewMatrix = p.computeViewMatrix(
        cameraEyePosition=cameraPos[0],
        cameraTargetPosition=camera_target,
        cameraUpVector=[0, 0, 1]  # Up vector is along the z-axis
    )

    projectionMatrix = p.computeProjectionMatrixFOV(
        fov=fov,
        aspect=aspect,
        nearVal=nearplane,
        farVal=farplane
    )

    img_arr = p.getCameraImage(width, height, viewMatrix, projectionMatrix)
    
    '''

    #_, angular_velocity = p.getBaseVelocity(robotId)
    #print("Angular Velocity: ", angular_velocity)

    # Hips
    p.setJointMotorControl2(robotId, 0, controlMode=mode, targetPosition=math.sin(i * 0.1))
    p.setJointMotorControl2(robotId, 2, controlMode=mode, targetPosition=math.cos(i * 0.1))
    p.setJointMotorControl2(robotId, 4, controlMode=mode, targetPosition=math.cos(i * 0.1))
    p.setJointMotorControl2(robotId, 6, controlMode=mode, targetPosition=math.sin(i * 0.1))

    # Feet
    p.setJointMotorControl2(robotId, 1, controlMode=mode, targetPosition=math.sin(i * 0.1))
    p.setJointMotorControl2(robotId, 3, controlMode=mode, targetPosition=math.cos(i * 0.1))
    p.setJointMotorControl2(robotId, 5, controlMode=mode, targetPosition=math.cos(i * 0.1))
    p.setJointMotorControl2(robotId, 7, controlMode=mode, targetPosition=math.sin(i * 0.1))

    time.sleep(1./240.)

cubePos, cubeOrn = p.getBasePositionandOrientation(robotId)
print(cubePos, cubeOrn)

p.disconnect()