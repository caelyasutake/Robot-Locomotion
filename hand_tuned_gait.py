import pybullet as p
import pybullet_data
import time
import numpy as np
import math

physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -98.1)
plane_id = p.loadURDF("plane.urdf")

cubeStartPos = [0, 0, 0.1]
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
p.setAdditionalSearchPath("/cad2")
robot_id = p.loadURDF("shibot_inu.urdf", cubeStartPos, cubeStartOrientation)
mode = p.POSITION_CONTROL

p.changeDynamics(robot_id, 4, lateralFriction=2)
p.changeDynamics(robot_id, 5, lateralFriction=2)
p.changeDynamics(robot_id, 6, lateralFriction=2)
p.changeDynamics(robot_id, 7, lateralFriction=2)

z = 0.1
# EA GAIT (FITNESS: DISTANCE)
a, b, c, d, e, f, g, h = 0.06331716760078639, -0.1325761177097645, 0.5434685394143213, 3.1457615737348466, 2.546245999122622, -7.098439990793135, 1.802103217453233, 2.919263383708246

#a, b, c, d, e, f, g, h = -0.1473143376844095, 2.3959797333665076, 2.039703476071842, -3.6748192865857856, 3.5936885807997916, 1.5340999870069534, 0.15204666425504182, -0.43141839140676563

#ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, ex, ey, ez, fx, fy, fz, gx, gy, gz, hx, hy, hz = 4.769265015330216, 2.2050890565987467, 3.038964731967647, 7.295026543337501, -1.7650853902565264, 1.135905478524371, -2.8279360123669504, -4.921099445286428, -7.26662387378655, 3.754823390935754, -0.6212431461106838, 1.6066713232249783, 2.15776655446479, 1.8448036953509455, -0.18042956375755642, 4.092849528105379, -4.77276334139455, -3.8260946523165824, 2.967055498989229, 1.704722743114591, 1.1739098261193186, 10.788238128539815, -3.9422277448160705, -1.4385585966642704
#ax, ay, az, ad, ae, bx, by, bz, bd, be, cx, cy, cz, cd, ce, dx, dy, dz, dd, de, ex, ey, ez, ed, ee, fx, fy, fz, fd, fe, gx, gy, gz, gd, ge, hx, hy, hz, hd, he = -2.447966148680953, -0.4959297540498664, -1.1161577116628854, -0.032971495455589396, -1.2617456577574675, -2.2316731539349477, 1.2715878719058749, 0.22171679829553784, 0.5682369795728082, -0.2915559636300573, -1.5267627188027868, 1.0830981219599356, -0.6383964043024577, 0.25110349544169375, -1.1609725317670598, -1.0465319536358515, -0.43993591004005494, -0.569466325537685, -0.2812266935351042, 1.3397418727767771, -1.7721678309866262, 1.4393298266179348, 1.9499454571732837, -0.6384464430846164, -3.1606691741732518, -1.8736035119117367, 0.6626914762264802, 1.8377896026627218, -1.6724557149990158, 0.7764784559076072, -1.6436239376726505, -2.8903629982263634, 0.714470433079821, 1.4439639983780814, -0.6032628127798467, -1.4118333682394595, 1.113155913500036, -3.686142603174262, -0.17416418989431298, 1.6220786060924832

ax, ay, az, ad, ae, bx, by, bz, bd, be, cx, cy, cz, cd, ce, dx, dy, dz, dd, de, ex, ey, ez, ed, ee, fx, fy, fz, fd, fe, gx, gy, gz, gd, ge, hx, hy, hz, hd, he = 1.6310760420601125, 1.8125943462568777, 0.4538281958487003, -1.2510017778639244, 0.042453921739187966, -0.7508381434099609, -0.2918654542861938, 1.004003339349092, -0.5882878508231426, -1.521768274988172, 2.1869350566866768, 1.2439235518979592, -0.08518204013315334, 1.3426063057765236, -0.9310438152371494, -0.06020132533241616, -0.58186356606269, -1.3722620093510316, -3.8487359122830824, -0.16128392175477912, -1.3946928779560128, 1.3831625783976742, -1.6446861939426105, -1.5806007679804983, -1.5811639335607777, 1.0679862355990881, -0.1238045928121062, -0.774811029278355, -1.639919090377579, -0.7670627085634949, 0.9875313844471919, 1.9164717705523784, 2.031556030644345, -1.9996100206629897, 0.7487061299467741, 2.7540850912837045, -0.2094848068570398, -0.15347302968403775, 1.6512636867797923, -1.428889047170439


for i in range(10000):
    p.stepSimulation(physics_client)

    roll, pitch, yaw = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(robot_id)[1])
    print("roll, pitch, yaw", roll, pitch, yaw)

    linear_velocity, angular_velocity = p.getBaseVelocity(robot_id)
    print(f"Linear Velocity: {linear_velocity}")
    print(f"Angular Velocity: {angular_velocity}")

    '''
    p.setJointMotorControl2(robot_id, 0,
                            controlMode=mode,
                            targetPosition=a * math.sin(i * z))
    p.setJointMotorControl2(robot_id, 2,
                            controlMode=mode,
                            targetPosition=b * math.cos(i * z))
    p.setJointMotorControl2(robot_id, 4,
                            controlMode=mode,
                            targetPosition=c * math.cos(i * z))
    p.setJointMotorControl2(robot_id, 6,
                            controlMode=mode,
                            targetPosition=d * math.sin(i * z))

    # Feet
    p.setJointMotorControl2(robot_id, 1,
                            controlMode=mode,
                            targetPosition=e * math.sin(i * z))
    p.setJointMotorControl2(robot_id, 3,
                            controlMode=mode,
                            targetPosition=f * math.cos(i * z))
    p.setJointMotorControl2(robot_id, 5,
                            controlMode=mode,
                            targetPosition=g * math.cos(i * z))
    p.setJointMotorControl2(robot_id, 7,
                            controlMode=mode,
                            targetPosition=h * math.sin(i * z))

    '''
    '''
    p.setJointMotorControl2(robot_id, 0, controlMode=p.POSITION_CONTROL, targetPosition=ax + ay * math.sin(i + az))
    p.setJointMotorControl2(robot_id, 2, controlMode=p.POSITION_CONTROL, targetPosition=bx + by * math.cos(i + bz))
    p.setJointMotorControl2(robot_id, 4, controlMode=p.POSITION_CONTROL, targetPosition=cx + cy * math.cos(i + cz))
    p.setJointMotorControl2(robot_id, 6, controlMode=p.POSITION_CONTROL, targetPosition=dx + dy * math.sin(i + dz))
    p.setJointMotorControl2(robot_id, 1, controlMode=p.POSITION_CONTROL, targetPosition=ex + ey * math.sin(i + ez))
    p.setJointMotorControl2(robot_id, 3, controlMode=p.POSITION_CONTROL, targetPosition=fx + fy * math.cos(i + fz))
    p.setJointMotorControl2(robot_id, 5, controlMode=p.POSITION_CONTROL, targetPosition=gx + gy * math.cos(i + gz))
    p.setJointMotorControl2(robot_id, 7, controlMode=p.POSITION_CONTROL, targetPosition=hx + hy * math.sin(i + hz))
    '''

    # '''
    p.setJointMotorControl2(robot_id, 0,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=ax + ay * math.sin(i * az) + ad * math.sin(i * ae))
    p.setJointMotorControl2(robot_id, 2,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=bx + by * math.sin(i * bz) + bd * math.sin(i * be))
    p.setJointMotorControl2(robot_id, 4,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=cx + cy * math.sin(i * cz) + cd * math.sin(i * ce))
    p.setJointMotorControl2(robot_id, 6,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=dx + dy * math.sin(i * dz) + dd * math.sin(i * de))
    p.setJointMotorControl2(robot_id, 1,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=ex + ey * math.sin(i * ez) + ed * math.sin(i * ee))
    p.setJointMotorControl2(robot_id, 3,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=fx + fy * math.sin(i * fz) + fd * math.sin(i * fe))
    p.setJointMotorControl2(robot_id, 5,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=gx + gy * math.sin(i * gz) + gd * math.sin(i * ge))
    p.setJointMotorControl2(robot_id, 7,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=hx + hy * math.sin(i * hz) + hd * math.sin(i * he))
    # '''

    time.sleep(1./240.)

p.disconnect()