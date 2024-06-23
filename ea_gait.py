import random
import numpy as np
import math
import time
import pybullet as p
import pybullet_data
from deap import base, creator, tools, algorithms


# Define the simulation function
def run_simulation(params, gui=False):
    if gui:
        physics_client = p.connect(p.GUI)  # Use GUI to visualize the simulation
    else:
        physics_client = p.connect(p.DIRECT)  # Use DIRECT for faster headless simulation

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -98.1)
    plane_id = p.loadURDF("plane.urdf")

    cubeStartPos = [0, 0, 0.1]
    cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
    p.setAdditionalSearchPath("/cad2")  # Ensure this path points to your URDF file
    robot_id = p.loadURDF("shibot_inu.urdf", cubeStartPos, cubeStartOrientation)

    #a, b, c, d, e, f, g, h = params
    ax, ay, az, ad, ae, bx, by, bz, bd, be, cx, cy, cz, cd, ce, dx, dy, dz, dd, de, ex, ey, ez, ed, ee, fx, fy, fz, fd, fe, gx, gy, gz, gd, ge, hx, hy, hz, hd, he = params

    max_angular_velocity = 0
    for i in range(1000):  # Run the simulation for a fixed number of steps
        p.stepSimulation()
        '''
        p.setJointMotorControl2(robot_id, 0,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=a * math.sin(i * 0.1))
        p.setJointMotorControl2(robot_id, 2,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=b * math.cos(i * 0.1))
        p.setJointMotorControl2(robot_id, 4,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=c * math.cos(i * 0.1))
        p.setJointMotorControl2(robot_id, 6,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=d * math.sin(i * 0.1))

        # Feet
        p.setJointMotorControl2(robot_id, 1,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=e * math.sin(i * 0.1))
        p.setJointMotorControl2(robot_id, 3,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=f * math.cos(i * 0.1))
        p.setJointMotorControl2(robot_id, 5,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=g * math.cos(i * 0.1))
        p.setJointMotorControl2(robot_id, 7,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=h * math.sin(i * 0.1))
        '''
        #'''
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
        #'''

        if gui:
            time.sleep(1. / 240.)  # Slow down the simulation for visualization

        # Retrieve angular velocity for stability evaluation
        _, angular_velocity = p.getBaseVelocity(robot_id)
        max_angular_velocity = max(max_angular_velocity, np.linalg.norm(angular_velocity))

        # Check for collision between robot base and floor
        contacts = p.getContactPoints(bodyA=robot_id, bodyB=plane_id, linkIndexA=0)
        if contacts:
            print("Robot has fallen! Simulation stopped.")
            p.disconnect()
            return -100,  # Return a large negative value if the robot falls

    # Calculate the distance traveled
    pos, rot = p.getBasePositionAndOrientation(robot_id)
    distance = -pos[0]  # Invert the x-coordinate to reward travel in the -x direction
    # -pos[0] moves forward
    # pos[0] moves backward
    # -rot[1] turns right
    # rot[1] turns left

    # pos[1] jumps?

    # Calculate fitness with a penalty for instability
    fitness = distance #- 0.5 * max_angular_velocity # The coefficient can be adjusted based on the desired stability

    p.disconnect()
    return fitness,


# DEAP setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -4, 4)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=40)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", lambda ind: run_simulation(ind))
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=4)

# Genetic algorithm parameters
population = toolbox.population(n=100)
ngen = 100
cxpb = 0.5
mutpb = 0.2

# Run the genetic algorithm
final_pop = algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, verbose=True)

# Extracting and running the best individual in GUI mode
best_ind = tools.selBest(population, 1)[0]
print("Best individual is: ", best_ind)
print("Best fitness is: ", best_ind.fitness.values)
print("Best parameters are: ", best_ind)
run_simulation(best_ind, gui=True)  # Re-run simulation with GUI




