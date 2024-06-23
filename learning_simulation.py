import pybullet as p
import pybullet_data
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        reward = self.locals['rewards'] if 'rewards' in self.locals else None
        if reward is not None:
            print("Reward: ", reward[0])
        return True

class ShibotEnv(gym.Env):
    def __init__(self):
        super(ShibotEnv, self).__init__()
        self.client = p.connect(p.GUI)
        p.setGravity(0, 0, -98.1, physicsClientId=self.client)

        # Plane Initialization
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf")

        # Robot Initialization
        self.robot_start_pos = [0, 0, 0.1]
        self.robot_start_ori = p.getQuaternionFromEuler([0, 0, 0])
        p.setAdditionalSearchPath("/cad2")
        self.robot_id = p.loadURDF("shibot_inu.urdf", self.robot_start_pos, self.robot_start_ori)
        self.mode = p.POSITION_CONTROL

        num_joints = 8
        obs_limits = np.inf
        self.observation_space = gym.spaces.Box(low=-np.full((num_joints * 2,), obs_limits),
                                                high=np.full((num_joints * 2,), obs_limits),
                                                dtype=np.float32)
        # Define your action space based on the number of joints and their possible range
        self.action_space = gym.spaces.Box(low=-np.pi, high=np.pi, shape=(num_joints,), dtype=np.float32)

    def step(self, action):
        self.apply_action(action)
        for _ in range(10):
            p.stepSimulation(physicsClientId=self.client)

        new_obs = self.get_observation()
        reward = self.calculate_reward(new_obs)

        done = self.check_done(new_obs)

        info = {}

        return new_obs, reward, done, info

    def apply_action(self, action):
        for joint_index in range(len(action)):
            p.setJointMotorControl2(bodyUniqueId=self.robot_id,
                                    jointIndex=joint_index,
                                    controlMode=self.mode,
                                    targetPosition=action[joint_index],
                                    force=500,
                                    physicsClientId=self.client)

    def calculate_reward(self, observation):
        position, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        reward = position[0]
        return reward

    def check_done(self, observation):
        position, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        if position[2] < 0.01:
            return True
        return False

    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -98.1, physicsClientId=self.client)
        self.robot_id = p.loadURDF("shibot_inu.urdf", self.robot_start_pos, self.robot_start_ori,
                                   physicsClientId=self.client)
        return self.get_observation()

    def get_observation(self):
        observation = []
        for i in range(p.getNumJoints(self.robot_id)):
            joint_state = p.getJointState(self.robot_id, i, physicsClientId=self.client)
            observation.extend([joint_state[0], joint_state[1]])
        return np.array(observation, dtype=np.float32)

    def close(self):
        p.disconnect(self.client)

def make_env():
    def _init():
        env = ShibotEnv()
        return env
    return _init

env = DummyVecEnv([make_env() for _ in range(1)])
model = PPO('MlpPolicy', env, verbose=1)

callback = RewardLoggerCallback()

model.learn(total_timesteps=10, callback=callback)

model.save("ppo_shibot")
env.close()